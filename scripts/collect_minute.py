"""급등일 1분봉 수집 — 토스 `/api/v1/candles?interval=1m`.

## 왜 급등일만 모으는가

전 종목 전 기간 1분봉은 약 142,000호출이다. 그런데 **쓸 날이 4%뿐이다.** 장중
+10% 돌파는 종목-일의 4.1%에서만 일어나고, 어느 날인지는 **이미 가진 일봉으로
정확히 특정할 수 있다**(고가 ≥ 전일종가 × 1.10).

```
284종목 × 약 1,150거래일 = 326,600 종목-일
  → 트리거 4.1%          = 약 13,400 종목-일
  → 정규장 390봉 ÷ 200   = 2호출/일
  → 약 27,000호출 · 실측 9호출/초 = 약 50분
```

표본을 줄이는 타협이 아니다. **실거래에서도 트리거가 걸린 종목만 후보가 되므로,
학습 분포와 추론 분포가 일치한다.** 급등하지 않은 날의 분봉을 넣으면 모델이 "급등
여부"를 배우는데, 그건 일봉으로 이미 알고 있고 추론 시점에는 조건으로 고정된다.
우리가 배워야 하는 것은 **급등한 종목 중 어느 것이 계속 가는가**다.

실측 근거: 장중 +10% 돌파 중 **54.8%가 종가까지 밀린다.** 그 밀림을 걸러내면
30봉 `P(+100%)`가 3.27% → 4.42%로 오른다. 분봉의 가치는 싸게 사는 것(상대 6%)이
아니라 **밀릴 급등을 거르는 것**에 있다.

## 토스 분봉 API 사양 (실측 확인)

| 항목 | 값 |
|---|---|
| 지원 간격 | **`1m`만** — 3m·5m·15m·1h는 400 거부 |
| 호출당 최대 | **200봉** — 500 이상 거부, 응답에 커서 없음 |
| 과거 페이징 | `before`에 **ISO 타임스탬프**(`2026-09-23T12:00:00+09:00`) |
| 과거 한계 | **2022-01-03**까지. 2021-01-04는 빈 응답 |
| 포함 구간 | 08:41~20:00 — **장전·장후 시간외가 섞여 온다** |

마지막 항목이 중요하다. 시간외단일가는 유동성과 가격 형성이 정규장과 달라 학습에
넣으면 노이즈가 된다. **09:00~15:30만 남긴다.**

## 중단 복구

토스는 IP 화이트리스트를 쓰고 이 세션에서 IP가 **다섯 번** 바뀌었다. 그래서
종목 단위로 저장하고, 이미 받은 (종목, 날짜)는 건너뛴다. 중단 후 같은 명령을
다시 실행하면 이어서 받는다.

```bash
python scripts/collect_minute.py --probe          # 사양·속도만 확인
python scripts/collect_minute.py                  # 전 종목 급등일
python scripts/collect_minute.py --symbols 20     # 앞 20종목만
```
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.toss_data import TossDataClient, _rows_of  # noqa: E402
from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from utils.logger import enable_utf8_console, logger, setup_logging  # noqa: E402

MINUTE_DIR = "minute"
SESSION_OPEN = "0900"
SESSION_CLOSE = "1530"
BARS_PER_CALL = 200
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def surge_days(panel: Dict[str, pd.DataFrame], trigger: float,
               since: str = "", select: str = "price",
               burst_mult: float = 10.0) -> Dict[str, List[str]]:
    """분봉을 받을 (종목 → 날짜 목록)을 일봉에서 뽑는다.

    ## 선택 기준이 둘이고 집단이 다르다

    - `price`: **고가 ≥ 전일종가 × (1+trigger)**. 종가가 아니라 고가를 쓰는 이유는
      장중에 돌파했다가 밀린 날이 분별 대상이기 때문이다. 종가 기준으로 뽑으면
      성공한 급등만 모아 학습이 편향된다.
    - `turnover`: **거래대금 ≥ 20봉 중앙값 × burst_mult**. 꼬리 구조 전수 탐색에서
      승자 조건이 이쪽이었다(거래대금 10배 · 2종목 · 6봉 · 5회전 → P(+100%) 1.67%).
    - `both`: 두 집단의 합집합.

    **둘을 섞어 쓰면 커버리지 편향이 생긴다.** 분봉을 `price`(3%)로만 모았더니
    거래대금 10배 사건 4,902건 중 4,273건(87.2%)만 분봉이 있었다. 빠진 629건은
    "거래대금은 10배인데 가격이 3% 못 움직인 날" — 즉 **거래가 터졌는데 안 오른
    날**이므로 성질이 다르다. 그 상태로 재면 좋은 쪽만 본다.

    `turnover` 판정은 `strategy/catalyst_features.py`와 **같은 정의**를 쓴다
    (`turnover = 종가 × 거래량`, 20봉 **중앙값** 대비). 정의가 갈리면 학습·측정이
    보는 집단과 수집한 집단이 달라진다.
    """
    out: Dict[str, List[str]] = {}
    for sym, df in panel.items():
        if df is None or len(df) < 3:
            continue
        d = df.reset_index(drop=True)
        close = d["close"].to_numpy(float)
        high = d["high"].to_numpy(float)
        ts = d["ts"].astype(str).to_numpy()
        hit = np.zeros(len(d), dtype=bool)
        if select in ("price", "both"):
            prev = np.concatenate([[np.nan], close[:-1]])
            hit |= high >= prev * (1.0 + trigger)
        if select in ("turnover", "both"):
            turnover = pd.Series(close * d["volume"].to_numpy(float))
            med20 = turnover.rolling(20, min_periods=10).median()
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = (turnover / med20.replace(0.0, np.nan)).to_numpy(float)
            hit |= np.nan_to_num(ratio, nan=0.0) >= burst_mult
        days = [t for t, h in zip(ts, hit) if h and (not since or t >= since)]
        if days:
            out[sym] = days
    return out


def _iso(day: str, hhmm: str) -> str:
    """`YYYYMMDD` + `HHMM` → 토스가 받는 ISO 타임스탬프."""
    return (f"{day[:4]}-{day[4:6]}-{day[6:8]}"
            f"T{hhmm[:2]}:{hhmm[2:]}:00+09:00")


def _parse_rows(rows: List[dict]) -> pd.DataFrame:
    """토스 응답 → `ts(YYYYMMDDHHMM), open, high, low, close, volume`.

    `ts`를 문자열로 두는 것은 프로젝트 규약이다(일봉은 `YYYYMMDD`). 분봉은
    분까지 붙여 `YYYYMMDDHHMM` 12자리로 만들고 **과거→최신** 정렬을 지킨다.
    """
    recs = []
    for r in rows:
        t = str(r.get("timestamp") or "")
        if len(t) < 16:
            continue
        ts = t[0:4] + t[5:7] + t[8:10] + t[11:13] + t[14:16]
        try:
            recs.append((ts, float(r["openPrice"]), float(r["highPrice"]),
                         float(r["lowPrice"]), float(r["closePrice"]),
                         float(r.get("volume") or 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
    if not recs:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame(recs, columns=COLUMNS)
    return out.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """09:00~15:30만 남긴다. 장전(08:41~)·장후(~20:00) 시간외를 버린다."""
    if df.empty:
        return df
    hhmm = df["ts"].str[8:12]
    return df[(hhmm >= SESSION_OPEN) & (hhmm <= SESSION_CLOSE)].reset_index(drop=True)


async def fetch_day(cli: TossDataClient, symbol: str, day: str,
                    max_calls: int = 3) -> pd.DataFrame:
    """하루치 정규장 분봉. 15:31에서 뒤로 페이징한다.

    390봉이 필요하고 호출당 200봉이므로 2회가 기본이다. 장전 시간외가 섞여 오므로
    여유로 3회까지 허용하고, 09:00 이전까지 닿으면 멈춘다.
    """
    frames: List[pd.DataFrame] = []
    before = _iso(day, "1531")
    for _ in range(max_calls):
        payload = await cli._get(
            "/api/v1/candles",
            {"symbol": symbol, "interval": "1m", "count": BARS_PER_CALL,
             "before": before, "adjusted": "true"},
            cli._chart_bucket)
        got = _parse_rows(_rows_of(payload))
        if got.empty:
            break
        frames.append(got)
        oldest = got["ts"].iloc[0]
        # 09:00 이전에 닿았으면 그날의 정규장을 다 받은 것이다.
        if oldest[:8] < day or oldest[8:12] <= SESSION_OPEN:
            break
        before = _iso(oldest[:8], oldest[8:12])
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows[all_rows["ts"].str[:8] == day]
    return _regular_session(
        all_rows.sort_values("ts").drop_duplicates("ts").reset_index(drop=True))


def _load_existing(path: Path) -> Tuple[pd.DataFrame, Set[str]]:
    """이미 받은 분봉과 그 날짜들. 중단 복구의 근거다."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS), set()
    try:
        df = pd.read_csv(path, dtype={"ts": str})
        return df, set(df["ts"].str[:8].unique())
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[Minute] {path.name} 읽기 실패({exc}) — 새로 받습니다")
        return pd.DataFrame(columns=COLUMNS), set()


async def run(args) -> int:
    s = get_settings()
    setup_logging(s.LOG_DIR, s.LOG_LEVEL)
    symbols = list(s.UNIVERSE)
    if args.symbols:
        symbols = symbols[: args.symbols]

    panel = load_panel_from_dir(args.source, symbols)
    if not panel:
        logger.error(f"[Minute] {args.source} 에 일봉이 없습니다. 먼저 collect_data.py 를 돌리십시오")
        return 1
    targets = surge_days(panel, args.trigger, args.since,
                         select=args.select, burst_mult=args.burst_mult)
    total_days = sum(len(v) for v in targets.values())
    cond = (f"고가 +{args.trigger:.0%}" if args.select == "price"
            else f"거래대금 x{args.burst_mult:g}" if args.select == "turnover"
            else f"고가 +{args.trigger:.0%} 또는 거래대금 x{args.burst_mult:g}")
    logger.info(f"[Minute] 일봉 {len(panel)}종목 → 기준 '{cond}' "
                f"{len(targets)}종목 · {total_days:,}일 (예상 호출 {total_days * 2:,})")
    if args.probe:
        for sym in list(targets)[:2]:
            logger.info(f"[Minute] {sym}: {len(targets[sym])}일, 최근 {targets[sym][-3:]}")
        return 0

    out_dir = Path(args.source) / MINUTE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    done_days = 0
    saved_rows = 0
    skipped = 0
    failed = 0
    aborted = False
    async with TossDataClient(s) as cli:
        for i, (sym, days) in enumerate(sorted(targets.items()), 1):
            path = out_dir / f"{sym}.csv"
            have, have_days = _load_existing(path)
            todo = [d for d in days if d not in have_days]
            if not todo:
                # **진척 로그보다 먼저 continue 하지 않는다.** 그렇게 두면 받을
                # 것이 없는 종목 구간에서 로그가 완전히 조용해져, 14분 동안
                # 멈춘 것인지 도는 것인지 구분할 수 없었다(실제로 그랬다).
                skipped += len(days)
                if i % 20 == 0 or i == len(targets):
                    logger.info(f"[Minute] {i}/{len(targets)}종목 · 수집 {done_days:,}일 "
                                f"({saved_rows:,}봉) · 기존 {skipped:,}일 · 실패 {failed}")
                continue
            frames = [have] if not have.empty else []
            for day in todo:
                try:
                    got = await fetch_day(cli, sym, day)
                except Exception as exc:
                    failed += 1
                    logger.warning(f"[Minute] {sym} {day} 실패: {type(exc).__name__}: {str(exc)[:90]}")
                    # **403은 종목 문제가 아니라 전역 차단이다.** 종목 루프만 빠져
                    # 나가면 나머지 종목도 같은 403을 맞고, 루프만 돌다 "완료"로
                    # 끝난다(실제로 그렇게 끝나 19,887/88,275일만 받고 성공으로
                    # 보였다). 전체를 멈추고 0이 아닌 코드로 끝내야 한다.
                    if "IP address" in str(exc) or "403" in str(exc):
                        logger.error("[Minute] 접근 거부 — IP 화이트리스트를 확인하십시오. "
                                     "전체 수집을 중단합니다")
                        aborted = True
                        break
                    continue
                if got.empty:
                    continue
                frames.append(got)
                done_days += 1
                saved_rows += len(got)
            if frames:
                merged = (pd.concat(frames, ignore_index=True)
                          .sort_values("ts").drop_duplicates("ts").reset_index(drop=True))
                # 종목 단위로 즉시 기록한다. IP가 끊겨도 여기까지는 남는다.
                merged.to_csv(path, index=False)
            if i % 20 == 0 or i == len(targets):
                logger.info(f"[Minute] {i}/{len(targets)}종목 · 수집 {done_days:,}일 "
                            f"({saved_rows:,}봉) · 기존 {skipped:,}일 · 실패 {failed}")
            if aborted:
                break
    remaining = total_days - done_days - skipped
    if aborted:
        logger.error(f"[Minute] **중단** — {done_days:,}일 받고 약 {remaining:,}일 남음. "
                     f"IP를 등록한 뒤 같은 명령으로 이어받으십시오")
        return 2
    if failed:
        logger.warning(f"[Minute] 완료(실패 {failed}건) — {done_days:,}일 "
                       f"{saved_rows:,}봉 → {out_dir}")
    else:
        logger.success(f"[Minute] 완료 — {done_days:,}일 {saved_rows:,}봉 → {out_dir}")
    return 0


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="급등일 1분봉 수집")
    ap.add_argument("--source", default="data/cache", help="일봉 디렉토리(분봉은 그 아래 minute/)")
    ap.add_argument("--symbols", type=int, default=0, help="앞 N종목만(0=전체)")
    ap.add_argument("--trigger", type=float, default=0.03,
                    help="장중 돌파 트리거(고가/전일종가−1). 거래량 조건은 수집 단계에서 "
                         "판정할 수 없다(분봉이 있어야 안다) — 가격 조건으로 넓게 받고 "
                         "트리거 시점 판정은 strategy/minute_features.py 가 한다")
    ap.add_argument("--since", default="20220103", help="이 날짜 이후만(토스 한계 2022-01-03)")
    ap.add_argument("--select", choices=["price", "turnover", "both"], default="price",
                    help="수집 대상 선택 기준. price=고가 돌파 / "
                         "turnover=거래대금 20봉중앙값 x burst-mult / both=합집합. "
                         "기준이 다르면 집단이 달라 커버리지 편향이 생긴다")
    ap.add_argument("--burst-mult", type=float, default=10.0,
                    help="--select turnover/both 의 거래대금 폭증 배수")
    ap.add_argument("--probe", action="store_true", help="대상 규모만 보고 종료")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
