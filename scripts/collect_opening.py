"""장 시작 구간 1분봉 수집 — **모든 거래일**의 09:00~09:30.

## 왜 따로 모으는가

`collect_minute.py`는 급등일·폭증일만 받는다(종목당 전체 거래일의 15~25%). 그 날들은
**나중에 급등한 것으로 판명된 날**이라, "09:05에 신호가 났는데 결국 별일 없던 날"이
데이터에 없다. 장 시작 구간 규칙을 그 데이터로 재면 미래를 보고 고른 날만 보게 되고,
"지난 20일 같은 시간대" 기준선도 만들 수 없다.

장 시작 30분만이면 종목-일당 호출이 **1번**이다(`before=09:31`에서 200봉 뒤로 —
08:41~ 장전 시간외와 전날 장후가 섞여 오므로 그날 09:00~09:30만 남긴다).

## 저장 위치

`data/cache/opening/{종목}.csv` — `minute/`와 **섞지 않는다.** 다른 스크립트가
`minute/`를 "그날 하루 전체 분봉"으로 가정하므로, 30분짜리 날이 섞이면 조용히 틀린다.
이미 `minute/`에 하루 전체가 있는 날은 호출하지 않고 거기서 잘라 쓴다.

## 중단 복구

종목 단위로 저장하고, 받은 날은 건너뛴다. 403(IP)이면 전체를 멈추고 종료코드 2.

```bash
python scripts/collect_opening.py --probe          # 대상 규모만
python scripts/collect_opening.py --symbols 1      # 접근 확인(1종목)
python scripts/collect_opening.py                  # 2025-01-02~ 전 종목
```
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.toss_data import TossDataClient, _rows_of  # noqa: E402
from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.collect_minute import (BARS_PER_CALL, COLUMNS, MINUTE_DIR,  # noqa: E402
                                    _iso, _parse_rows, fetch_day)
from utils.krx_calendar import KRXCalendar  # noqa: E402
from utils.logger import enable_utf8_console, logger, setup_logging  # noqa: E402

OPENING_DIR = "opening"


def _window(df: pd.DataFrame, day: str, until: str) -> pd.DataFrame:
    if df.empty:
        return df
    hhmm = df["ts"].str[8:12]
    m = (df["ts"].str[:8] == day) & (hhmm >= "0900") & (hhmm <= until)
    return df[m].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


async def fetch_opening(cli: TossDataClient, symbol: str, day: str,
                        until: str) -> pd.DataFrame:
    end = f"{int(until[:2]):02d}{int(until[2:]) + 1:02d}" if until[2:] != "59" \
        else f"{int(until[:2]) + 1:02d}00"
    payload = await cli._get(
        "/api/v1/candles",
        {"symbol": symbol, "interval": "1m", "count": BARS_PER_CALL,
         "before": _iso(day, end), "adjusted": "true"},
        cli._chart_bucket)
    return _window(_parse_rows(_rows_of(payload)), day, until)


def _load(path: Path) -> Tuple[pd.DataFrame, Set[str]]:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS), set()
    try:
        df = pd.read_csv(path, dtype={"ts": str})
        return df, set(df["ts"].str[:8].unique())
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[Opening] {path.name} 읽기 실패({exc}) — 새로 받습니다")
        return pd.DataFrame(columns=COLUMNS), set()


async def run(args) -> int:
    s = get_settings()
    setup_logging(s.LOG_DIR, s.LOG_LEVEL)
    symbols = list(s.UNIVERSE)
    if args.symbols:
        symbols = symbols[: args.symbols]
    panel = load_panel_from_dir(args.source, symbols)
    if not panel:
        logger.error(f"[Opening] {args.source} 에 일봉이 없습니다")
        return 1
    base = Path(args.source)
    out_dir = base / args.dir
    whole_day = args.until >= "1530"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 대상 = 일봉이 있는 모든 거래일(since 이후). 일봉 날짜가 곧 거래일 달력이다.
    # **일봉 파일이 끝난 뒤의 날짜는 KRX 거래일 달력으로 채운다.** 일봉 캐시를 다시
    # 받지 않았으면(9/29에서 끝난다) 그 뒤 날짜를 몰라 "호출 0"으로 조용히 끝났다.
    cal = KRXCalendar()
    yesterday = dt.date.today() - dt.timedelta(days=1)

    def _extra(last: str) -> List[str]:
        d = dt.datetime.strptime(last, "%Y%m%d").date() + dt.timedelta(days=1)
        out: List[str] = []
        while d <= yesterday:
            if cal.is_trading_day(d):
                out.append(d.strftime("%Y%m%d"))
            d += dt.timedelta(days=1)
        return out

    targets = {}
    for sym, df in panel.items():
        if df is None or df.empty:
            continue
        days = [t for t in df["ts"].astype(str) if t >= args.since]
        targets[sym] = days + [t for t in _extra(str(df["ts"].astype(str).iloc[-1]))
                               if t >= args.since]
    total = sum(len(v) for v in targets.values())
    logger.info(f"[Opening] {len(targets)}종목 · {total:,}종목-일 (since {args.since}, "
                f"~{args.until}) — 기존 분봉에 있는 날은 호출하지 않는다")
    if args.probe:
        return 0

    fetched = reused = skipped = empty = failed = 0
    aborted = False
    async with TossDataClient(s) as cli:
        for i, (sym, days) in enumerate(sorted(targets.items()), 1):
            path = out_dir / f"{sym}.csv"
            have, have_days = _load(path)
            todo = [d for d in days if d not in have_days]
            skipped += len(days) - len(todo)
            frames: List[pd.DataFrame] = [have] if not have.empty else []
            if todo:
                full, full_days = _load(base / MINUTE_DIR / f"{sym}.csv")
                need: List[str] = []
                for day in todo:
                    if day in full_days:
                        w = _window(full, day, args.until)
                        if not w.empty:
                            frames.append(w)
                            reused += 1
                            continue
                    need.append(day)

                async def _one(day: str) -> pd.DataFrame:
                    # 하루 전체(--until 1530)는 15:31에서 뒤로 2~3회 페이징한다
                    # (collect_minute.fetch_day 그대로). 장 시작 구간은 1회.
                    return (await fetch_day(cli, sym, day) if whole_day
                            else await fetch_opening(cli, sym, day, args.until))

                # **동시에 요청한다.** 순서대로 보내면 응답 왕복이 병목이라 한도(TOSS_CHART_TPS)
                # 보다 훨씬 낮게(초당 약 5.5건) 돈다. 한도는 토큰 버킷이 지키고, 넘으면 429
                # 백오프가 처리한다.
                step = max(1, int(args.concurrency))
                for k in range(0, len(need), step):
                    batch = need[k:k + step]
                    results = await asyncio.gather(*[_one(d) for d in batch],
                                                   return_exceptions=True)
                    for day, got in zip(batch, results):
                        if isinstance(got, Exception):
                            failed += 1
                            logger.warning(f"[Opening] {sym} {day} 실패: "
                                           f"{type(got).__name__}: {str(got)[:90]}")
                            if "IP address" in str(got) or "403" in str(got):
                                aborted = True
                            continue
                        if got.empty:
                            empty += 1
                            continue
                        frames.append(got)
                        fetched += 1
                    if aborted:
                        logger.error("[Opening] 접근 거부 — IP 화이트리스트를 확인하십시오. "
                                     "전체 수집을 중단합니다(받은 것은 저장한다)")
                        break
                if frames:
                    merged = (pd.concat(frames, ignore_index=True)
                              .sort_values("ts").drop_duplicates("ts").reset_index(drop=True))
                    merged.to_csv(path, index=False)
            if i % 10 == 0 or i == len(targets) or aborted:
                logger.info(f"[Opening] {i}/{len(targets)}종목 · 호출 {fetched:,}일 · "
                            f"재사용 {reused:,} · 기존 {skipped:,} · 빈응답 {empty:,} · 실패 {failed}")
            if aborted:
                break
    if aborted:
        logger.error(f"[Opening] **중단** — IP 등록 후 같은 명령으로 이어받으십시오")
        return 2
    logger.success(f"[Opening] 완료 — 호출 {fetched:,} · 재사용 {reused:,} · 기존 {skipped:,} · "
                   f"빈응답 {empty:,} · 실패 {failed} → {out_dir}")
    return 0


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="장 시작 구간 1분봉 수집(모든 거래일)")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--symbols", type=int, default=0, help="앞 N종목만(0=전체)")
    ap.add_argument("--since", default="20250102")
    ap.add_argument("--until", default="0930",
                    help="받을 구간 끝(HHMM, 포함). 1530이면 하루 전체(종목-일당 2~3호출)")
    ap.add_argument("--dir", default=OPENING_DIR,
                    help="저장 폴더(source 아래). 하루 전체는 fullday — minute/(급등일 모음)와 "
                         "섞지 않는다: 다른 측정 스크립트가 그 폴더를 '급등일 집단'으로 쓴다")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="동시 요청 날짜 수(유량 한도는 TOSS_CHART_TPS 토큰 버킷이 지킨다)")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
