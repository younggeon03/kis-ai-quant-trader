"""분봉 리플레이 — 수집한 1분봉으로 장중 전략을 하루씩 재연한다.

## 왜 필요한가

분봉 전략의 핵심 세 가지가 **시각**에 걸려 있는데 Mock으로는 검증할 수 없다.
`main.py::_mock_epoch`(2025-01-02 09:00)에 **일 단위만** 더하므로 가상 시각이
항상 09:00이고, 그래서 아침 스캔(09:05)·마감 정리(15:15)·신선도의 경과분이
전부 발동하지 않거나 0으로 고정된다.

**마감 정리가 안 돌면 오버나이트를 들고 간다.** 그러면 분봉 전략이 아니라 스윙이
되고 측정한 분포와 완전히 달라진다. 지금 손절도 꺼져 있어(`INTRADAY_STOP_PCT=0.0`)
하방이 밤새 열린다. 10/12 장중에 처음 확인하는 것은 — 그날이 수익률 초기화 창이
열린 1주차 베팅 날이라 — 받아들일 수 없는 위험이다.

## 무엇을 검증하는가

**실제 `IntradayWatcher`를 그대로 구동한다.** 재구현하면 검증이 아니라 다른 코드를
테스트하는 것이 된다. 그래서 이 스크립트는 시세를 만들어 넣고 결과를 집계하는
일만 한다.

1. 트리거가 하루 한 번만 발동하는가 (실거래에서 중복 진입은 치명적이다)
2. 마감 정리가 실제로 모든 포지션을 닫는가 → **오버나이트 0건이어야 한다**
3. 신선도의 경과분·속도가 0이 아닌 값으로 분포하는가
4. 아침 스캔이 고른 종목에서 실제로 돌파가 나오는가
5. 그리고 **전략 자체의 성과** — 하루 2종목 × 42%로 30거래일 복리 분포

## 시세를 어떻게 만드는가

분봉 하나를 **고가 → 종가** 두 번의 틱으로 흘린다. 실거래는 틱이 연속으로 오므로
분 안에서 돌파선을 지나면 그 순간 발동하는데, 종가만 넣으면 분 안에서 찍고 되돌아온
돌파를 놓친다. 고가를 먼저 넣으면 그 경로가 재연된다.

누적거래량은 **그 분이 끝난 시점의 값**을 쓴다. 고가 시점의 실제 누적은 그보다
작으므로 리플레이가 실거래보다 **관대하다** — 이 점에서 리플레이는 백테스트
(`minute_exit_grid.py`, 분 전체 거래량으로 판정)와 같고 실거래가 더 엄격하다.

## 진입가

돌파 **다음 분의 시가**다. 백테스트와 같은 가정이고 실거래(감지 즉시, 루프 3초)
보다 불리하다. 실측 진입 슬리피지가 +0.785%였으므로 그 차이가 작지 않다.

```bash
python scripts/replay_intraday.py
python scripts/replay_intraday.py --days 200 --per-day 2 --weight 0.42
```
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.intraday import IntradayWatcher  # noqa: E402
from utils.logger import enable_utf8_console, logger, setup_logging  # noqa: E402


def _ts_to_dt(ts: str) -> dt.datetime:
    """`YYYYMMDDHHMM` → datetime."""
    return dt.datetime(int(ts[:4]), int(ts[4:6]), int(ts[6:8]),
                       int(ts[8:10]), int(ts[10:12]))


def load_minutes(minute_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        p = Path(minute_dir) / f"{sym}.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, dtype={"ts": str})
        except Exception:
            continue
        if not df.empty:
            out[sym] = df
    return out


def replay_day(day: str, day_bars: Dict[str, Dict[str, tuple]],
               prev: Dict[str, Tuple[float, float]],
               w: IntradayWatcher, settings,
               per_day: int, weight: float, cost: float,
               buy_band: float = 0.0,
               scores: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, object]:
    """하루를 재연한다. 반환은 그날의 거래 목록과 진단 수치.

    `day_bars`는 `{종목: {분ts: (open, high, low, close, cum)}}`로 **미리 잘려 있어야
    한다.** 여기서 DataFrame을 필터링하면 날짜마다 전 종목 전체 행을 훑어 O(일수 ×
    종목수 × 전체행수)가 된다(처음에 그렇게 만들어 10분이 지나도 끝나지 않았다).

    종목별 분봉을 분 단위로 **합쳐서** 시간 순서대로 흘린다. 종목마다 따로 돌리면
    "같은 시각에 여러 종목이 돌파했을 때 무엇을 고르는가"를 검증할 수 없다 —
    그 선택이 사실상 종목 선정이다.
    """
    w.reset_day(day)
    by_minute: Dict[str, List[Tuple[str, tuple]]] = {}
    for sym, rows in day_bars.items():
        pc, pv = prev.get(sym, (0.0, 0.0))
        if pc <= 0:
            continue
        w.prime(sym, pc, pv)
        for ts, bar in rows.items():
            by_minute.setdefault(ts, []).append((sym, bar))

    opens = day_bars
    minutes = sorted(by_minute)
    held: Dict[str, Dict[str, float]] = {}      # 종목 → {entry, qty_w}
    trades: List[Dict[str, object]] = []
    triggers = 0
    swaps = 0
    unfilled = 0
    diag_minutes: List[int] = []
    diag_speed: List[int] = []
    flatten_closes = 0

    for i, ts in enumerate(minutes):
        now = _ts_to_dt(ts)
        rows = by_minute[ts]
        # --- 시세 주입: 고가 → 종가 ---
        fired = []
        for sym, bar in rows:
            cum = bar[4]
            for px in (bar[1], bar[3]):
                sig = w.on_quote(sym, px, cum, now)
                if sig is not None:
                    fired.append(sig)
        triggers += len(fired)
        # 명부에 올린다. 매 분 비우면 "하루의 첫 2건"만 사게 된다 — 실거래
        # (`main.py`)와 같은 구조를 써야 측정이 실거래를 예측한다.
        for sig in fired:
            # 점수 공급자를 바꿔 끼운다. `scores`가 없으면 신선도 고정 공식이다.
            # 같은 날짜·같은 슬롯 제약에서 두 방식을 비교해야 답이 나온다 —
            # 기존 모델 측정은 "하루가 끝난 뒤 상위 2개"라 미래를 보고 골랐다.
            sc = sig.freshness if scores is None else scores.get((sig.symbol, day), 0.0)
            w.add_candidate(sig, sc)

        # --- 청산 (진입보다 먼저. 자리를 비워야 다음이 들어간다) ---
        for sym in list(held):
            bar = opens.get(sym, {}).get(ts)
            if bar is None:
                continue
            reason = w.exit_reason(sym, bar[3], now)
            if reason is None:
                continue
            pos = held.pop(sym)
            ret = bar[3] / pos["entry"] - 1.0 - cost
            trades.append({"day": day, "symbol": sym, "ret": ret, "reason": reason,
                           "exit_ts": ts})
            w.close_watch(sym)
            if "마감" in reason:
                flatten_closes += 1

        # --- 진입: 명부 상위부터, 보유 한도까지 ---
        if not w.should_flatten(now):
            nxt = minutes[i + 1] if i + 1 < len(minutes) else None
            free = max(0, per_day - len(held))
            for sig in w.pick_entries(now, free):
                if nxt is None or sig.symbol in held:
                    continue
                bar = opens.get(sig.symbol, {}).get(nxt)
                if bar is None:
                    continue
                entry = bar[0]
                if entry <= 0:
                    continue
                if buy_band > 0:
                    # --- 지정가 체결 판정 ---
                    #
                    # 실거래는 **지정가 매수**를 낸다(`main.py::_submit`,
                    # `BUY_LIMIT_BAND`). 가격이 지정가 위로 달아나면 **체결되지
                    # 않는다** — 실측 33,482 사건에서 2.0%에도 14.3%가 미체결이다.
                    # 모사하지 않으면 리플레이가 "고른 것은 반드시 산다"고 가정해
                    # 그만큼 낙관적이 된다.
                    #
                    # 지정가 기준은 **돌파선**이다(`전일종가 × (1+트리거)`). 실거래에서
                    # 돌파를 감지한 순간의 시세가 돌파선 근처이기 때문이다. 리플레이의
                    # `sig.price`는 그 분의 **고가**(먼저 주입한 값)라 그대로 쓰면
                    # 지정가가 실거래보다 높아져 체결이 과대평가된다.
                    limit = sig.prev_close * (1.0 + w.trigger_pct) * (1.0 + buy_band)
                    if bar[2] > limit:          # 다음 분 저가가 지정가 위 → 미체결
                        unfilled += 1
                        continue
                    entry = min(limit, bar[0])  # 시가가 더 싸면 시가에 체결
                sc = (sig.freshness if scores is None
                      else scores.get((sig.symbol, day), 0.0))
                held[sig.symbol] = {"entry": entry, "qty_w": weight, "score": sc}
                w.open_watch(sig.symbol, entry, sc)
                diag_minutes.append(sig.trig_minute)
                diag_speed.append(sig.speed)
            # --- 교체: 슬롯이 꽉 찼고 명부에 훨씬 좋은 후보가 있으면 ---
            if len(held) >= per_day:
                tgt = w.swap_target(now)
                if tgt is not None:
                    drop, incoming = tgt
                    bar = opens.get(drop, {}).get(ts)
                    if bar is not None and drop in held:
                        pos = held.pop(drop)
                        ret = bar[3] / pos["entry"] - 1.0 - cost
                        trades.append({"day": day, "symbol": drop, "ret": ret,
                                       "reason": f"교체({incoming.symbol})",
                                       "exit_ts": ts})
                        w.close_watch(drop)
                        w.note_swap(incoming)
                        swaps += 1

    # --- 장이 끝났는데 남아 있으면 오버나이트다. 있어서는 안 된다. ---
    overnight = []
    for sym, pos in held.items():
        rows = day_bars.get(sym) or {}
        px = rows[max(rows)][3] if rows else pos["entry"]
        trades.append({"day": day, "symbol": sym,
                       "ret": px / pos["entry"] - 1.0 - cost,
                       "reason": "오버나이트(미청산)", "exit_ts": ""})
        overnight.append(sym)
    return {"trades": trades, "triggers": triggers, "overnight": overnight,
            "flatten_closes": flatten_closes, "swaps": swaps,
            "unfilled": unfilled,
            "trig_minutes": diag_minutes, "speeds": diag_speed}


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="분봉 리플레이 검증")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--minute-dir", default="data/cache/minute")
    ap.add_argument("--days", type=int, default=0, help="최근 N거래일만(0=전체)")
    ap.add_argument("--per-day", type=int, default=2)
    ap.add_argument("--weight", type=float, default=0.42)
    ap.add_argument("--vol-mult", type=float, default=0.0,
                    help="돌파 봉 거래량 하한 배수를 덮어쓴다(0=설정값 사용). "
                         "장 시작 직후는 거래가 원래 많아(10분에 하루의 9.7%) "
                         "하한이 낮으면 09:00 노이즈가 전부 통과한다")
    ap.add_argument("--trigger", type=float, default=0.0,
                    help="돌파 기준을 덮어쓴다(0=설정값 사용)")
    ap.add_argument("--ttl", type=int, default=-1,
                    help="후보 명부 보관 분(−1=설정값). 0이면 그 분에 발동한 것만 본다")
    ap.add_argument("--select", default="freshness", choices=["freshness", "model"],
                    help="돌파 사건 선정 점수. freshness=고정 공식 / model=분봉 모델")
    ap.add_argument("--train-frac", type=float, default=0.4,
                    help="model 선택 시 앞쪽 몇 비율을 학습에 쓸지(나머지를 재연)")
    ap.add_argument("--no-fill-sim", action="store_true",
                    help="지정가 미체결을 모사하지 않는다(이전 동작, 비교용)")
    ap.add_argument("--out", default="reports/replay_intraday.json")
    args = ap.parse_args()

    s = get_settings()
    setup_logging(s.LOG_DIR, s.LOG_LEVEL)
    cost = s.round_trip_cost

    # --- 중복 실행 방지 ---
    #
    # 같은 리플레이가 두 번 돌아 **같은 출력 파일에 쓰는** 일이 있었다. 백그라운드
    # 체인을 `nohup`으로 띄웠더니 래퍼의 종료 코드가 1로 보고돼 "실패"로 판단하고
    # 같은 작업을 다시 시작했는데, 실제로는 자식 프로세스가 살아 있었다. 두
    # 프로세스가 각각 1.8GB를 써서 가용 메모리가 0.7GB까지 떨어졌고, 결과 파일도
    # 서로 덮어썼다.
    #
    # 출력 경로 기준으로 잠금 파일을 둔다. 리플레이는 몇 분이면 끝나므로 오래된
    # 잠금은 무시한다(비정상 종료로 남은 것).
    lock = Path(args.out).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < 3600:
            logger.error(f"[Replay] 이미 실행 중으로 보입니다 — {lock} "
                         f"({age / 60:.0f}분 전 생성). 중복 실행을 막습니다. "
                         f"비정상 종료였다면 그 파일을 지우십시오")
            return 3
        logger.warning(f"[Replay] 오래된 잠금({age / 3600:.1f}시간) 무시: {lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    syms = sorted(p.stem for p in Path(args.minute_dir).glob("*.csv"))
    if not syms:
        logger.error(f"[Replay] {args.minute_dir} 에 분봉이 없습니다")
        return 1
    bars = load_minutes(args.minute_dir, syms)
    daily = load_panel_from_dir(args.source, list(bars))
    logger.info(f"[Replay] 분봉 {len(bars)}종목 · 트리거 "
                f"+{(args.trigger or s.INTRADAY_TRIGGER_PCT):.1%} · 거래량 x"
                f"{(args.vol_mult or s.INTRADAY_VOL_MULT)} · 마감 {s.INTRADAY_FLATTEN_HHMM}")

    # 전일 종가·거래량. 일봉에서 가져온다 — 분봉만으로는 '전일 대비'를 모른다.
    prev_map: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for sym, d in daily.items():
        dd = d.reset_index(drop=True)
        ts = dd["ts"].astype(str).tolist()
        c = dd["close"].astype(float).tolist()
        v = dd["volume"].astype(float).tolist()
        prev_map[sym] = {ts[i]: (c[i - 1], v[i - 1]) for i in range(1, len(ts))}

    # 누적거래량 컬럼을 미리 만든다(감시기가 차분으로 분당 거래량을 만든다).
    for sym, df in bars.items():
        df["cum"] = df.groupby(df["ts"].str[:8])["volume"].cumsum()

    days = sorted({t[:8] for df in bars.values() for t in df["ts"]})
    if args.days:
        days = days[-args.days:]
    logger.info(f"[Replay] {len(days)}거래일 재연 ({days[0]}~{days[-1]})")

    # 종목별로 **정렬된 배열 하나**만 들고, 날짜 구간은 이분탐색으로 잘라 쓴다.
    #
    # 처음에는 날짜 루프 안에서 DataFrame을 필터링해 O(일수 x 종목수 x 전체행수)가
    # 되어 10분이 지나도 끝나지 않았다. 그래서 전 날짜를 미리 `{분ts: 튜플}` 사전으로
    # 분할했는데, 985MB 분봉에서 **3.5GB를 쓰게 되어** 메모리 부족으로 다른 작업이
    # 죽었다(파이썬 사전·튜플 객체 오버헤드가 원본의 몇 배다). 수집이 완주해 282종목이
    # 되면 아예 돌지 않는다.
    #
    # 이분탐색은 둘 다 피한다 — 날짜당 종목당 O(log n)이고 메모리는 배열뿐이다.
    import bisect

    index: Dict[str, tuple] = {}
    for sym, df in bars.items():
        d = df.sort_values("ts")
        index[sym] = (d["ts"].tolist(),
                      d["open"].to_numpy(float), d["high"].to_numpy(float),
                      d["low"].to_numpy(float), d["close"].to_numpy(float),
                      d["cum"].to_numpy(float))
    bars.clear()          # DataFrame을 놓아준다
    logger.info(f"[Replay] 색인 완료 ({len(index)}종목)")

    # --- 데이터 지문 ---
    # **이걸 빼먹어 같은 코드가 +0.557%와 −0.442%로 갈렸다.** 수집이 계속 돌아
    # 같은 종목의 추가 급등일이 들어오면 날짜별 후보 풀이 달라져 선택이 바뀐다.
    # 두 실행 모두 "178종목 · 20250917~20260929"로 보였기 때문에 코드 차이를
    # 몇 시간 뒤졌다. `scripts/ablate_exits.py`가 같은 이유로 지문을 찍는데
    # 리플레이에 넣지 않아 같은 함정을 다시 밟았다.
    import hashlib

    h = hashlib.md5()
    n_rows = 0
    for sym in sorted(index):
        ts, _o, _h, _l, c, _cum = index[sym]
        h.update(sym.encode())
        h.update(np.asarray(c, dtype=np.float64).tobytes())
        n_rows += len(ts)
    fingerprint = h.hexdigest()[:16]
    logger.warning(f"[Replay] 데이터 지문 {fingerprint} "
                   f"({len(index)}종목 · {n_rows:,}봉) — 지문이 다른 실행과는 비교할 수 없다")

    def day_slice(day: str) -> Dict[str, Dict[str, tuple]]:
        """그날의 `{종목: {분ts: (o,h,l,c,cum)}}`. 호출 뒤 버려지므로 메모리가 누적되지 않는다."""
        out: Dict[str, Dict[str, tuple]] = {}
        lo_key, hi_key = day + "0000", day + "9999"
        for sym, (ts, o, h, l, c, cum) in index.items():
            a = bisect.bisect_left(ts, lo_key)
            b = bisect.bisect_right(ts, hi_key)
            if a >= b:
                continue
            out[sym] = {ts[k]: (o[k], h[k], l[k], c[k], cum[k]) for k in range(a, b)}
        return out

    trig = args.trigger or s.INTRADAY_TRIGGER_PCT
    vmult = args.vol_mult or s.INTRADAY_VOL_MULT
    # --- 모델 점수(선택 시) ---
    #
    # 앞쪽 `train_frac`을 학습에, 나머지를 재연에 쓴다. **학습 구간은 재연 구간보다
    # 앞선다** — 섞이면 룩어헤드다.
    #
    # 점수를 미리 (종목, 날짜)로 계산해 둔다. 트리거는 데이터와 설정이 같으면
    # 결정적이므로 재연 중 발동하는 사건과 1:1로 맞는다. 루프 안에서 피처를
    # 다시 만들면 느리고, 무엇보다 **감시기에 피처 계산을 넣지 않기 위해서다**
    # (순수 로직 유지).
    model_scores: Optional[Dict[Tuple[str, str], float]] = None
    if args.select == "model":
        from scripts.minute_skill import build_events
        from strategy.minute_features import MINUTE_FEATURES

        cut = days[max(1, int(len(days) * args.train_frac)) - 1]
        logger.info(f"[Replay] 모델 선택 — 학습 ~{cut} / 재연 {cut}~ "
                    f"(학습 비율 {args.train_frac:.0%})")
        ev = build_events(daily, args.minute_dir, trig, "close",
                          200, vmult)
        if ev.empty:
            logger.error("[Replay] 사건이 없어 모델을 학습할 수 없습니다")
            return 1
        ev["ret"] = ev["ret"] - cost
        feats = [c for c in MINUTE_FEATURES if c in ev.columns]
        tr = ev[ev["day"] <= cut]
        te = ev[ev["day"] > cut]
        if len(tr) < 500 or te.empty:
            logger.error(f"[Replay] 학습 {len(tr)}건 · 재연 {len(te)}건 — 부족합니다")
            return 1
        thr = float(tr["ret"].quantile(0.75))
        y = (tr["ret"] >= thr).astype(int).to_numpy()
        try:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=60,
                                   subsample=0.8, colsample_bytree=0.8,
                                   reg_lambda=1.0, verbose=-1, random_state=16)
            m.fit(tr[feats].to_numpy(float), y)
            p_te = m.predict_proba(te[feats].to_numpy(float))[:, 1]
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                               max_leaf_nodes=15, random_state=16)
            m.fit(np.nan_to_num(tr[feats].to_numpy(float)), y)
            p_te = m.predict_proba(np.nan_to_num(te[feats].to_numpy(float)))[:, 1]
        model_scores = {(r.symbol, r.day): float(p)
                        for r, p in zip(te.itertuples(), p_te)}
        logger.info(f"[Replay] 학습 {len(tr):,}건(문턱 {thr * 100:+.2f}% · "
                    f"양성 {y.mean():.1%}) → 재연 점수 {len(model_scores):,}건")
        # 재연 구간만 돌린다. 학습 구간을 재연하면 in-sample이다.
        days = [d for d in days if d > cut]

    w = IntradayWatcher(
        trigger_pct=trig, vol_mult=vmult,
        target_pct=s.INTRADAY_TARGET_PCT, stop_pct=s.INTRADAY_STOP_PCT,
        trail_pct=s.INTRADAY_TRAIL_PCT,
        trail_activate_pct=s.INTRADAY_TRAIL_ACTIVATE_PCT,
        flatten_hhmm=s.INTRADAY_FLATTEN_HHMM,
        candidate_ttl=(args.ttl if args.ttl >= 0 else s.INTRADAY_CANDIDATE_TTL),
        swap_margin=s.INTRADAY_SWAP_MARGIN,
        max_swaps=s.INTRADAY_MAX_SWAPS)

    all_trades: List[Dict[str, object]] = []
    tot_trig = 0
    overnight: List[str] = []
    flatten_closes = 0
    tot_swaps = 0
    tot_unfilled = 0
    tmins: List[int] = []
    speeds: List[int] = []
    for k, day in enumerate(days, 1):
        # `bars`가 아니라 `index`를 돈다 — 색인을 만든 뒤 `bars`를 비웠다.
        # 여기서 `bars`를 쓰면 전일 종가가 전부 0이 되어 트리거가 하나도 안 걸린다
        # (실제로 그렇게 깨져 거래 0건이 나왔다).
        prev = {sym: prev_map.get(sym, {}).get(day, (0.0, 0.0)) for sym in index}
        r = replay_day(day, day_slice(day), prev, w, s,
                       args.per_day, args.weight, cost,
                       buy_band=(0.0 if args.no_fill_sim else s.BUY_LIMIT_BAND),
                       scores=model_scores)
        all_trades.extend(r["trades"])          # type: ignore[arg-type]
        tot_trig += int(r["triggers"])          # type: ignore[arg-type]
        overnight.extend(r["overnight"])        # type: ignore[arg-type]
        flatten_closes += int(r["flatten_closes"])  # type: ignore[arg-type]
        tot_swaps += int(r["swaps"])            # type: ignore[arg-type]
        tot_unfilled += int(r["unfilled"])      # type: ignore[arg-type]
        tmins.extend(r["trig_minutes"])         # type: ignore[arg-type]
        speeds.extend(r["speeds"])              # type: ignore[arg-type]
        if k % 50 == 0:
            logger.info(f"[Replay] {k}/{len(days)}일 · 거래 {len(all_trades):,}건")

    print()
    print("=" * 100)
    print(f" 분봉 리플레이 — {len(index)}종목 · {len(days)}거래일 · "
          f"하루 {args.per_day}종목 x {args.weight:.0%} · 왕복비용 {cost:.3%}")
    print(f" 데이터 지문 {fingerprint} ({n_rows:,}봉) — 지문이 다른 실행과는 비교할 수 없다")
    print("=" * 100)
    print(" [시각 기반 로직 검증]")
    print(f"  돌파 발동          {tot_trig:,}건  (하루 평균 {tot_trig / max(1, len(days)):.1f})")
    print(f"  진입               {len(tmins):,}건")
    print(f"  마감 정리로 청산    {flatten_closes:,}건")
    print(f"  교체               {tot_swaps:,}건  "
          f"(하루 한도 {s.INTRADAY_MAX_SWAPS}회 · 여유 {s.INTRADAY_SWAP_MARGIN})")
    if args.no_fill_sim:
        print(f"  지정가 미체결       모사 안 함 (--no-fill-sim) — 결과가 낙관적이다")
    else:
        tot_try = len(tmins) + tot_unfilled
        print(f"  지정가 미체결       {tot_unfilled:,}건 / 시도 {tot_try:,}건 "
              f"({tot_unfilled / max(1, tot_try):.1%}) · 매수 폭 {s.BUY_LIMIT_BAND:.1%}")
    print(f"  **오버나이트 잔여   {len(overnight):,}건**  "
          f"{'→ 정상(0건이어야 한다)' if not overnight else '→ 버그! 마감 정리가 동작하지 않는다'}")
    if tmins:
        print(f"  돌파 경과분 분포    중앙 {int(np.median(tmins))}분 · "
              f"p10 {int(np.percentile(tmins, 10))}분 · p90 {int(np.percentile(tmins, 90))}분"
              f"  {'→ 정상(0 고정이 아니다)' if np.ptp(tmins) > 0 else '→ 버그! 0으로 고정'}")
        print(f"  급등 속도 분포      중앙 {int(np.median(speeds))}분 · "
              f"p90 {int(np.percentile(speeds, 90))}분")
    print("-" * 100)

    rets = np.array([t["ret"] for t in all_trades], dtype=float)
    if len(rets):
        print(" [전략 성과]")
        print(f"  거래 {len(rets):,}건 · 승률 {(rets > 0).mean():.1%} · "
              f"평균/회 {rets.mean() * 100:+.3f}% · 중앙/회 {np.median(rets) * 100:+.3f}% · "
              f"표준편차 {rets.std() * 100:.2f}%")
        reasons: Dict[str, int] = {}
        for t in all_trades:
            key = str(t["reason"]).split()[0]
            reasons[key] = reasons.get(key, 0) + 1
        print("  청산 사유: " + " · ".join(f"{k} {v:,}" for k, v in
                                        sorted(reasons.items(), key=lambda x: -x[1])))
        rng = np.random.default_rng(16)
        draws = rng.choice(rets, size=(40000, 30, args.per_day), replace=True)
        eq = np.prod(1.0 + args.weight * draws.sum(axis=2), axis=1) - 1.0
        print(f"  30거래일 복리: 중앙 {np.median(eq) * 100:+.1f}% · "
              f"P(+50%) {(eq >= 0.5).mean() * 100:.2f}% · "
              f"P(+100%) {(eq >= 1.0).mean() * 100:.2f}% · "
              f"P(-50%) {(eq <= -0.5).mean() * 100:.2f}%")
    print("=" * 100)
    print(" 리플레이는 분 전체 거래량으로 판정하므로 실거래(분 중간 누적)보다 관대하다.")
    print(" **매도 미체결은 모사하지 않는다** — 실측 체결률 99.71%로 영향이 미미하다.")
    print(" 진입가는 다음 분 시가로 실거래(감지 즉시)보다 불리하다. 두 차이가 상쇄되지 않는다.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fingerprint": fingerprint, "rows": n_rows,
        "config": {"symbols": len(index), "days": len(days), "per_day": args.per_day,
                   "weight": args.weight, "cost": cost,
                   "trigger": trig, "vol_mult": vmult,
                   "flatten": s.INTRADAY_FLATTEN_HHMM},
        "validation": {"triggers": tot_trig, "entries": len(tmins),
                       "flatten_closes": flatten_closes,
                       "overnight": len(overnight),
                       "unfilled": tot_unfilled,
                       "fill_sim": not args.no_fill_sim},
        "performance": ({"n": int(len(rets)), "win": float((rets > 0).mean()),
                         "mean": float(rets.mean()), "std": float(rets.std())}
                        if len(rets) else {}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"[Replay] 저장 → {out.resolve()}")
    lock.unlink(missing_ok=True)
    return 0 if not overnight else 2


if __name__ == "__main__":
    raise SystemExit(main())
