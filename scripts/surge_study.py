"""급등 포착 연구 — 소재 흔적이 나타난 뒤에 들어가도 돈이 되는가.

## 이 스크립트가 답하는 질문

대회 목표는 30거래일 절대수익 top10(지난 대회 +100%)이다. 그 수익은 급등주에서만
나오고, 급등은 소재로 일어난다. 그래서 전략의 핵심은 **급등을 얼마나 빨리 잡는가**다.

그런데 `strategy/catalyst_features.py`의 흔적 피처는 모두 **과거를 본다** — 거래대금이
이미 터졌고, 이미 급등했고, 테마가 이미 함께 올랐다. 그 뒤에 들어가는 것이 고점 매수인지
추세 진입인지는 **가정이 아니라 측정으로** 가려야 한다.

모델이 필요 없다. 조건을 걸고 **그 다음 30봉 수익률 분포**를 보면 된다. 비교 기준은
같은 데이터의 무조건 분포다.

## 무엇을 보고 판정하는가

`scripts/ablate_exits.py`와 같은 원칙이다 — 평균이 아니라 **꼬리**를 본다.
심사가 계단함수이므로 중앙값이 낮아도 `P(+100%)`가 높으면 쓸 만하다. 반대로
중앙값이 높고 꼬리가 얇으면 중위권으로 끝난다.

하방도 같이 찍는다. 꼬리를 사는 대가가 얼마인지 알아야 "전면 해제"가 합리적 선택인지
판단할 수 있다.

```bash
python scripts/surge_study.py                        # 284종목 전체
python scripts/surge_study.py --sample 120 --step 5  # 빠르게
```

## 한계

유니버스가 2026-09-29 시점의 거래대금 30억 이상으로 만들어졌으므로 **생존 편향**이
있다. "많이 올라서 지금 유동성이 있는 종목"이 들어 있어, 과거를 돌아보면 그 상승
구간이 보인다. 조건부와 무조건부를 **같은 유니버스에서** 비교하므로 두 분포의 차이는
편향에 덜 민감하지만, 절대 수준은 부풀려진 값으로 읽어야 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.catalyst_features import (  # noqa: E402
    add_theme_features,
    build_catalyst_features,
)
from utils.logger import enable_utf8_console, logger  # noqa: E402

# 흔적 피처는 log1p(배수)다. 배수를 문턱으로 쓰려면 같은 변환을 거쳐야 한다.
B3 = float(np.log1p(3.0))
B5 = float(np.log1p(5.0))
B10 = float(np.log1p(10.0))

# (이름, 조건 함수) — 조건은 봉 t까지의 정보만 쓴다.
TRIGGERS: List[Tuple[str, Callable[[pd.DataFrame], np.ndarray]]] = [
    ("무조건 (기준선)", lambda d: np.ones(len(d), dtype=bool)),
    ("거래대금 3배", lambda d: d["turnover_burst_20"].to_numpy() >= B3),
    ("거래대금 5배", lambda d: d["turnover_burst_20"].to_numpy() >= B5),
    ("거래대금 10배", lambda d: d["turnover_burst_20"].to_numpy() >= B10),
    ("일간 +10%", lambda d: d["ret_1"].to_numpy() >= 0.10),
    ("일간 +15%", lambda d: d["ret_1"].to_numpy() >= 0.15),
    ("급등+폭증 동시", lambda d: (d["ret_1"].to_numpy() >= 0.10)
                              & (d["turnover_burst_20"].to_numpy() >= B3)),
    ("신고가+폭증", lambda d: (d["high_60_break"].to_numpy() >= 0.5)
                            & (d["turnover_burst_20"].to_numpy() >= B3)),
    # 늦게 들어가도 되는지 — "빠르게 캐치"가 얼마나 빨라야 하는가
    ("급등 1~3봉 경과", lambda d: (d["bars_since_surge"].to_numpy() >= 1)
                                & (d["bars_since_surge"].to_numpy() <= 3)),
    ("급등 4~10봉 경과", lambda d: (d["bars_since_surge"].to_numpy() >= 4)
                                 & (d["bars_since_surge"].to_numpy() <= 10)),
    ("급등 11~30봉 경과", lambda d: (d["bars_since_surge"].to_numpy() >= 11)
                                  & (d["bars_since_surge"].to_numpy() <= 30)),
    # 테마 동반 상승 — 소재는 떼로 움직인다
    ("테마 동반(breadth≥0.2)", lambda d: d["theme_breadth"].to_numpy() >= 0.20),
    ("테마 동반 + 폭증", lambda d: (d["theme_breadth"].to_numpy() >= 0.20)
                                & (d["turnover_burst_20"].to_numpy() >= B3)),
]


def forward_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """봉 t+1 시가가 아니라 종가 기준 선행 수익률. 조건의 변별력만 보는 연구이므로
    체결 가정(백테스트는 t+1 시가)은 넣지 않는다. 절대 수준은 그만큼 낙관적이다."""
    out = np.full(len(close), np.nan)
    if len(close) > horizon:
        out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return out


def describe(fwd: np.ndarray, label: str, base_n: int) -> Dict[str, float]:
    r = fwd[np.isfinite(fwd)]
    if not len(r):
        return {}
    return {
        "n": len(r),
        "share": len(r) / max(1, base_n),
        "median": float(np.median(r)),
        "p90": float(np.percentile(r, 90)),
        "p99": float(np.percentile(r, 99)),
        "max": float(r.max()),
        "p50up": float((r >= 0.50).mean()),
        "p100up": float((r >= 1.00).mean()),
        "dn30": float((r <= -0.30).mean()),
        "mean": float(r.mean()),
    }


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="급등 포착 조건 연구")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--sample", type=int, default=0, help="종목 샘플 수(0=전체)")
    ap.add_argument("--horizon", type=int, default=30, help="선행 수익률 구간(봉)")
    ap.add_argument("--step", type=int, default=1, help="봉 샘플링 간격")
    ap.add_argument("--warmup", type=int, default=300, help="앞쪽 제외 봉 수")
    args = ap.parse_args()

    s = get_settings()
    symbols = list(s.UNIVERSE)
    if args.sample and args.sample < len(symbols):
        rng = np.random.default_rng(16)
        symbols = sorted(rng.choice(symbols, size=args.sample, replace=False).tolist())

    panel = load_panel_from_dir(args.source, symbols)
    if not panel:
        logger.error(f"[Surge] {args.source} 에서 데이터를 찾지 못했습니다")
        return 1
    logger.info(f"[Surge] {len(panel)}종목 로드 · 선행 {args.horizon}봉")

    # 흔적 피처. ret_1은 여기서 직접 만든다(build_features 전체를 돌릴 필요가 없다).
    featured: Dict[str, pd.DataFrame] = {}
    for sym, df in panel.items():
        if df is None or len(df) < args.warmup + args.horizon + 10:
            continue
        d = df.reset_index(drop=True).copy()
        cat = build_catalyst_features(d)
        for c in cat.columns:
            d[c] = cat[c].to_numpy()
        d["ret_1"] = d["close"].astype(float).pct_change()
        d["ret_5"] = d["close"].astype(float).pct_change(5)
        featured[sym] = d
    logger.info(f"[Surge] 흔적 피처 {len(featured)}종목")

    featured = add_theme_features(featured, panel)
    logger.info("[Surge] 테마 피처 완료")

    # 조건별로 전 종목·전 시점을 모은다.
    pooled: Dict[str, List[np.ndarray]] = {n: [] for n, _ in TRIGGERS}
    base_n = 0
    for sym, d in featured.items():
        close = d["close"].to_numpy(float)
        fwd = forward_return(close, args.horizon)
        keep = np.zeros(len(d), dtype=bool)
        keep[args.warmup::args.step] = True
        keep &= np.isfinite(fwd)
        base_n += int(keep.sum())
        for name, fn in TRIGGERS:
            try:
                mask = np.asarray(fn(d), dtype=bool) & keep
            except KeyError:
                continue
            if mask.any():
                pooled[name].append(fwd[mask])

    print()
    print("=" * 108)
    print(f" 급등 포착 연구 — {len(featured)}종목 · 선행 {args.horizon}봉 · 표본 {base_n:,}개")
    print(" 조건이 나타난 **다음** 30봉 수익률 분포. 종가 기준이므로 실제보다 낙관적이다.")
    print("=" * 108)
    print(f" {'조건':24} {'표본':>8} {'빈도':>6} {'중앙값':>8} {'p90':>8} {'p99':>9} "
          f"{'최대':>9} {'+50%':>6} {'+100%':>6} {'-30%':>6}")
    print("-" * 108)
    rows = []
    for name, _ in TRIGGERS:
        if not pooled[name]:
            continue
        st = describe(np.concatenate(pooled[name]), name, base_n)
        if st:
            rows.append((name, st))
    base = dict(rows)[TRIGGERS[0][0]] if rows else {}
    for name, st in rows:
        print(f" {name:24} {st['n']:8,.0f} {st['share'] * 100:5.1f}% "
              f"{st['median'] * 100:+7.1f}% {st['p90'] * 100:+7.1f}% "
              f"{st['p99'] * 100:+8.1f}% {st['max'] * 100:+8.1f}% "
              f"{st['p50up'] * 100:5.2f}% {st['p100up'] * 100:5.2f}% "
              f"{st['dn30'] * 100:5.1f}%")
    print("=" * 108)
    if base:
        print(f" 기준선 대비 P(+100%) 배수 — 1.0 미만은 그 조건이 꼬리를 **깎는다**는 뜻")
        print("-" * 108)
        for name, st in sorted(rows, key=lambda kv: -kv[1]["p100up"]):
            if name == TRIGGERS[0][0] or base["p100up"] <= 0:
                continue
            print(f" {name:24} P(+100%) {st['p100up'] * 100:5.2f}% "
                  f"({st['p100up'] / base['p100up']:5.2f}배)  "
                  f"P(+50%) {st['p50up'] * 100:5.2f}% "
                  f"({st['p50up'] / max(1e-9, base['p50up']):5.2f}배)  "
                  f"하방 P(-30%) {st['dn30'] / max(1e-9, base['dn30']):5.2f}배")
        print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
