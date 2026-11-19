"""꼬리 최적화 — P(수익률 ≥ +100%)를 최대로 만드는 포트폴리오 구조를 찾는다.

## 이 스크립트가 푸는 문제

대회 심사는 절대수익 top10이고 지난 대회 10위권이 +100%였다. 보상이 계단함수이므로
목적함수는 `E[R]`이 아니라 **`P(R ≥ 100%)`**다. 안정적 자산운용이 아니라 꼬리에
닿을 확률을 높이는 게임이다.

종목 하나의 꼬리는 `scripts/research/surge_study.py`가 재 준다(일간 +15% 조건에서 30봉
`P(+100%)` = 4.42%, 기준선의 4.66배). 그런데 **종목의 꼬리와 포트폴리오의 꼬리는
다르다.** 단일 종목 50% 한도가 있기 때문이다.

```
2종목 × 45% → 0.45(r₁+r₂) = 1.0  → 두 종목 평균 +111% 필요
1종목 × 50% → 0.50·r     = 1.0  → 한 종목이 +200% 필요
```

한 번에 몰아서는 닿기 어렵고 **회전이 필요하다.** 몇 종목으로 몇 번 돌리는 것이
최적인지는 가정이 아니라 실측 분포로 계산해야 한다.

## 어떻게 계산하는가

몬테카를로를 **실측 분포에서 부트스트랩**한다. 모수를 가정하지 않는다(정규분포를
가정하면 꼬리가 사라지고, 로그정규를 가정하면 꼬리 두께가 가정값이 된다).

1. 조건(예: 일간 +15%)이 걸린 모든 종목·시점에서 **`h`봉 선행 수익률**을 모은다.
2. 라운드마다 `n`종목을 그 풀에서 뽑아 비중 `min(0.5, 0.9/n)`으로 담는다.
3. 라운드 수익률을 복리로 누적해 30거래일 최종 수익률을 만든다.
4. 4만 회 반복해 `P(R ≥ 100%)`, `P(R ≥ 50%)`, 중앙값, `P(R ≤ −50%)`를 센다.

## 독립 추출은 꼬리를 **과소평가**한다

같은 라운드의 종목들을 독립으로 뽑는다. 실제로는 같은 장세·같은 테마라 양의 상관이
있고, 상관은 포트폴리오 분산을 **키워** 꼬리를 두껍게 한다. 따라서 여기 숫자는
보수적인 하한이다. 반대로 생존 편향(유니버스가 2026-09-29 기준 거래대금 상위)은
숫자를 부풀린다. 두 편향이 반대 방향이므로 **절대값보다 구조 간 순위**를 믿는다.

## 스킬 모형

조건이 걸린 종목 중 **무작위로** 뽑는 것이 기준선이다. 모델이 쓸모 있다면 풀의
상위에서 더 자주 뽑는다. `--skill`은 "상위 25% 구간에서 뽑을 확률"이며, 0이면
무작위다. 모델 성능을 모르는 상태에서 숫자 하나로 가정하는 대신 **여러 값에서의
민감도**를 같이 찍는다.

```bash
python scripts/research/tail_optimizer.py
python scripts/research/tail_optimizer.py --trigger "거래대금 10배" --trials 80000
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.catalyst_features import build_catalyst_features  # noqa: E402
from utils.logger import enable_utf8_console, logger  # noqa: E402

B3 = float(np.log1p(3.0))
B10 = float(np.log1p(10.0))

TRIGGERS: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "무조건": lambda d: np.ones(len(d), dtype=bool),
    "일간 +10%": lambda d: d["ret_1"].to_numpy() >= 0.10,
    "일간 +15%": lambda d: d["ret_1"].to_numpy() >= 0.15,
    "급등+폭증 동시": lambda d: (d["ret_1"].to_numpy() >= 0.10)
                              & (d["turnover_burst_20"].to_numpy() >= B3),
    "거래대금 10배": lambda d: d["turnover_burst_20"].to_numpy() >= B10,
    "신고가+폭증": lambda d: (d["high_60_break"].to_numpy() >= 0.5)
                            & (d["turnover_burst_20"].to_numpy() >= B3),
}

# (라운드 수, 라운드당 봉 수) — 대회 30거래일을 나눈다.
ROUND_PLANS: List[Tuple[int, int]] = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5)]
POSITION_COUNTS: List[int] = [1, 2, 3, 5]

MAX_SINGLE = 0.50      # 대회 규정 단일 종목 한도
MAX_GROSS = 0.90       # 총 익스포저 상한(현금 여유)


def collect_pools(panel: Dict[str, pd.DataFrame], trigger: str,
                  horizons: List[int], warmup: int, step: int,
                  cost: float) -> Dict[int, np.ndarray]:
    """조건이 걸린 시점들의 `h`봉 선행 수익률 풀. 비용을 차감한 실현 가능 수익률."""
    fn = TRIGGERS[trigger]
    pools: Dict[int, List[np.ndarray]] = {h: [] for h in horizons}
    for sym, df in panel.items():
        if df is None or len(df) < warmup + max(horizons) + 10:
            continue
        d = df.reset_index(drop=True).copy()
        cat = build_catalyst_features(d)
        for c in cat.columns:
            d[c] = cat[c].to_numpy()
        d["ret_1"] = d["close"].astype(float).pct_change()
        try:
            trig = np.asarray(fn(d), dtype=bool)
        except KeyError:
            continue
        # 체결 가정은 백테스트와 같다 — 봉 t 종가로 판단하고 t+1 시가에 들어간다.
        open_ = d["open"].to_numpy(float)
        close = d["close"].to_numpy(float)
        n = len(d)
        base = np.zeros(n, dtype=bool)
        base[warmup::step] = True
        for h in horizons:
            idx = np.where(trig & base)[0]
            idx = idx[(idx + 1 + h) < n]
            if not len(idx):
                continue
            entry = open_[idx + 1]
            exit_px = close[idx + 1 + h]
            ok = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_px)
            r = exit_px[ok] / entry[ok] - 1.0 - cost
            if len(r):
                pools[h].append(r)
    return {h: (np.concatenate(v) if v else np.array([])) for h, v in pools.items()}


def simulate(pool: np.ndarray, n_pos: int, n_rounds: int, trials: int,
             skill: float, rng: np.random.Generator) -> np.ndarray:
    """포트폴리오 최종 수익률 표본.

    비중은 `min(단일한도, 총익스포저/종목수)`다. 1종목이면 50%(규정 한도)이고
    나머지 40%는 현금으로 남는다 — 규정이 강제하는 구조적 손실이다.
    """
    if not len(pool):
        return np.array([])
    w = min(MAX_SINGLE, MAX_GROSS / n_pos)
    top = np.sort(pool)[int(len(pool) * 0.75):]      # 상위 25% 구간
    equity = np.ones(trials)
    for _ in range(n_rounds):
        draws = rng.choice(pool, size=(trials, n_pos), replace=True)
        if skill > 0 and len(top):
            # 스킬만큼 상위 구간에서 다시 뽑는다. "모델이 좋은 것을 더 자주 고른다".
            hit = rng.random((trials, n_pos)) < skill
            good = rng.choice(top, size=(trials, n_pos), replace=True)
            draws = np.where(hit, good, draws)
        # 투자분만 수익/손실을 보고 나머지는 현금으로 남는다.
        equity = equity * (1.0 + w * draws.sum(axis=1))
        equity = np.maximum(equity, 0.0)             # 파산은 0에서 멈춘다
    return equity - 1.0


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="꼬리 확률 최적 구조 탐색")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--trigger", default="일간 +15%", choices=list(TRIGGERS))
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--skills", default="0,0.3,0.6")
    args = ap.parse_args()

    s = get_settings()
    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    if not panel:
        logger.error(f"[Tail] {args.source} 에서 데이터를 찾지 못했습니다")
        return 1
    horizons = sorted({h for _, h in ROUND_PLANS})
    logger.info(f"[Tail] {len(panel)}종목 · 조건 '{args.trigger}' · 지평 {horizons}")

    pools = collect_pools(panel, args.trigger, horizons, args.warmup,
                          args.step, s.round_trip_cost)
    print()
    print("=" * 104)
    print(f" 꼬리 최적화 — 조건 '{args.trigger}' · {len(panel)}종목 · 왕복비용 {s.round_trip_cost:.3%}")
    print(" 목적함수는 P(R ≥ +100%)다. 중앙값은 참고용으로만 찍는다.")
    print("=" * 104)
    print(f" {'지평':>6} {'표본':>9} {'중앙값':>9} {'p90':>9} {'p99':>10} {'+100%':>8}")
    print("-" * 104)
    for h in horizons:
        p = pools[h]
        if not len(p):
            continue
        print(f" {h:5d}봉 {len(p):9,d} {np.median(p) * 100:+8.1f}% "
              f"{np.percentile(p, 90) * 100:+8.1f}% {np.percentile(p, 99) * 100:+9.1f}% "
              f"{(p >= 1.0).mean() * 100:7.2f}%")
    print("=" * 104)

    rng = np.random.default_rng(16)
    for skill in [float(x) for x in args.skills.split(",")]:
        print()
        print(f" [스킬 {skill:.0%} — 상위 25% 구간에서 뽑을 확률]")
        print(f" {'구조':<22} {'비중/종목':>9} {'중앙값':>9} {'P(+50%)':>9} "
              f"{'P(+100%)':>10} {'P(-50%)':>9}")
        print("-" * 104)
        rows = []
        for n_rounds, h in ROUND_PLANS:
            for n_pos in POSITION_COUNTS:
                pool = pools.get(h)
                if pool is None or not len(pool):
                    continue
                r = simulate(pool, n_pos, n_rounds, args.trials, skill, rng)
                if not len(r):
                    continue
                w = min(MAX_SINGLE, MAX_GROSS / n_pos)
                rows.append((f"{n_pos}종목 × {n_rounds}라운드({h}봉)", w,
                             float(np.median(r)), float((r >= 0.5).mean()),
                             float((r >= 1.0).mean()), float((r <= -0.5).mean())))
        for label, w, med, p50, p100, dn in sorted(rows, key=lambda x: -x[4])[:12]:
            print(f" {label:<22} {w * 100:8.0f}% {med * 100:+8.1f}% "
                  f"{p50 * 100:8.2f}% {p100 * 100:9.2f}% {dn * 100:8.2f}%")
    print("=" * 104)
    print(" 독립 추출이므로 꼬리는 과소평가다(실제는 테마 상관으로 분산이 커진다).")
    print(" 생존 편향은 반대로 과대평가다. 절대값보다 구조 간 순위를 믿는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
