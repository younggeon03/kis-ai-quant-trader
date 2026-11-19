"""분봉 청산 규칙 격자 — 급등을 잡은 뒤 언제 빠질 것인가.

## 이 격자가 답하는 질문

급등 포착 자체는 효과가 확인됐다(장중 +10% 돌파 조건의 30봉 `P(+100%)`가 기준선의
3.3배). 남은 문제는 **청산**이다. 일봉 실측에서 청산 규칙이 결과를 완전히 바꿨다.

| 청산 | 스킬 0% | 스킬 10% | 스킬 20% |
|---|---|---|---|
| 익절 3% | 0.00% | 0.00% | **0.00%** |
| 익절 10% | 0.19% | 4.04% | **24.79%** |

**익절 3%는 스킬이 아무리 좋아도 `P(+100%)`가 영구히 0이다.** 상승을 3%에서 자르면
꼬리가 구조적으로 존재할 수 없다. 그런데 크게 열어두면 되돌림을 다 맞는다.
그 사이의 최적점은 가정으로 정할 수 없다.

## 일봉으로는 왜 못 쟀는가

일봉은 당일 저가가 **돌파 전인지 후인지** 구분하지 못한다. 가격이 저가에서 올라와
돌파선을 지났을 수도 있으므로, 일봉 기준 "손절 도달률 96%"는 의미 없는 비관 상한이었다.
**목표와 손절의 선후**를 알아야 기대값이 나오고, 그것이 분봉을 수집한 이유다.

## 규칙 종류

- **고정 목표/손절**: 목표 먼저면 +목표, 손절 먼저면 −손절, 둘 다 없으면 종가
- **트레일링**: `activate` 이상 오르면 고점 대비 `trail`만큼 밀릴 때 청산.
  상승을 열어두면서 되돌림만 끊는다 — 꼬리 목적함수에 맞는 유일한 형태다
- **종가 청산**: 아무 규칙 없이 당일 종가 (비교 기준)

## 같은 분봉 안에서 양쪽을 건드리면 손절을 먼저 본다

1분봉 안의 가격 순서는 알 수 없다. 유리한 쪽을 가정하면 결과가 부풀려진다.
삼중배리어 라벨이 같은 이유로 같은 선택을 한다(`ai_model.py`).

## 판정 기준

평균 수익률이 아니라 **`P(+100%)`**다. 30거래일 × 하루 2종목 × 45% 비중으로
복리 누적한 분포에서 센다. 스킬은 "풀 상위 25%에서 뽑을 확률"이며, 모델 성능을
모르는 상태에서 하나로 가정하는 대신 민감도를 같이 찍는다.

```bash
python scripts/research/minute_exit_grid.py
python scripts/research/minute_exit_grid.py --trigger 0.08 --out reports/minute_grid_08.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.minute_features import find_trigger  # noqa: E402
from utils.logger import enable_utf8_console, logger, setup_logging  # noqa: E402

# (이름, 목표, 손절, 트레일 발동, 트레일 폭) — None은 미적용
RULES: List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]] = [
    ("종가 청산(기준)",        None,  None,  None,  None),
    ("목표 3% / 손절 2%",      0.03,  0.02,  None,  None),
    ("목표 5% / 손절 3%",      0.05,  0.03,  None,  None),
    ("목표 10% / 손절 5%",     0.10,  0.05,  None,  None),
    ("목표 15% / 손절 5%",     0.15,  0.05,  None,  None),
    ("목표 20% / 손절 8%",     0.20,  0.08,  None,  None),
    ("손절 3%만",              None,  0.03,  None,  None),
    ("손절 5%만",              None,  0.05,  None,  None),
    ("손절 8%만",              None,  0.08,  None,  None),
    ("트레일 2%(즉시)",         None,  None,  0.00,  0.02),
    ("트레일 3%(즉시)",         None,  None,  0.00,  0.03),
    ("트레일 5%(즉시)",         None,  None,  0.00,  0.05),
    ("트레일 3%(+3% 후)",       None,  0.03,  0.03,  0.03),
    ("트레일 5%(+5% 후)",       None,  0.05,  0.05,  0.05),
    ("트레일 8%(+5% 후)",       None,  0.05,  0.05,  0.08),
    ("트레일 5%(+10% 후)",      None,  0.05,  0.10,  0.05),
]


def simulate_rule(h: np.ndarray, l: np.ndarray, c: np.ndarray, entry: float,
                  target: Optional[float], stop: Optional[float],
                  activate: Optional[float], trail: Optional[float]) -> float:
    """사건 하나에 규칙 하나를 적용한 실현 수익률(비용 전).

    진입 후 분봉을 순서대로 보며 가장 먼저 발동하는 청산을 택한다. 아무것도
    발동하지 않으면 종가다. **같은 봉에서 손절과 목표가 함께 닿으면 손절**이다.
    """
    n = len(c)
    if n == 0 or entry <= 0:
        return 0.0
    up = h / entry - 1.0
    dn = l / entry - 1.0

    i_stop = n
    if stop is not None:
        hit = np.flatnonzero(dn <= -stop)
        if len(hit):
            i_stop = int(hit[0])

    i_tgt = n
    if target is not None:
        hit = np.flatnonzero(up >= target)
        if len(hit):
            i_tgt = int(hit[0])

    i_trail = n
    trail_px = 0.0
    if trail is not None:
        run_max = np.maximum.accumulate(up)
        armed = run_max >= (activate or 0.0)
        # 고점 대비 trail 만큼 밀렸는가. 발동 전 봉은 제외한다.
        giveback = (1.0 + run_max) * (1.0 - trail) - 1.0
        hit = np.flatnonzero(armed & (dn <= giveback))
        if len(hit):
            i_trail = int(hit[0])
            trail_px = float(giveback[i_trail])

    first = min(i_stop, i_tgt, i_trail)
    if first >= n:
        return float(c[-1] / entry - 1.0)
    # 동시 발동 우선순위: 손절 > 트레일 > 목표 (보수적)
    if first == i_stop:
        return float(-stop)
    if first == i_trail:
        return float(trail_px)
    return float(target)


def collect_outcomes(daily: Dict[str, pd.DataFrame], minute_dir: str,
                     trigger: float, min_bars: int,
                     vol_mult: float = 1.5) -> Dict[str, np.ndarray]:
    """규칙별 사건 수익률 배열. 분봉을 종목 단위로 흘려보내 메모리를 묶지 않는다."""
    acc: Dict[str, List[float]] = {name: [] for name, *_ in RULES}
    mdir = Path(minute_dir)
    n_events = 0
    for k, (sym, ddf) in enumerate(sorted(daily.items()), 1):
        path = mdir / f"{sym}.csv"
        if not path.exists() or ddf is None or len(ddf) < 2:
            continue
        d = ddf.reset_index(drop=True)
        prev_c = dict(zip(d["ts"].astype(str), d["close"].astype(float).shift(1)))
        prev_v = dict(zip(d["ts"].astype(str), d["volume"].astype(float).shift(1)))
        try:
            mdf = pd.read_csv(path, dtype={"ts": str})
        except Exception:
            continue
        if mdf.empty:
            continue
        for day, grp in mdf.groupby(mdf["ts"].str[:8], sort=True):
            if len(grp) < min_bars:
                continue
            pc = prev_c.get(str(day))
            if pc is None or not np.isfinite(pc) or pc <= 0:
                continue
            g = grp.reset_index(drop=True)
            idx = find_trigger(g, float(pc), trigger,
                               float(prev_v.get(str(day)) or 0.0), vol_mult)
            if idx is None:
                continue
            o = g["open"].to_numpy(float)
            entry = float(o[idx + 1])
            h = g["high"].to_numpy(float)[idx + 1:]
            l = g["low"].to_numpy(float)[idx + 1:]
            c = g["close"].to_numpy(float)[idx + 1:]
            if not len(c) or entry <= 0:
                continue
            n_events += 1
            for name, tgt, stp, act, tr in RULES:
                acc[name].append(simulate_rule(h, l, c, entry, tgt, stp, act, tr))
        if k % 40 == 0:
            logger.info(f"[Grid] {k}/{len(daily)}종목 · 사건 {n_events:,}건")
    logger.info(f"[Grid] 사건 {n_events:,}건 수집")
    return {k: np.asarray(v, dtype=float) for k, v in acc.items()}


def tail_probability(pool: np.ndarray, skill: float, trials: int,
                     days: int, per_day: int, weight: float,
                     rng: np.random.Generator) -> Dict[str, float]:
    """30거래일 복리 분포에서 꼬리 확률. `scripts/research/tail_optimizer.py`와 같은 방식."""
    if not len(pool):
        return {}
    top = np.sort(pool)[int(len(pool) * 0.75):]
    draws = rng.choice(pool, size=(trials, days, per_day), replace=True)
    if skill > 0 and len(top):
        hit = rng.random((trials, days, per_day)) < skill
        draws = np.where(hit, rng.choice(top, size=(trials, days, per_day),
                                         replace=True), draws)
    eq = np.prod(1.0 + weight * draws.sum(axis=2), axis=1) - 1.0
    return {"median": float(np.median(eq)), "p50up": float((eq >= 0.5).mean()),
            "p100up": float((eq >= 1.0).mean()), "dn50": float((eq <= -0.5).mean())}


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="분봉 청산 규칙 격자")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--minute-dir", default="data/cache/minute")
    ap.add_argument("--trigger", type=float, default=0.0,
                    help="돌파 기준(0=설정값 INTRADAY_TRIGGER_PCT)")
    ap.add_argument("--vol-mult", type=float, default=0.0,
                    help="돌파 봉 거래량 / 전일 분당 평균 하한(0=설정값 INTRADAY_VOL_MULT)")
    ap.add_argument("--min-bars", type=int, default=200)
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--days", type=int, default=30, help="대회 거래일 수")
    ap.add_argument("--per-day", type=int, default=2, help="하루 동시 보유 종목 수")
    ap.add_argument("--weight", type=float, default=0.45, help="종목당 비중")
    ap.add_argument("--skills", default="0,0.10,0.20")
    ap.add_argument("--out", default="reports/minute_exit_grid.json")
    args = ap.parse_args()

    s = get_settings()
    setup_logging(s.LOG_DIR, s.LOG_LEVEL)
    args.trigger = args.trigger or s.INTRADAY_TRIGGER_PCT
    args.vol_mult = args.vol_mult or s.INTRADAY_VOL_MULT
    have = sorted(p.stem for p in Path(args.minute_dir).glob("*.csv"))
    if not have:
        logger.error(f"[Grid] {args.minute_dir} 에 분봉이 없습니다")
        return 1
    daily = load_panel_from_dir(args.source, have)
    logger.info(f"[Grid] 분봉 {len(have)}종목 · 트리거 +{args.trigger:.0%} "
                f"· 돌파봉 거래량 ≥ 전일분당×{args.vol_mult}")

    pools = collect_outcomes(daily, args.minute_dir, args.trigger, args.min_bars,
                             args.vol_mult)
    cost = s.round_trip_cost
    rng = np.random.default_rng(16)
    skills = [float(x) for x in args.skills.split(",")]

    print()
    print("=" * 112)
    print(f" 분봉 청산 격자 — 트리거 +{args.trigger:.0%} · 왕복비용 {cost:.3%} · "
          f"하루 {args.per_day}종목 × {args.weight:.0%} × {args.days}일")
    print(" 판정 기준은 P(+100%)다. 평균은 참고용이다.")
    print("=" * 112)
    head = (f" {'청산 규칙':22} {'표본':>7} {'평균/회':>8} {'중앙/회':>8} {'승률':>6} "
            f"{'표준편차':>8}")
    for sk in skills:
        head += f" {'P100@' + format(sk * 100, '.0f') + '%':>10}"
    print(head)
    print("-" * 112)

    report: Dict[str, Dict] = {}
    rows = []
    for name, *_ in RULES:
        pool = pools[name] - cost
        if not len(pool):
            continue
        rec = {"n": int(len(pool)), "mean": float(pool.mean()),
               "median": float(np.median(pool)), "win": float((pool > 0).mean()),
               "std": float(pool.std()), "skills": {}}
        line = (f" {name:22} {len(pool):7,d} {pool.mean() * 100:+7.2f}% "
                f"{np.median(pool) * 100:+7.2f}% {(pool > 0).mean() * 100:5.1f}% "
                f"{pool.std() * 100:7.2f}%")
        for sk in skills:
            st = tail_probability(pool, sk, args.trials, args.days,
                                  args.per_day, args.weight, rng)
            rec["skills"][f"{sk:.2f}"] = st
            line += f" {st['p100up'] * 100:9.2f}%"
        rows.append((name, rec, line))
        report[name] = rec
    # P(+100%)가 가장 높은 순으로 정렬한다(마지막 스킬 기준).
    key = f"{skills[-1]:.2f}"
    for name, rec, line in sorted(rows, key=lambda x: -x[1]["skills"][key]["p100up"]):
        print(line)
    print("=" * 112)

    best = max(report.items(), key=lambda kv: kv[1]["skills"][key]["p100up"])
    bst = best[1]["skills"][key]
    print(f" 최선(스킬 {skills[-1]:.0%}): {best[0]} — "
          f"30일 중앙 {bst['median'] * 100:+.1f}% · P(+50%) {bst['p50up'] * 100:.2f}% · "
          f"P(+100%) {bst['p100up'] * 100:.2f}% · P(-50%) {bst['dn50'] * 100:.2f}%")
    print(" 독립 추출이라 꼬리는 과소평가, 생존 편향은 과대평가다. 규칙 간 순위를 믿는다.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"trigger": args.trigger, "cost": cost, "days": args.days,
                    "per_day": args.per_day, "weight": args.weight,
                    "symbols": len(have), "skills": skills},
         "rules": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"[Grid] 저장 → {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
