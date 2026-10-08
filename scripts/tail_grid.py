"""꼬리 구조 전수 탐색 — `P(R ≥ +100%)`를 최대로 만드는 조합을 찾는다.

## 왜 별도 스크립트인가

`scripts/tail_optimizer.py`는 조건 하나(`--trigger`)만 받고 결과를 출력만 한다.
구조를 **고르려면** 조건 6종 × 종목수 4 × 회전 5 = 120조합을 한 표에 모아
정렬해야 한다. 그래서 이 파일은 `tail_optimizer`의 `collect_pools()` /
`simulate()`를 **그대로 import해서** 돌리는 얇은 래퍼다. 시뮬레이션 로직을
복제하면 검증이 아니라 다른 코드를 측정하는 것이 된다.

## 왜 스킬 0을 기준으로 읽는가

기존 기록에 남은 꼬리 수치는 전부 **스킬 60% 가정**이었다. 그런데 실측 스킬은
0에 가깝다.

- 분봉 모델 AUC 0.5585 (표본 2.6배로 늘리자 0.6063 → 0.5661 → 0.5585로 하락)
- 분봉 선정은 **무작위보다 나빴다** — 선택 평균 −0.366% vs 전체 −0.213%
- 일봉 전략 승구간 11/28 vs 동일가중 매수보유 21/28

그래서 기본 `--skills`를 `0,0.1,0.2`로 둔다. 60%를 가정한 수치로 구조를 고르면
가정이 결론을 만든다.

## 비교 기준선

스윙(8종목 분산) 30거래일 `P(+100%)`는 **0.96%**, 분봉 단타는 **0.00%**다.
여기서 나온 최선이 0.96%를 넘지 못하면 구조를 바꿀 이유가 없다.

## 이 측정이 보지 않는 것

`simulate()`는 라운드 안에서 손절·익절을 모사하지 않고 **보유 후 청산만** 본다.
그래서 여기서 고른 구조는 실엔진(`scripts/ablate_exits.py`)으로 다시 확인해야
한다. 또 독립 추출이라 테마 상관을 놓쳐 꼬리를 **과소평가**하고, 유니버스 생존
편향은 **과대평가**한다 — 절대값이 아니라 **구조 간 순위**를 믿는다.

```bash
python scripts/tail_grid.py                        # 6조건 전수, 약 몇 분
python scripts/tail_grid.py --trials 80000         # 꼬리 추정 정밀도를 올린다
python scripts/tail_grid.py --top 30               # 표를 더 길게 본다
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.ablate_exits import panel_fingerprint  # noqa: E402
from scripts.tail_optimizer import (  # noqa: E402
    MAX_GROSS,
    MAX_SINGLE,
    POSITION_COUNTS,
    ROUND_PLANS,
    TRIGGERS,
    collect_pools,
    simulate,
)
from utils.logger import enable_utf8_console, logger  # noqa: E402

# 비교 기준선 — 이것을 넘지 못하면 구조를 바꿀 이유가 없다.
BASELINE = {"스윙 8종목 분산": 0.0096, "분봉 단타": 0.0000}


def _write(out_path: Path, payload: Dict[str, Any]) -> None:
    """원자적으로 쓴다. 조건마다 호출되므로 중간에 죽어도 앞의 결과가 남는다.

    장시간 측정이 체크포인트 없이 세 번 전손한 적이 있다(8/31 · 5/28 · 14/31).
    중단은 예외가 아니라 정상 경로다.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(out_path)


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="꼬리 구조 전수 탐색")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--skills", default="0,0.1,0.2",
                    help="상위 25% 구간에서 뽑을 확률. 실측 스킬이 0에 가까우므로 "
                         "0을 기준으로 읽고 나머지는 민감도다")
    ap.add_argument("--triggers", default="",
                    help="쉼표로 구분한 조건 이름. 비우면 6종 전부")
    ap.add_argument("--top", type=int, default=20, help="표에 찍을 상위 조합 수")
    ap.add_argument("--out", default="reports/tail_grid.json")
    args = ap.parse_args()

    s = get_settings()
    panel = load_panel_from_dir(args.source, list(s.UNIVERSE))
    if not panel:
        logger.error(f"[TailGrid] {args.source} 에서 데이터를 찾지 못했습니다")
        return 1

    fp = panel_fingerprint(panel)
    logger.warning(f"[TailGrid] 데이터 지문 {fp['hash']} ({fp['symbols']}종목 · "
                   f"{fp['bars']}봉 · {fp['ts_from']}~{fp['ts_to']}) — "
                   "지문이 다른 실행과는 비교할 수 없다")

    names = ([t.strip() for t in args.triggers.split(",") if t.strip()]
             or list(TRIGGERS))
    unknown = [n for n in names if n not in TRIGGERS]
    if unknown:
        logger.error(f"[TailGrid] 모르는 조건: {unknown} / 가능: {list(TRIGGERS)}")
        return 1

    skills = [float(x) for x in args.skills.split(",")]
    horizons = sorted({h for _, h in ROUND_PLANS})
    out_path = Path(args.out)
    rows: List[Dict[str, Any]] = []
    pool_stats: List[Dict[str, Any]] = []
    rng = np.random.default_rng(16)

    for i, name in enumerate(names, 1):
        logger.info(f"[TailGrid] {i}/{len(names)} 조건 '{name}' 풀 수집")
        pools = collect_pools(panel, name, horizons, args.warmup, args.step,
                              s.round_trip_cost)
        for h in horizons:
            p = pools.get(h)
            if p is None or not len(p):
                continue
            pool_stats.append({
                "trigger": name, "horizon": h, "n": int(len(p)),
                "median": float(np.median(p)),
                "p90": float(np.percentile(p, 90)),
                "p99": float(np.percentile(p, 99)),
                "single_p100": float((p >= 1.0).mean()),
            })
        for skill in skills:
            for n_rounds, h in ROUND_PLANS:
                for n_pos in POSITION_COUNTS:
                    pool = pools.get(h)
                    if pool is None or not len(pool):
                        continue
                    r = simulate(pool, n_pos, n_rounds, args.trials, skill, rng)
                    if not len(r):
                        continue
                    rows.append({
                        "trigger": name, "skill": skill, "n_pos": n_pos,
                        "rounds": n_rounds, "horizon": h,
                        "weight": min(MAX_SINGLE, MAX_GROSS / n_pos),
                        "pool_n": int(len(pool)),
                        "median": float(np.median(r)),
                        "p50up": float((r >= 0.5).mean()),
                        "p100up": float((r >= 1.0).mean()),
                        "p50dn": float((r <= -0.5).mean()),
                    })
        # 조건 하나가 끝날 때마다 쓴다.
        _write(out_path, {
            "complete": i == len(names),
            "triggers_done": names[:i],
            "triggers_planned": names,
            "data_fingerprint": fp,
            "baseline_p100": BASELINE,
            "config": {"trials": args.trials, "warmup": args.warmup,
                       "step": args.step, "skills": skills,
                       "round_trip_cost": s.round_trip_cost,
                       "max_single": MAX_SINGLE, "max_gross": MAX_GROSS},
            "pools": pool_stats,
            "rows": rows,
        })

    # ---------------------------------------------------------------- 출력
    print()
    print("=" * 108)
    print(f" 꼬리 구조 전수 탐색 — {fp['symbols']}종목 · {fp['bars']}봉 · "
          f"조건 {len(names)}종 · 시행 {args.trials:,} · 왕복비용 {s.round_trip_cost:.3%}")
    print(f" 데이터 지문 {fp['hash']} — 지문이 다른 실행과는 비교할 수 없다")
    print(" 목적함수는 P(R ≥ +100%)다. 중앙값은 참고용이다.")
    print("=" * 108)

    print()
    print(" [단일 종목 풀 — 구조를 씌우기 전의 원재료]")
    print(f" {'조건':<16}{'지평':>6}{'표본':>10}{'중앙값':>10}{'p90':>9}{'p99':>10}{'+100%':>9}")
    print("-" * 108)
    for st in sorted(pool_stats, key=lambda x: (-x["single_p100"], x["trigger"]))[:18]:
        print(f" {st['trigger']:<16}{st['horizon']:5d}봉{st['n']:10,d}"
              f"{st['median'] * 100:+9.1f}%{st['p90'] * 100:+8.1f}%"
              f"{st['p99'] * 100:+9.1f}%{st['single_p100'] * 100:8.2f}%")

    for skill in skills:
        sel = [r for r in rows if r["skill"] == skill]
        if not sel:
            continue
        print()
        print(f" [스킬 {skill:.0%} — 상위 {args.top}조합, P(+100%) 내림차순]")
        print(f" {'조건':<16}{'구조':<22}{'비중':>7}{'중앙값':>10}"
              f"{'P(+50%)':>10}{'P(+100%)':>11}{'P(-50%)':>10}")
        print("-" * 108)
        for r in sorted(sel, key=lambda x: -x["p100up"])[:args.top]:
            label = f"{r['n_pos']}종목 × {r['rounds']}라운드({r['horizon']}봉)"
            print(f" {r['trigger']:<16}{label:<22}{r['weight'] * 100:6.0f}%"
                  f"{r['median'] * 100:+9.1f}%{r['p50up'] * 100:9.2f}%"
                  f"{r['p100up'] * 100:10.2f}%{r['p50dn'] * 100:9.2f}%")
        best = max(sel, key=lambda x: x["p100up"])
        print("-" * 108)
        for bname, bval in BASELINE.items():
            verdict = "넘는다" if best["p100up"] > bval else "넘지 못한다"
            print(f" 기준선 {bname:<16} P(+100%) {bval * 100:5.2f}%  "
                  f"→ 최선 {best['p100up'] * 100:5.2f}% 로 {verdict}")

    print("=" * 108)
    print(" 라운드 안의 손절·익절은 모사하지 않는다(보유 후 청산만) — 실엔진으로 재확인한다.")
    print(" 독립 추출이라 꼬리는 과소평가, 생존 편향은 과대평가다. 구조 간 순위를 믿는다.")
    print(f" 저장 → {out_path}")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
