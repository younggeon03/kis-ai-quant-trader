"""분봉 모델의 스킬 측정 — 이 전략이 성립하는지의 최종 판정.

## 왜 스킬이 전부인가

꼬리 최적화(`scripts/tail_optimizer.py`)에서 구조는 `P(+100%)`를 2~3배 바꿨는데
스킬은 **6배 이상** 바꿨다. 그리고 청산 격자에서 "조건 통과 사건 중 무작위" 가정의
`P(+100%)`는 **0.00%**였다. 즉 모델이 급등 사건을 분별하지 못하면 이 전략은
성립하지 않고, 분별한다면 스윙 전략을 압도한다. 그 갈림길의 숫자가 스킬이다.

## 가정하지 않고 측정한다

`tail_optimizer`와 `minute_exit_grid`는 스킬을 "풀 상위 25%에서 뽑을 확률"이라는
**파라미터로 가정**했다. 모델 성능을 모르는 상태에서 민감도를 보려는 장치였다.

여기서는 가정하지 않는다. 실제로 모델을 학습해서 **모델이 고른 사건의 실현 수익률
분포**를 만들고, 그 분포를 그대로 30거래일 복리에 넣는다. 파라미터 변환이 필요하면
`스킬 = (상위 사분위 적중률 − 0.25) / 0.75`로 환산해 비교한다.

## 워크포워드 — 라벨 중첩이 없다

일봉 전략의 골칫거리였던 라벨 중첩이 여기서는 없다. 돌파 사건의 라벨은 **같은 날
장 마감 전에 확정**되므로 다음 날 사건과 미래를 공유하지 않는다. 그래서 퍼지 구간
없이 날짜로만 자르면 된다. `CV_EMBARGO` 같은 보정이 필요하지 않다.

학습은 **과거만** 본다. 구간 t의 사건을 예측할 때 t 이전 사건으로만 학습한다.

## 두 가지를 센다

1. **분별력** — AUC, 그리고 모델 상위 선택이 실현 수익률 상위 사분위에 드는 비율
2. **실전 결과** — 날마다 모델 상위 K개를 골라 담았을 때의 30거래일 복리 분포

2번이 본질이다. AUC가 0.6이어도 그것이 수익으로 바뀌는지는 별개이고, 대회 심사는
`P(+100%)`만 본다.

```bash
python scripts/minute_skill.py
python scripts/minute_skill.py --trigger 0.08 --exit close
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data import load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.minute_features import MINUTE_FEATURES, find_trigger  # noqa: E402
from scripts.minute_exit_grid import simulate_rule  # noqa: E402
from utils.logger import enable_utf8_console, logger, setup_logging  # noqa: E402

# 청산 규칙 — 격자에서 꼬리가 가장 두꺼웠던 것들
EXITS: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = {
    "close": (None, None, None, None),            # 종가 청산 (격자 1위)
    "t10s5": (0.10, 0.05, None, None),            # 목표 10% / 손절 5%
    "s8": (None, 0.08, None, None),               # 손절 8%만
}


def build_events(daily: Dict[str, pd.DataFrame], minute_dir: str,
                 trigger: float, exit_key: str, min_bars: int,
                 vol_mult: float = 1.5) -> pd.DataFrame:
    """사건 테이블 — 피처 + 선택한 청산 규칙의 실현 수익률 + 분류 라벨."""
    from strategy.minute_features import build_event_features

    tgt, stp, act, tr = EXITS[exit_key]
    mdir = Path(minute_dir)
    rows: List[Dict[str, float]] = []
    for k, (sym, ddf) in enumerate(sorted(daily.items()), 1):
        path = mdir / f"{sym}.csv"
        if not path.exists() or ddf is None or len(ddf) < 2:
            continue
        d = ddf.reset_index(drop=True)
        prev_c = dict(zip(d["ts"].astype(str), d["close"].astype(float).shift(1)))
        prev_v = dict(zip(d["ts"].astype(str), d["volume"].astype(float).shift(1)))
        prev_t = dict(zip(d["ts"].astype(str),
                          (d["close"].astype(float) * d["volume"].astype(float)).shift(1)))
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
            h = g["high"].to_numpy(float)
            l = g["low"].to_numpy(float)
            c = g["close"].to_numpy(float)
            v = g["volume"].to_numpy(float)
            feats = build_event_features(o[:idx + 1], h[:idx + 1], l[:idx + 1],
                                         c[:idx + 1], v[:idx + 1], float(pc),
                                         float(prev_t.get(str(day)) or 0.0), trigger)
            entry = float(o[idx + 1])
            ret = simulate_rule(h[idx + 1:], l[idx + 1:], c[idx + 1:],
                                entry, tgt, stp, act, tr)
            row = {"symbol": sym, "day": str(day), "ret": float(ret)}
            row.update(feats)
            rows.append(row)
        if k % 50 == 0:
            logger.info(f"[Skill] {k}/{len(daily)}종목 · 사건 {len(rows):,}건")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["day", "symbol"]).reset_index(drop=True)


def walkforward_scores(ev: pd.DataFrame, cost: float, n_folds: int,
                       label_q: float) -> pd.DataFrame:
    """날짜 순서로 접어가며 과거만으로 학습하고 미래 사건에 점수를 매긴다.

    라벨은 **그 학습 구간 안에서의 상위 `label_q` 분위**다. 절대 문턱(+10% 등)을
    쓰면 장세에 따라 양성비율이 크게 흔들려 모델이 장세를 배운다. 분위로 잡으면
    "같은 시기의 다른 급등보다 좋았는가"를 배운다 — 우리가 실제로 묻는 질문이다.
    """
    from strategy.ai_model import HAS_LGB

    days = sorted(ev["day"].unique())
    if len(days) < n_folds * 2:
        logger.error(f"[Skill] 날짜 {len(days)}일로는 {n_folds}겹 워크포워드가 불가합니다")
        return pd.DataFrame()
    bounds = [days[int(len(days) * (i + 1) / (n_folds + 1))] for i in range(n_folds)]

    feats = [c for c in MINUTE_FEATURES if c in ev.columns]
    out = []
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else None
        tr = ev[ev["day"] < start]
        te = ev[(ev["day"] >= start) & ((end is None) | (ev["day"] < (end or "9")))]
        if len(tr) < 200 or len(te) < 20:
            continue
        thr = float(tr["ret"].quantile(1.0 - label_q))
        y = (tr["ret"] >= thr).astype(int).to_numpy()
        X = tr[feats].to_numpy(float)
        Xte = te[feats].to_numpy(float)
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            continue

        if HAS_LGB:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=40,
                                   subsample=0.8, colsample_bytree=0.8,
                                   reg_lambda=1.0, verbose=-1, random_state=16)
            m.fit(X, y)
            p = m.predict_proba(Xte)[:, 1]
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=15, random_state=16)
            m.fit(np.nan_to_num(X), y)
            p = m.predict_proba(np.nan_to_num(Xte))[:, 1]

        chunk = te.copy()
        chunk["score"] = p
        chunk["fold"] = i
        out.append(chunk)
        logger.info(f"[Skill] 폴드 {i}: 학습 {len(tr):,} · 평가 {len(te):,} · "
                    f"문턱 {thr * 100:+.2f}% · 양성 {y.mean():.1%}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """순위 기반 AUC. sklearn 없이도 돌게 직접 계산한다."""
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="분봉 모델 스킬 측정")
    ap.add_argument("--source", default="data/cache")
    ap.add_argument("--minute-dir", default="data/cache/minute")
    ap.add_argument("--trigger", type=float, default=0.03)
    ap.add_argument("--vol-mult", type=float, default=1.5,
                    help="돌파 봉 거래량 / 전일 분당 평균 하한")
    ap.add_argument("--exit", default="close", choices=list(EXITS))
    ap.add_argument("--min-bars", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--label-q", type=float, default=0.25, help="라벨 상위 분위")
    ap.add_argument("--per-day", type=int, default=2, help="하루 선택 종목 수")
    ap.add_argument("--weight", type=float, default=0.45)
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--out", default="reports/minute_skill.json")
    args = ap.parse_args()

    s = get_settings()
    setup_logging(s.LOG_DIR, s.LOG_LEVEL)
    cost = s.round_trip_cost
    have = sorted(p.stem for p in Path(args.minute_dir).glob("*.csv"))
    if not have:
        logger.error(f"[Skill] {args.minute_dir} 에 분봉이 없습니다")
        return 1
    daily = load_panel_from_dir(args.source, have)
    logger.info(f"[Skill] 분봉 {len(have)}종목 · 트리거 +{args.trigger:.0%} "
                f"· 거래량 ≥ 전일분당×{args.vol_mult} · 청산 {args.exit}")

    ev = build_events(daily, args.minute_dir, args.trigger, args.exit,
                      args.min_bars, args.vol_mult)
    if ev.empty:
        logger.error("[Skill] 사건이 없습니다")
        return 1
    ev["ret"] = ev["ret"] - cost
    logger.info(f"[Skill] 사건 {len(ev):,}건 · {ev['day'].nunique()}일 · "
                f"하루 평균 {len(ev) / max(1, ev['day'].nunique()):.1f}건")

    scored = walkforward_scores(ev, cost, args.folds, args.label_q)
    if scored.empty:
        logger.error("[Skill] 워크포워드 실패")
        return 1

    # --- 1) 분별력 ---
    q_thr = float(scored["ret"].quantile(1.0 - args.label_q))
    y = (scored["ret"] >= q_thr).astype(int).to_numpy()
    a = auc(y, scored["score"].to_numpy(float))

    # 날마다 상위 K개를 고른다. 실거래와 같은 방식이다.
    picks = (scored.sort_values(["day", "score"], ascending=[True, False])
             .groupby("day").head(args.per_day))
    rand_mean = float(scored["ret"].mean())
    pick_mean = float(picks["ret"].mean())
    topq_rate = float((picks["ret"] >= q_thr).mean())
    skill_equiv = max(0.0, (topq_rate - args.label_q) / (1.0 - args.label_q))

    print()
    print("=" * 100)
    print(f" 분봉 모델 스킬 — 트리거 +{args.trigger:.0%} · 청산 {args.exit} · "
          f"{len(have)}종목 · 사건 {len(ev):,}건")
    print("=" * 100)
    print(f" 워크포워드 평가 사건      {len(scored):,}건 ({scored['fold'].nunique()}폴드)")
    print(f" AUC                     {a:.4f}   (0.5 = 분별력 없음)")
    print(f" 하루 상위 {args.per_day}개 선택       {len(picks):,}건")
    print(f" 상위 사분위 적중률        {topq_rate:.2%}   (무작위 {args.label_q:.0%})")
    print(f" → 환산 스킬              {skill_equiv:.1%}")
    print(f" 선택 평균 수익/회         {pick_mean * 100:+.3f}%   (전체 {rand_mean * 100:+.3f}%)")
    print("-" * 100)

    # --- 2) 실전 결과: 모델이 고른 사건만으로 30거래일 복리 ---
    rng = np.random.default_rng(16)
    def boot(pool: np.ndarray, label: str) -> Dict[str, float]:
        if not len(pool):
            return {}
        d = rng.choice(pool, size=(args.trials, 30, args.per_day), replace=True)
        eq = np.prod(1.0 + args.weight * d.sum(axis=2), axis=1) - 1.0
        st = {"median": float(np.median(eq)), "p50up": float((eq >= 0.5).mean()),
              "p100up": float((eq >= 1.0).mean()), "dn50": float((eq <= -0.5).mean())}
        print(f" {label:22} 중앙 {st['median'] * 100:+8.1f}%  "
              f"P(+50%) {st['p50up'] * 100:6.2f}%  P(+100%) {st['p100up'] * 100:6.2f}%  "
              f"P(-50%) {st['dn50'] * 100:5.2f}%")
        return st

    print(" 30거래일 복리 분포 (하루 {}종목 × {:.0%})".format(args.per_day, args.weight))
    st_pick = boot(picks["ret"].to_numpy(float), "모델 선택")
    st_rand = boot(scored["ret"].to_numpy(float), "무작위 선택")
    print("=" * 100)
    if st_pick and st_rand:
        if st_rand["p100up"] > 0:
            print(f" 모델이 P(+100%)를 {st_pick['p100up'] / st_rand['p100up']:.2f}배로 만든다")
        print(" 비교: 스윙 2종목×6라운드 — 스킬0 0.96% · 스킬10% 2.71% · 스킬20% 6.17%")
    print(" 단서: 부트스트랩은 사건을 독립 추출한다(실제는 장세 상관으로 분산이 더 크다).")
    print(" 유니버스 생존 편향은 반대로 수치를 부풀린다. 절대값보다 비교를 믿는다.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"trigger": args.trigger, "vol_mult": args.vol_mult,
                   "exit": args.exit, "symbols": len(have),
                   "events": int(len(ev)), "folds": args.folds,
                   "label_q": args.label_q, "per_day": args.per_day,
                   "weight": args.weight, "cost": cost},
        "discrimination": {"auc": a, "topq_rate": topq_rate,
                           "skill_equiv": skill_equiv,
                           "pick_mean": pick_mean, "all_mean": rand_mean},
        "bootstrap": {"model": st_pick, "random": st_rand},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"[Skill] 저장 → {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
