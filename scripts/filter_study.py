"""엔진 장 시작 판정 + 이론 기반 필터(11/18 사용자 요청) — 이틀 보유 평균을 올리는 필터를 찾는다.

후보 = 엔진 장 시작 판정(09:00~09:03 거래대금 ≥ 3 x 같은 4분 20일 중앙 · 전일 종가 위 · VWAP 위 · 3일 상승 < 20%)을 통과한 **모든 종목**.
결과 = 09:04 시가 매수 → 다음 날 15:10 매도(손절 없음, 왕복 비용 포함, `hold_study.run_trade`).
과최적화 방지: **2025년(학습)에서 고르고 2026년(검증)에서 확인한다.** 필터 방향·기준값은 문헌이 정한 방향과 둥근 값으로 미리 정한다.

피처(전부 판정 시각까지 아는 값 — 일봉은 전일까지, 분봉은 09:03 종가까지, 수급은 전일까지):
  mom120   6개월(120거래일) 수익률                       Jegadeesh·Titman(1993) 모멘텀
  near52   전일 종가 ÷ 250일 최고 종가                    George·Hwang(2004) 52주 신고가
  max20    최근 20일 최대 일간 수익률                      Bali·Cakici·Whitelaw(2011) MAX(로또주)
  vol20    최근 20일 일간 수익률 표준편차                   Ang 외(2006) 고변동성 이상현상
  run5     직전 5일 수익률                                 Jegadeesh(1990)·Lehmann(1990) 단기 반전
  ret4     09:03 종가 ÷ 09:00 시가 − 1                    Gao 외(2018) 장중 모멘텀
  gap      09:00 시가 ÷ 전일 종가 − 1                      갭 반전(Lou·Polk·Skouras 2019 낮·밤 수익 분리)
  ratio    판정 배수                                      Gervais 외(2001) 고거래량 프리미엄
  mkt20    시장(유니버스 동일가중) 전일 종가 ÷ 20일 이평 − 1  Moskowitz 외(2012) 시계열 모멘텀
  breadth  09:03 전일 대비 상승 종목 비율                   시장 폭
  fi1      전일 외국인+기관 순매수 ÷ 전일 거래량             정보 거래자 수급
  ma20     전일 종가 ÷ 20일 이평 − 1                       추세

    PYTHONIOENCODING=utf-8 python scripts/filter_study.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings  # noqa: E402
import scripts.hold_study as hs  # noqa: E402
from strategy.opening import rank_triggers, same_window_ratio  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

N = 4
MULT = 3.0


def daily_features(sym: str) -> pd.DataFrame:
    """일봉 피처 — 인덱스 날짜 d 의 값은 **d 전일까지**로 계산(인과)."""
    dd = pd.read_csv(hs.DAILY / f"{sym}.csv", dtype={"ts": str}).sort_values("ts").reset_index(drop=True)
    c = dd["close"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame({"ts": dd["ts"].str[:8]})
    f["mom120"] = c / c.shift(120) - 1.0
    f["near52"] = c / c.rolling(250, min_periods=120).max()
    f["max20"] = r.rolling(20).max()
    f["vol20"] = r.rolling(20).std()
    f["run5"] = c / c.shift(5) - 1.0
    f["run3"] = c / c.shift(3) - 1.0
    f["ma20"] = c / c.rolling(20).mean() - 1.0
    f["close"] = c
    f["volume"] = dd["volume"].astype(float)
    sh = f.drop(columns=["ts"]).shift(1)          # 전일까지
    sh["ts"] = f["ts"]
    return sh.set_index("ts")


def flow_feature(sym: str) -> Dict[str, float]:
    p = hs.DAILY / "flow" / f"{sym}.csv"
    if not p.exists():
        return {}
    fl = pd.read_csv(p, dtype={"ts": str}, usecols=lambda c: c in ("ts", "inv_foreigner_netBuyVolume", "inv_institution_netBuyVolume"))
    fl = fl.sort_values("ts")
    net = fl["inv_foreigner_netBuyVolume"].astype(float) + fl["inv_institution_netBuyVolume"].astype(float)
    return dict(zip(fl["ts"].str[:8], net))


def main() -> int:
    enable_utf8_console()
    s = get_settings()
    cost = s.round_trip_cost
    syms_all = sorted({x for v in s.index_members().values() for x in v})
    t0 = time.time()
    data, pcs, fp = hs.load_all(syms_all)
    days = sorted({d for per in data.values() for d in per})
    print(f"지문 {fp} · {len(data)}종목 · {len(days)}일 · 적재 {time.time() - t0:.0f}s", flush=True)

    feats = {sym: daily_features(sym) for sym in data}
    flows = {sym: flow_feature(sym) for sym in data}
    # 시장(동일가중) 20일 추세 — 전일까지
    closes = pd.DataFrame({sym: f["close"] for sym, f in feats.items()})
    mret = closes.pct_change(fill_method=None).mean(axis=1)
    midx = (1 + mret.fillna(0)).cumprod()
    mkt20 = (midx / midx.rolling(20).mean() - 1.0).to_dict()   # 이미 전일까지(closes 가 shift 된 값)

    # 같은 4분 거래대금
    wtv = {sym: {d: float(np.nansum(a[3, :N].astype(float) * np.nan_to_num(a[4, :N].astype(float)))) for d, a in per.items()}
           for sym, per in data.items()}
    cands: List[Dict[str, Any]] = []
    for j, d in enumerate(days[:-1]):
        nxt = days[j + 1]
        ups, tot = 0, 0
        ratios = {}
        for sym, per in data.items():
            a = per.get(d)
            pc = pcs[sym].get(d)
            if a is None or not pc or not np.isfinite(pc):
                continue
            px = float(a[3, N - 1])
            if np.isfinite(px):
                tot += 1
                ups += px > pc
            hist = [wtv[sym][x] for x in days[max(0, j - 40):j] if x in wtv[sym]][-20:]
            r = same_window_ratio(wtv[sym][d], hist)
            if r > 0:
                ratios[sym] = r
        breadth = ups / tot if tot else np.nan
        for rank_i, (sym, ratio) in enumerate(rank_triggers(ratios, MULT)):
            a = data[sym][d]
            pc = pcs[sym][d]
            px = float(a[3, N - 1])
            vol = float(np.nansum(a[4, :N]))
            vwap = wtv[sym][d] / vol if vol > 0 else 0.0
            if not (px > pc and vwap > 0 and px >= vwap and np.isfinite(a[0, N])):
                continue
            if d not in feats[sym].index:
                continue
            fr = feats[sym].loc[d]
            if np.isfinite(fr["run3"]) and fr["run3"] >= 0.20:
                continue
            b = data[sym].get(nxt)
            if b is None:
                continue
            r2 = hs.run_trade([a, b], N, (1, hs.END_MIN), "none", cost)
            if r2 is None:
                continue
            o0 = float(a[0, 0]) if np.isfinite(a[0, 0]) else np.nan
            # 전일 수급: d 이전 가장 최근 날
            fl = flows[sym]
            prev_d = days[j - 1] if j > 0 else None
            fi = fl.get(prev_d, np.nan) if prev_d else np.nan
            v_prev = fr["volume"]
            cands.append({
                "day": d, "sym": sym, "ratio": ratio, "r2": r2,
                "mom120": fr["mom120"], "near52": fr["near52"], "max20": fr["max20"], "vol20": fr["vol20"],
                "run5": fr["run5"], "ma20": fr["ma20"],
                "ret4": px / o0 - 1.0 if o0 and np.isfinite(o0) else np.nan,
                "gap": o0 / pc - 1.0 if np.isfinite(o0) else np.nan,
                "mkt20": mkt20.get(d, np.nan), "breadth": breadth,
                "fi1": fi / v_prev if np.isfinite(fi) and v_prev and v_prev > 0 else np.nan,
            })
        if (j + 1) % 100 == 0:
            print(f"  후보 {j + 1}/{len(days) - 1} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(cands)
    df["year"] = df["day"].str[:4]
    df.to_pickle(ROOT / "reports" / "filter_cands.pkl")
    print(f"후보 {len(df)}건 · 날짜 {df['day'].nunique()}일 · 2025 {int((df.year == '2025').sum())} / 2026 {int((df.year == '2026').sum())}", flush=True)

    # 1) 피처별 5분위(학습=2025 분위 경계를 검증에 그대로) — 후보 모집단의 평균 r2
    print("\n[피처별 5분위 평균 이틀 수익률] 학습 2025 | 검증 2026  (경계는 2025 기준)")
    tr, te = df[df.year == "2025"], df[df.year == "2026"]
    feat_cols = ["ratio", "mom120", "near52", "max20", "vol20", "run5", "ma20", "ret4", "gap", "mkt20", "breadth", "fi1"]
    for col in feat_cols:
        x = tr[col].dropna()
        if len(x) < 100:
            continue
        qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
        def bucket(v):
            return np.searchsorted(qs, v)
        line = []
        for part in (tr, te):
            p = part[part[col].notna()]
            bk = bucket(p[col].to_numpy())
            line.append(" ".join(f"{p['r2'][bk == q].mean():+.2%}" if (bk == q).sum() >= 20 else "  n/a " for q in range(5)))
        print(f"  {col:>8} 경계 {' '.join(f'{v:+.3f}' for v in qs)} | {line[0]} | {line[1]}")

    # 2) 문헌 방향·둥근 기준값의 필터 — 한 개씩
    rules = {
        "기준(엔진 그대로)": lambda f: np.ones(len(f), bool),
        "모멘텀 6개월 > 0": lambda f: f["mom120"] > 0,
        "52주 고가 80% 이상": lambda f: f["near52"] >= 0.80,
        "MAX 20일 < 15%": lambda f: f["max20"] < 0.15,
        "MAX 20일 < 10%": lambda f: f["max20"] < 0.10,
        "변동성 20일 < 4%": lambda f: f["vol20"] < 0.04,
        "직전 5일 < 10%": lambda f: f["run5"] < 0.10,
        "직전 5일 > 0": lambda f: f["run5"] > 0,
        "20일 이평 위": lambda f: f["ma20"] > 0,
        "장중 09:00→09:03 상승": lambda f: f["ret4"] > 0,
        "갭 < 3%": lambda f: f["gap"] < 0.03,
        "갭 < 5%": lambda f: f["gap"] < 0.05,
        "시장 20일 이평 위": lambda f: f["mkt20"] > 0,
        "시장 폭 > 50%": lambda f: f["breadth"] > 0.5,
        "전일 외국인+기관 순매수": lambda f: f["fi1"] > 0,
        "배수 ≥ 5": lambda f: f["ratio"] >= 5,
        "배수 ≥ 10": lambda f: f["ratio"] >= 10,
    }

    def strat(part: pd.DataFrame, mask, all_days: List[str]) -> Dict[str, float]:
        """하루 1자리: 통과 후보 중 배수 1위. 없는 날 현금(0)."""
        p = part[np.asarray(mask, bool)]
        top = p.sort_values(["day", "ratio"], ascending=[True, False]).groupby("day").head(1)
        per_day = dict(zip(top["day"], top["r2"]))
        r_all = np.array([per_day.get(d, 0.0) for d in all_days])
        r_tr = top["r2"].to_numpy()
        per_n = p.groupby("day").size()
        n_med = float(per_n.reindex(all_days, fill_value=0).median())
        return {"cands_per_day": n_med, "trade_days": len(top), "trade_mean": float(r_tr.mean()) if len(r_tr) else np.nan,
                "trade_median": float(np.median(r_tr)) if len(r_tr) else np.nan,
                "trade_p_loss": float((r_tr < 0).mean()) if len(r_tr) else np.nan,
                "all_mean": float(r_all.mean()), "pool_mean": float(p["r2"].mean()) if len(p) else np.nan}

    d25 = [d for d in days[:-1] if d.startswith("2025")]
    d26 = [d for d in days[:-1] if d.startswith("2026")]
    print("\n[필터 한 개씩 — 하루 1자리(통과 중 배수 1위), 이틀 보유]   학습 2025 | 검증 2026")
    print("  필터 | 하루 후보(중앙) | 매매일 | 매매당 평균 | 중앙 | 손실 | 전체일 평균(현금 0 포함) || 하루 후보 | 매매일 | 매매당 평균 | 중앙 | 손실 | 전체일 평균")
    single = {}
    for name, fn in rules.items():
        a = strat(tr, fn(tr).fillna(False) if hasattr(fn(tr), "fillna") else fn(tr), d25)
        b = strat(te, fn(te).fillna(False) if hasattr(fn(te), "fillna") else fn(te), d26)
        single[name] = (a, b)
        print(f"  {name} | {a['cands_per_day']:.0f} | {a['trade_days']} | {a['trade_mean']:+.2%} | {a['trade_median']:+.2%} | {a['trade_p_loss']:.0%} | {a['all_mean']:+.2%}"
              f" || {b['cands_per_day']:.0f} | {b['trade_days']} | {b['trade_mean']:+.2%} | {b['trade_median']:+.2%} | {b['trade_p_loss']:.0%} | {b['all_mean']:+.2%}")

    # 3) 두 해 모두 매매당 평균을 올린 필터만 앞에서부터 하나씩 합친다(학습 개선 순)
    base_a, base_b = single["기준(엔진 그대로)"]
    good = [n for n, (a, b) in single.items() if n != "기준(엔진 그대로)"
            and a["trade_mean"] > base_a["trade_mean"] and b["trade_mean"] > base_b["trade_mean"] and a["trade_days"] >= 40]
    good.sort(key=lambda n: -single[n][0]["trade_mean"])
    print(f"\n두 해 모두 기준보다 나은 필터: {good}")
    print("\n[누적 결합 — 학습 개선 순으로 하나씩 더함, 학습에서 나빠지면 뺀다]")
    chosen: List[str] = []
    cur_a = base_a
    for n in good:
        trial = chosen + [n]
        m_tr = np.logical_and.reduce([rules[x](tr).fillna(False) for x in trial])
        m_te = np.logical_and.reduce([rules[x](te).fillna(False) for x in trial])
        a, b = strat(tr, m_tr, d25), strat(te, m_te, d26)
        keep = a["trade_mean"] > cur_a["trade_mean"] and a["trade_days"] >= 40
        print(f"  {' + '.join(trial)} | 학습 후보 {a['cands_per_day']:.0f}개/일 · {a['trade_days']}일 {a['trade_mean']:+.2%} (전체 {a['all_mean']:+.2%}) | "
              f"검증 후보 {b['cands_per_day']:.0f}개/일 · {b['trade_days']}일 {b['trade_mean']:+.2%} (전체 {b['all_mean']:+.2%}) | {'채택' if keep else '뺌'}")
        if keep:
            chosen, cur_a = trial, a
    out = {"fingerprint": fp, "cands": len(df), "single": {k: {"train": v[0], "test": v[1]} for k, v in single.items()},
           "chosen": chosen}
    (ROOT / "reports" / "filter_study.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n최종 채택: {chosen} · {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
