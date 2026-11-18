"""에이전트 팀용 공용 데이터(11/18) — 모든 지수 종목 x 모든 날의 09:03 피처 + 이틀 결과, 판정 통과 후보의 분 경로.

결과 정의(two_day_study 의 B 와 같다): d1 09:04 시가 매수 → 분봉 고가가 매수가 x 1.10/(1-비용)에 닿으면 그 가격(시가가 위면 시가)
→ 아니면 d2 15:10 시가. 수익 = 매도가/매수가 x (1-비용) - 1. hit10 = 수익 ≥ +10%(-1e-9).
출력:
  reports/agent_data/rows.pkl   — 행 = (day, sym). 피처 + 결과(rB, hit10, r_d1c, r_d2c, up_max, dn_min, up_d1, dn_d1, ...)
  reports/agent_data/paths.npz  — engine_pass 행의 분 경로 [n, 2, 4, 391](시가·고가·저가·종가), 키 = rows 의 path_idx
피처는 전부 09:03 종가까지·전일 일봉까지로 계산한다(룩어헤드 없음).

    PYTHONIOENCODING=utf-8 python scripts/build_agent_data.py
"""
from __future__ import annotations

import csv
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
from scripts.filter_study import daily_features, flow_feature  # noqa: E402
from strategy.opening import same_window_ratio  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

N = 4
END = hs.END_MIN


def main() -> int:
    enable_utf8_console()
    s = get_settings()
    cost = s.round_trip_cost
    tp_px_mult = 1.10 / (1 - cost)
    syms_all = sorted({x for v in s.index_members().values() for x in v})
    t0 = time.time()
    data, pcs, fp = hs.load_all(syms_all)
    days = sorted({d for per in data.values() for d in per})
    print(f"지문 {fp} · {len(data)}종목 · {len(days)}일 · 적재 {time.time() - t0:.0f}s", flush=True)
    feats = {sym: daily_features(sym) for sym in data}
    flows = {sym: flow_feature(sym) for sym in data}
    closes = pd.DataFrame({sym: f["close"] for sym, f in feats.items()})
    mret = closes.pct_change(fill_method=None).mean(axis=1)
    midx = (1 + mret.fillna(0)).cumprod()
    mkt20 = (midx / midx.rolling(20).mean() - 1.0).to_dict()
    mkt_vol20 = mret.rolling(20).std().to_dict()
    kind = ROOT / "data" / "krx" / "kind_corplist_20261118.csv"
    rows_ = list(csv.reader(open(kind, encoding="utf-8")))
    ind_of = {r[2].zfill(6): r[3] for r in rows_[1:] if len(r) > 3}
    blob = s._universe_blob()
    mcap_of, shares_of, mkt_of = blob.get("market_cap") or {}, blob.get("shares") or {}, blob.get("sectors") or {}
    wtv = {sym: {d: float(np.nansum(a[3, :N].astype(float) * np.nan_to_num(a[4, :N].astype(float)))) for d, a in per.items()}
           for sym, per in data.items()}

    out: List[Dict[str, Any]] = []
    paths: List[np.ndarray] = []
    for j, d in enumerate(days[:-1]):
        nxt = days[j + 1]
        day_rows = []
        for sym, per in data.items():
            a = per.get(d)
            b = per.get(nxt)
            pc = pcs[sym].get(d)
            if a is None or b is None or not pc or not np.isfinite(pc) or d not in feats[sym].index:
                continue
            entry = float(a[0, N])
            px3 = float(a[3, N - 1])
            if not (np.isfinite(entry) and entry > 0 and np.isfinite(px3) and px3 > 0):
                continue
            # 결과(B) + 경로 요약
            tpx = entry * tp_px_mult
            hit, rB, up_max, dn_min = False, np.nan, 0.0, 0.0
            up_d1, dn_d1 = 0.0, 0.0
            for dd_, arr in ((0, a), (1, b)):
                m0 = N if dd_ == 0 else 0
                hi = arr[1, m0:END].astype(float)
                lo = arr[2, m0:END].astype(float)
                op = arr[0, m0:END].astype(float)
                if dd_ == 0:
                    op = op.copy()
                    op[0] = entry
                if np.isfinite(hi).any():
                    up_max = max(up_max, float(np.nanmax(hi)) / entry - 1.0)
                    dn_min = min(dn_min, float(np.nanmin(lo)) / entry - 1.0)
                    if dd_ == 0:
                        up_d1, dn_d1 = up_max, dn_min
                if not hit:
                    idx = np.flatnonzero(np.isfinite(hi) & (hi >= tpx))
                    if len(idx):
                        k = idx[0]
                        fill = op[k] if np.isfinite(op[k]) and op[k] >= tpx else tpx
                        rB = fill / entry * (1 - cost) - 1.0
                        hit = True
            px_end = float(b[0, END]) if np.isfinite(b[0, END]) else float(b[3, END - 1])
            r_d2 = px_end / entry * (1 - cost) - 1.0
            if not hit:
                rB = r_d2
            c_d1 = float(a[3, 380]) if np.isfinite(a[3, 380]) else float(a[3, END])
            fr = feats[sym].loc[d]
            hist = [wtv[sym][x] for x in days[max(0, j - 40):j] if x in wtv[sym]][-20:]
            ratio = same_window_ratio(wtv[sym][d], hist)
            vol4 = float(np.nansum(a[4, :N]))
            vwap = wtv[sym][d] / vol4 if vol4 > 0 else np.nan
            o0 = float(a[0, 0]) if np.isfinite(a[0, 0]) else np.nan
            prev_d = days[j - 1] if j > 0 else None
            fi = flows[sym].get(prev_d, np.nan) if prev_d else np.nan
            v_prev = fr["volume"]
            run3 = fr["run3"]
            engine_pass = bool(ratio >= 3.0 and px3 > pc and np.isfinite(vwap) and px3 >= vwap
                               and not (np.isfinite(run3) and run3 >= 0.20))
            row = {
                "day": d, "next_day": nxt, "sym": sym, "ind": ind_of.get(sym, "?"),
                "kosdaq": 1.0 if mkt_of.get(sym) == "KOSDAQ" else 0.0,
                "entry": entry, "prev_close": pc, "px0903": px3,
                # 결과
                "rB": rB, "hit10": bool(rB >= 0.10 - 1e-9), "r_d2c": r_d2, "r_d1c": c_d1 / entry * (1 - cost) - 1.0,
                "up_max": up_max, "dn_min": dn_min, "up_d1": up_d1, "dn_d1": dn_d1,
                # 엔진 판정
                "ratio": ratio, "engine_pass": engine_pass, "tv4": wtv[sym][d],
                # 장중(09:00~09:03)
                "chg": px3 / pc - 1.0, "gap": o0 / pc - 1.0 if np.isfinite(o0) else np.nan,
                "ret4": px3 / o0 - 1.0 if np.isfinite(o0) and o0 > 0 else np.nan,
                "range4": (float(np.nanmax(a[1, :N])) - float(np.nanmin(a[2, :N]))) / o0 if np.isfinite(o0) and o0 > 0 else np.nan,
                "vwap_dev": px3 / vwap - 1.0 if np.isfinite(vwap) and vwap > 0 else np.nan,
                # 일봉(전일까지)
                **{k: float(fr[k]) for k in ("mom120", "mom20", "near52", "max20", "vol20", "run5", "run3", "ma20", "rng1", "atr10", "absr1")},
                "prev_volume": float(v_prev),
                "fi1": fi / v_prev if np.isfinite(fi) and v_prev and v_prev > 0 else np.nan,
                "mcap": float(mcap_of.get(sym) or np.nan),
                "turn1": float(v_prev) / float(shares_of[sym]) if shares_of.get(sym) else np.nan,
                "mkt20": mkt20.get(d, np.nan), "mkt_vol20": mkt_vol20.get(d, np.nan),
                "path_idx": -1,
            }
            if engine_pass:
                row["path_idx"] = len(paths)
                paths.append(np.stack([a[:4], b[:4]]).astype(np.float32))
            day_rows.append(row)
        # 그날 전체로 만드는 피처: 시장 폭 · 업종 동반 상승 · 판정 통과 수 · 같은 업종 판정 통과 수
        if day_rows:
            up = np.array([r["chg"] > 0 for r in day_rows])
            breadth = float(up.mean())
            by_ind: Dict[str, List[int]] = {}
            for i_, r in enumerate(day_rows):
                by_ind.setdefault(r["ind"], []).append(i_)
            n_pass = sum(r["engine_pass"] for r in day_rows)
            for r in day_rows:
                r["breadth"] = breadth
                r["n_pass_day"] = n_pass
                mates = [day_rows[k] for k in by_ind[r["ind"]] if day_rows[k] is not r]
                r["ind_up"] = float(np.mean([m["chg"] > 0 for m in mates])) if mates else np.nan
                r["ind_chg"] = float(np.mean([m["chg"] for m in mates])) if mates else np.nan
                r["ind_pass"] = int(sum(m["engine_pass"] for m in mates))
            out += day_rows
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(days) - 1} ({time.time() - t0:.0f}s) · 행 {len(out)} · 경로 {len(paths)}", flush=True)
    df = pd.DataFrame(out)
    od = ROOT / "reports" / "agent_data"
    od.mkdir(parents=True, exist_ok=True)
    df.to_pickle(od / "rows.pkl")
    np.savez_compressed(od / "paths.npz", paths=np.stack(paths) if paths else np.zeros((0, 2, 4, 391), np.float32))
    (od / "README.txt").write_text(
        f"지문 {fp} · 행 {len(df)} · 판정 통과 {int(df.engine_pass.sum())} · 기간 {days[0]}~{days[-2]} · 왕복 비용 {cost}\n"
        "rows.pkl 피처 정의는 scripts/build_agent_data.py 참고. paths.npz['paths'][path_idx] = [2일, (시가·고가·저가·종가), 391분], "
        "분 인덱스 0 = 09:00, 4 = 09:04(매수), 370 = 15:10(매도).\n", encoding="utf-8")
    print(f"저장 {od} · 행 {len(df)} · 판정 통과 {int(df.engine_pass.sum())} · hit10 비율 전체 {df.hit10.mean():.1%} · "
          f"통과 {df[df.engine_pass].hit10.mean():.1%} · {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
