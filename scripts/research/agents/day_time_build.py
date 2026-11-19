"""시간대 효과 연구용 표(개발 구간만) — 날마다 Ctx(i).bars(391)로 체크포인트 값을 뽑는다(분석 전용, 결정 함수는 쓰지 않음).

출력: reports/agents/day_time_tab.npz
  O[d,s,c]  체크포인트 c 분의 시가(매수·매도 가격)    C[d,s,c] c-1 분 종가(그 시각까지 본 마지막 가격)
  HI/LO[d,s,c] 0..c-1 고가·저가   TV[d,s,c] 0..c-1 누적 거래대금   VW = 누적 VWAP
  T20[d,s] 전일까지 20일 평균 거래대금(일봉)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.day_eval import Ctx, dev_days  # noqa: E402

CP = [0, 5, 15, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 340, 350, 360, 370]


def main():
    days = dev_days()
    D, S, NC = len(days), None, len(CP)
    out = {}
    for n, i in enumerate(days):
        ctx = Ctx(i)
        b = ctx.bars(391).astype(np.float64)  # [S,5,391]
        S = b.shape[0]
        if not out:
            for k in ("O", "C", "HI", "LO", "TV", "VW"):
                out[k] = np.full((D, S, NC), np.nan)
            out["PC"] = np.full((D, S), np.nan)
            out["T20"] = np.full((D, S), np.nan)
        o, h, l, c, v = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
        tp = np.where(np.isfinite(c), (h + l + c) / 3, np.nan)
        tv = np.nancumsum(np.nan_to_num(tp) * np.nan_to_num(v), axis=1)
        cv = np.nancumsum(np.nan_to_num(v), axis=1)
        ch = np.fmax.accumulate(np.where(np.isfinite(h), h, -np.inf), axis=1)
        cl = np.fmin.accumulate(np.where(np.isfinite(l), l, np.inf), axis=1)
        # 마지막 유효 종가(전방 채움)
        cf = c.copy()
        for t in range(1, cf.shape[1]):
            m = ~np.isfinite(cf[:, t])
            cf[m, t] = cf[m, t - 1]
        for ci, m in enumerate(CP):
            out["O"][n, :, ci] = o[:, m] if m < 391 else np.nan
            if m > 0:
                out["C"][n, :, ci] = cf[:, m - 1]
                out["HI"][n, :, ci] = np.where(np.isfinite(ch[:, m - 1]), ch[:, m - 1], np.nan)
                out["LO"][n, :, ci] = np.where(np.isfinite(cl[:, m - 1]), cl[:, m - 1], np.nan)
                out["TV"][n, :, ci] = tv[:, m - 1]
                out["VW"][n, :, ci] = np.where(cv[:, m - 1] > 0, tv[:, m - 1] / np.maximum(cv[:, m - 1], 1), np.nan)
        out["PC"][n] = ctx.prev_close
        if n % 20 == 0:
            for j in range(S):
                d = ctx.daily(j, 20)
                if len(d):
                    out["T20"][n, j] = float((d["close"] * d["volume"]).mean())
        else:
            out["T20"][n] = out["T20"][n - 1]
        if n % 50 == 0:
            print(n, ctx.day, flush=True)
    _, meta = __import__("scripts.research.day_eval", fromlist=["_load_all"])._load_all()
    np.savez(ROOT / "reports" / "agents" / "day_time_tab.npz", days=np.array([ctx_day for ctx_day in [str(meta["days"][i]) for i in days]]),
             cp=np.array(CP), kosdaq=meta["kosdaq"], **out)


if __name__ == "__main__":
    main()
