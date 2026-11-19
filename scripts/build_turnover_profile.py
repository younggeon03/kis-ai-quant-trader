"""장중 VWAP 청산의 기준선 — 종목별 '평소 같은 시각 직전 W분 거래대금'(지난 20일 중앙) 표를 만든다.

측정(`scripts/research/takeprofit_study.py --vwap`)과 같은 정의다: 하루 전체 분봉에서 분 단위 거래대금을 만들고
(`strategy.sliding.minute_turnover`), 분 m 에서 직전 W분 합(`rolling_sum`)을 낸 뒤, 마지막 20일
같은 분의 중앙값(최소 10일, `same_time_base`)을 기준선으로 쓴다.

실서버에는 하루 전체 분봉이 없어(디스크 10GB) 노트북에서 만들어 보낸다. 결과는 종목당 380개 숫자라 작다.
분봉 수집(`collect_opening.py --until 1530 --dir fullday`) 뒤에 다시 만들어 보내면 기준선이 최신이 된다.

    UNIVERSE_FILE=config/universe_10b.json python scripts/build_turnover_profile.py
    → data/cache/turnover_profile_w10.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings  # noqa: E402
from strategy.opening import minute_index  # noqa: E402
from strategy.sliding import minute_turnover, rolling_sum, same_time_base  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="같은 시각 직전 W분 거래대금 기준선 표")
    ap.add_argument("--fullday-dir", default="data/cache/fullday")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    s = get_settings()
    fdir = Path(args.fullday_dir)
    out_path = Path(args.out or f"data/cache/turnover_profile_w{args.window}.json")
    prof, last_days = {}, []
    for sym in s.UNIVERSE:
        p = fdir / f"{sym}.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p, dtype={"ts": str})
        m = m.assign(day=m["ts"].str[:8])
        days = sorted(m["day"].unique())[-args.days:]
        rows = []
        for dy in days:
            g = m[m["day"] == dy]
            tv = minute_turnover(minute_index(g["ts"]), g["close"].to_numpy(float),
                                 g["volume"].to_numpy(float))
            rows.append(rolling_sum(tv, args.window))
        if not rows:
            continue
        base = same_time_base(np.vstack(rows), days=args.days)
        prof[sym] = [None if not np.isfinite(x) else round(float(x)) for x in base]
        last_days.append(days[-1])
    payload = {"window": args.window, "days": args.days,
               "as_of": max(last_days) if last_days else "", "profile": prof}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"저장 → {out_path} · {len(prof)}종목 · 최신 {payload['as_of']} · "
          f"{out_path.stat().st_size / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
