"""기존 유니버스 파일에서 거래대금 하한만 낮춘 파일을 만든다(다른 필터·제외 사유는 그대로).

`refresh_universe.py`를 다시 돌리면 KRX CSV 기준일·토스 확인을 처음부터 다시 하고, 모의투자
매매불가(증거금 100%) 제외가 사라진다. 그래서 **거래대금 때문에만 빠진 종목** 중 새 하한 이상인
것만 되살린다. 되살린 종목은 `added`에 남긴다. 모의투자 매매불가는 장 마감 후 상태를 조회해
`--status`로 넘기면 뺀다(엔진과 같은 앱 키·유량이라 장중에는 돌리지 않는다).

    python scripts/widen_universe.py --base config/universe_10b.json \\
        --min-trading-value 5e8 --out config/universe_5b.json
    # 서버에서, 장 마감 후: 추가 종목 상태 조회
    #   .venv/bin/python scripts/probe_warn.py <추가 종목들> --out state/stock_status_5b.json
    python scripts/widen_universe.py --status state/stock_status_5b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.eligibility import vts_tradable  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="config/universe_10b.json")
    ap.add_argument("--min-trading-value", type=float, default=5e8)
    ap.add_argument("--out", default="config/universe_5b.json")
    ap.add_argument("--status", default="", help="probe_warn.py --out 결과(추가 종목 상태) — 매매불가를 뺀다")
    args = ap.parse_args()

    blob = json.loads(Path(args.base).read_text(encoding="utf-8"))
    tv = blob.get("trading_value", {})
    excluded = dict(blob.get("excluded", {}))
    added = sorted(
        (sym for sym, why in excluded.items()
         if str(why).startswith("거래대금") and float(tv.get(sym, 0.0)) >= args.min_trading_value),
        key=lambda s: -float(tv.get(s, 0.0)))
    status = json.loads(Path(args.status).read_text(encoding="utf-8")) if args.status else {}
    blocked = [s for s in added if s in status and not vts_tradable(status[s])]
    for sym in added:
        excluded.pop(sym)
    for sym in blocked:
        st = status[sym]
        excluded[sym] = (f"모의투자 매매불가(상태 {st.get('iscd_stat_cls_code')} · 증거금 "
                         f"{st.get('marg_rate')}%, {args.status} 조회)")
    added = [s for s in added if s not in blocked]

    blob["tradable"] = list(blob.get("tradable", [])) + added
    blob["excluded"] = excluded
    blob["added"] = {sym: f"거래대금 {float(tv.get(sym, 0.0)) / 1e8:.1f}억 — 하한 "
                          f"{args.min_trading_value / 1e8:g}억으로 낮춰 추가"
                          + ("" if sym in status else "(모의투자 상태 미확인)")
                     for sym in added}
    blob.setdefault("filters", {})["min_trading_value"] = args.min_trading_value
    blob["widened_from"] = args.base
    Path(args.out).write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.base} {len(blob['tradable']) - len(added)}종목 + 추가 {len(added)}종목 "
          f"(거래대금 {args.min_trading_value / 1e8:g}억 이상 · 매매불가 제외 {len(blocked)}) "
          f"→ {args.out} {len(blob['tradable'])}종목")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
