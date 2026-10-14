"""종목 현재가 응답의 시장경보·상태 필드를 찍는다 — 모의투자 '매매불가 종목'(40070000)의 원인 확인용.

    python scripts/probe_warn.py 278650 396300 105740          # 몇 종목만
    python scripts/probe_warn.py --universe --out state/stock_status.json   # 유니버스 전체(초당 1건)

2026-10-14 확인: 모의투자에서 거부된 5종목은 모두 증거금률 100%(iscd_stat_cls_code 57 · 경고 53),
체결된 2종목은 40%(55, 신용 가능)였다. 엔진과 같은 앱 키·유량을 쓴다 — 장 마감 후에 돌린다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.factory import create_broker  # noqa: E402
from config.settings import get_settings  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

FIELDS = ("iscd_stat_cls_code", "mrkt_warn_cls_code", "short_over_yn", "invt_caful_yn",
          "temp_stop_yn", "mang_issu_cls_code", "crdt_able_yn", "marg_rate", "rprs_mrkt_kor_name")


async def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="종목 상태 필드 조회")
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--universe", action="store_true", help="설정의 유니버스 전체")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    s = get_settings()
    syms = list(s.UNIVERSE) if args.universe else args.symbols
    broker = create_broker(s)
    await broker.connect()
    res = {}
    try:
        for k, sym in enumerate(syms, 1):
            try:
                payload = await broker.inquire_price(sym)
            except Exception as exc:
                print(f"{sym} 실패: {exc}")
                continue
            out = payload.get("output") or {}
            res[sym] = {f: out.get(f, "") for f in FIELDS}
            if not args.universe:
                print(f"{sym} [{payload.get('msg_cd')}] " +
                      " · ".join(f"{f}={res[sym][f]}" for f in FIELDS))
            elif k % 50 == 0:
                print(f"{k}/{len(syms)}")
    finally:
        await broker.close()
    if args.universe:
        print("상태 코드:", dict(Counter(v["iscd_stat_cls_code"] for v in res.values())))
        print("증거금률:", dict(Counter(v["marg_rate"] for v in res.values())))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장 → {args.out} ({len(res)}종목)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
