"""KIS 순위 API 가 지금 서버(모의/실전)에서 되는지, 무엇을 돌려주는지 확인한다.

장중 감지 지연을 줄이려는 후보다 — 494종목을 복수시세로 훑는 대신(17호출) **시장 전체 거래 급증
종목을 1호출**로 받을 수 있는지 본다. 순위 API 는 모의투자에서 막혀 있는 경우가 많아 먼저 확인한다.

    python scripts/probe_rank.py            # .env 의 서버(지금은 모의)
    python scripts/probe_rank.py --raw      # 첫 줄의 필드 전체를 찍는다

엔진과 같은 앱 키·유량(초당 1건)을 나눠 쓴다 — **장 마감 후**에 돌린다. 호출은 기준마다 1번씩이다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.base import is_ok  # noqa: E402
from api.factory import create_broker  # noqa: E402
from config.settings import get_settings  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

# 거래량순위(v1_국내주식-047). FID_BLNG_CLS_CODE: 0 평균거래량 · 1 거래증가율 · 2 평균거래회전율
#                                             · 3 거래금액순 · 4 평균거래금액회전율
VOLUME_RANK = ("/uapi/domestic-stock/v1/quotations/volume-rank", "FHPST01710000")
BLNG = {"1": "거래증가율", "3": "거래금액순", "0": "평균거래량"}


async def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="KIS 순위 API 확인")
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    broker = create_broker(s)
    await broker.connect()
    print(f"서버: {'모의투자' if not s.is_prod else '실전'} ({s.KIS_ENV})")
    try:
        for code, label in BLNG.items():
            params = {
                "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "0", "FID_BLNG_CLS_CODE": code,
                "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "", "FID_INPUT_PRICE_2": "", "FID_VOL_CNT": "",
                "FID_INPUT_DATE_1": "",
            }
            t0 = time.perf_counter()
            payload = await broker._request("GET", VOLUME_RANK[0], VOLUME_RANK[1],  # type: ignore[attr-defined]
                                            params=params)
            dt_ms = (time.perf_counter() - t0) * 1000
            rows = payload.get("output") or []
            print(f"\n[{label}] ok={is_ok(payload)} · {payload.get('msg_cd')} {payload.get('msg1')} · "
                  f"{len(rows)}줄 · {dt_ms:.0f}ms")
            for r in rows[:10]:
                print(f"  {r.get('data_rank', ''):>3} {r.get('mksc_shrn_iscd', ''):>6} "
                      f"{r.get('hts_kor_isnm', '')[:10]:<10} 현재가 {r.get('stck_prpr', '')} "
                      f"등락 {r.get('prdy_ctrt', '')}% · 거래량 {r.get('acml_vol', '')} · "
                      f"증가율 {r.get('vol_inrt', '')}% · 대금 {r.get('acml_tr_pbmn', '')}")
            if args.raw and rows:
                print(json.dumps(rows[0], ensure_ascii=False, indent=1))
    finally:
        await broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
