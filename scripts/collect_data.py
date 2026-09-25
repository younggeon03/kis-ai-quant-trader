"""토스증권 Open API에서 학습·백테스트용 과거 데이터를 수집한다.

    python scripts/collect_data.py --probe          # 연결 확인 + 응답 필드명 출력 (먼저 이것부터)
    python scripts/collect_data.py --symbols 3      # 3종목만 시험 수집
    python scripts/collect_data.py                  # 유니버스 전체 수집
    python scripts/collect_data.py --refresh        # 증분 무시하고 전체 재수집
    python scripts/collect_data.py --no-flow        # 일봉만 수집

저장 위치
    data/cache/<종목코드>.csv        OHLCV  (ts, open, high, low, close, volume)
    data/cache/flow/<종목코드>.csv   수급    (ts, inv_*, short_*, credit_*, lend_*, prog_*)

두 파일 모두 `.gitignore` 대상이다. 저장소에 올리지 않는다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()

import pandas as pd  # noqa: E402

from api.toss_data import TossAPIError, TossDataClient  # noqa: E402
from config.settings import get_settings  # noqa: E402
from utils.logger import logger, setup_logging  # noqa: E402


def _existing_span(path: Path) -> tuple[int, str, str]:
    """이미 받아 둔 파일의 (행 수, 시작일, 종료일)."""
    if not path.exists():
        return 0, "", ""
    try:
        df = pd.read_csv(path, dtype={"ts": str})
        if df.empty:
            return 0, "", ""
        return len(df), str(df["ts"].iloc[0]), str(df["ts"].iloc[-1])
    except Exception:
        return 0, "", ""


async def run_probe(settings) -> int:
    """연결과 응답 필드명을 확인한다. 실제 수집 전에 반드시 한 번 돌린다."""
    async with TossDataClient(settings) as client:
        result = await client.probe()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result.get("ok"):
        logger.error("[Collect] 연결 점검 실패 — 위 출력을 확인하세요.")
        return 1
    logger.success("[Collect] 연결 정상. 위 필드명이 파서가 찾는 이름과 다르면 알려주세요.")
    return 0


async def collect(settings, symbols: list[str], bars: int, flow_days: int,
                  with_flow: bool, refresh: bool) -> int:
    cache = Path(settings.DATA_CACHE_DIR)
    flow_dir = Path(settings.flow_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    ok, skipped, failed = 0, 0, []
    started = time.monotonic()

    async with TossDataClient(settings) as client:
        for i, symbol in enumerate(symbols, 1):
            price_path = cache / f"{symbol}.csv"
            have, first, last = _existing_span(price_path)
            if have >= bars and not refresh:
                skipped += 1
                logger.info(f"[{i}/{len(symbols)}] {symbol} 건너뜀 ({have}봉 {first}~{last})")
                continue

            try:
                df = await client.fetch_candles(symbol, count=bars)
                if df.empty:
                    failed.append((symbol, "일봉 없음"))
                    continue
                df.to_csv(price_path, index=False, encoding="utf-8")
                msg = f"[{i}/{len(symbols)}] {symbol} 일봉 {len(df)}봉 ({df['ts'].iloc[0]}~{df['ts'].iloc[-1]})"

                if with_flow:
                    flow = await client.fetch_flow(symbol, days=flow_days)
                    if not flow.empty and len(flow.columns) > 1:
                        flow.to_csv(flow_dir / f"{symbol}.csv", index=False, encoding="utf-8")
                        msg += f" · 수급 {len(flow)}일 {len(flow.columns) - 1}컬럼"
                    else:
                        msg += " · 수급 없음"
                logger.info(msg)
                ok += 1
            except TossAPIError as exc:
                failed.append((symbol, str(exc)[:80]))
                logger.warning(f"[{i}/{len(symbols)}] {symbol} 실패: {exc}")
            except Exception as exc:
                failed.append((symbol, f"{type(exc).__name__}: {exc}"[:80]))
                logger.warning(f"[{i}/{len(symbols)}] {symbol} 예외: {exc}")

    elapsed = time.monotonic() - started
    print()
    print("=" * 70)
    print(f" 수집 완료 : 성공 {ok} · 건너뜀 {skipped} · 실패 {len(failed)}")
    print(f" 소요 시간 : {elapsed:.1f}초")
    print(f" 저장 위치 : {cache}")
    if failed:
        print(" 실패 종목 :")
        for symbol, reason in failed[:10]:
            print(f"   {symbol}  {reason}")
    print("=" * 70)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="토스증권 과거 데이터 수집")
    parser.add_argument("--probe", action="store_true", help="연결·응답 필드명 확인만 수행")
    parser.add_argument("--symbols", type=int, default=0, help="앞에서 N종목만 수집 (0=전체)")
    parser.add_argument("--bars", type=int, default=None, help="종목당 일봉 수")
    parser.add_argument("--flow-days", type=int, default=None, help="종목당 수급 일수")
    parser.add_argument("--no-flow", action="store_true", help="수급 데이터 건너뛰기")
    parser.add_argument("--refresh", action="store_true", help="이미 받은 종목도 다시 수집")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_DIR, settings.LOG_LEVEL)

    if not settings.toss_ready():
        logger.error(
            "[Collect] TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 이 비어 있습니다.\n"
            "          .env 파일에 아래 두 줄을 채운 뒤 다시 실행하세요.\n"
            "            TOSS_CLIENT_ID=발급받은_클라이언트_ID\n"
            "            TOSS_CLIENT_SECRET=발급받은_시크릿"
        )
        return 1

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.probe:
        return asyncio.run(run_probe(settings))

    symbols = settings.UNIVERSE[: args.symbols] if args.symbols else settings.UNIVERSE
    bars = args.bars or settings.COLLECT_BARS
    flow_days = args.flow_days or settings.COLLECT_FLOW_DAYS
    logger.info(f"[Collect] {len(symbols)}종목 · 일봉 {bars}봉 · 수급 {flow_days}일"
                + (" (수급 생략)" if args.no_flow else ""))
    return asyncio.run(collect(settings, symbols, bars, flow_days,
                               not args.no_flow, args.refresh))


if __name__ == "__main__":
    raise SystemExit(main())
