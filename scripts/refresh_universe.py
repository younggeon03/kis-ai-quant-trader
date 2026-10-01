"""코스피200 · 코스닥150 구성종목을 받아 `config/universe.json`으로 저장한다.

## 왜 별도 스크립트인가

지수 구성종목은 **런타임 데이터가 아니라 참조 데이터**다. 정기변경이 6월·12월 두 번뿐이고
대회 기간(30거래일) 중에는 바뀌지 않는다. 런타임에 가져오면 이득 없이 실패 지점만 늘어난다.

그래서 pykrx를 import하는 파일은 **이 스크립트 하나뿐**이고, 매매 런타임은 산출물인
`config/universe.json`만 읽는다. `lightgbm`·`loguru`와 같은 취급이다 — 있으면 쓰고
없으면 기존 동작(`DEFAULT_UNIVERSE` 40종목)으로 폴백한다.

## 토스 교차 확인을 하는 이유

지수에 속해도 토스 API로 일봉을 받을 수 없으면 학습·백테스트에 쓸 수 없다. 신규 상장,
거래정지, 종목코드 변경 등이 있다. 그래서 받아온 종목마다 실제로 시세가 오는지 확인하고
안 오는 종목은 사유와 함께 `excluded`에 남긴다. 조용히 빠뜨리면 나중에 원인을 찾기 어렵다.

## 사용 (실제로 쓰는 경로)

pykrx 1.2.9의 목록성 조회는 KRX 계정 로그인을 요구하도록 바뀌어 인증 없이는 빈 응답이
온다(`get_index_portfolio_deposit_file`, `get_market_ticker_list`, `get_market_cap` 모두).
그래서 **KRX CSV 경로가 기본**이다. 원본은 `data/krx/`에 둔다.

    python scripts/refresh_universe.py \\
        --all-stocks data/krx/data_5347_20260929.csv \\
        --kospi200   data/krx/data_3917_20260929.csv \\
        --kosdaq150  data/krx/data_3936_20260929.csv \\
        --min-trading-value 3e9

CSV는 `data.krx.co.kr`에서 내린다.
- 전종목시세: 기본통계 → 주식 → 종목시세 → 전종목 시세 (14컬럼)
- 지수구성종목: 기본통계 → 지수 → 주가지수 → 지수구성종목 (6컬럼)

옵션: `--no-verify`(토스 확인 생략), `--date YYYYMMDD`(pykrx 경로의 기준일).

산출물을 쓰려면 `.env`에 `UNIVERSE_FILE=config/universe.json`을 넣는다.
지수 정기변경(6월·12월) 때 CSV를 다시 내려 재실행한다.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()

from config.settings import get_settings  # noqa: E402
from utils.logger import logger, setup_logging  # noqa: E402

try:
    from pykrx import stock as _krx
    HAS_PYKRX = True
except ImportError:                                    # pragma: no cover
    _krx = None
    HAS_PYKRX = False

# KRX 지수 코드
INDEX_CODES: Dict[str, str] = {
    "KOSPI200": "1028",
    "KOSDAQ150": "2203",
}


def _latest_business_day() -> str:
    """KRX 기준 최근 영업일(YYYYMMDD). 주말이면 금요일로 물린다."""
    day = dt.date.today()
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day.strftime("%Y%m%d")


def fetch_from_pykrx(date: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """지수별 구성종목과 종목명을 받는다."""
    if not HAS_PYKRX:
        raise RuntimeError(
            "pykrx가 설치되지 않았습니다. `pip install pykrx` 후 다시 실행하거나, "
            "KRX에서 내린 CSV를 --from-csv 로 넘기십시오.")

    indices: Dict[str, List[str]] = {}
    names: Dict[str, str] = {}
    for label, code in INDEX_CODES.items():
        try:
            tickers = list(_krx.get_index_portfolio_deposit_file(code))
        except Exception as exc:
            raise RuntimeError(f"{label}({code}) 구성종목 조회 실패: {exc}") from exc
        # 6자리 숫자만 남긴다(응답에 이질적인 값이 섞이는 경우가 있다)
        tickers = [t for t in (str(x).strip() for x in tickers)
                   if len(t) == 6 and t.isdigit()]
        indices[label] = sorted(set(tickers))
        logger.info(f"[Universe] {label}: {len(indices[label])}종목")

    for sym in {s for lst in indices.values() for s in lst}:
        try:
            names[sym] = str(_krx.get_market_ticker_name(sym))
        except Exception:
            names[sym] = ""
    return indices, names


def _read_krx_csv(path: Path):
    """KRX CSV를 읽는다. 인코딩(cp949/utf-8)과 종목코드 컬럼을 자동 판별한다."""
    import pandas as pd

    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"{path} 인코딩을 판별할 수 없습니다")

    code_col = None
    for col in df.columns:
        vals = df[col].astype(str).str.strip().str.zfill(6)
        if vals.str.fullmatch(r"\d{6}").mean() > 0.8:
            code_col = col
            break
    if code_col is None:
        raise RuntimeError(f"{path} 에서 6자리 종목코드 컬럼을 찾지 못했습니다: "
                           f"{list(df.columns)}")
    df[code_col] = df[code_col].astype(str).str.strip().str.zfill(6)
    df = df[df[code_col].str.fullmatch(r"\d{6}")].copy()

    def find(*keys) -> Optional[str]:
        return next((c for c in df.columns
                     if any(k in str(c) for k in keys)), None)

    return df, code_col, find("종목명", "한글", "name", "Name"), \
        find("시가총액", "상장시가총액"), find("거래대금")


# 대회 규정상 매매할 수 없는 소속부 표기. KRX 전종목시세의 `소속부` 컬럼으로 걸러진다.
BLOCKED_SECTIONS = ("관리종목", "투자주의환기", "SPAC", "외국기업")
# 대회 규정: KOSPI·KOSDAQ 개별 종목. KONEX는 대상이 아니다.
ALLOWED_MARKETS = ("KOSPI", "KOSDAQ", "KOSDAQ GLOBAL")
MIN_PRICE = 1_000.0        # 현재가 1,000원 미만은 매수 불가(동전주)
MIN_SHARES = 100_000.0     # 발행주식수 10만주 미만은 거래 불가


def fetch_all_stocks(path: str, min_trading_value: float
                     ) -> Tuple[List[str], Dict[str, Any], Dict[str, str]]:
    """KRX 전종목시세 CSV에 **대회 규정 필터**를 적용해 매매 후보를 만든다.

    반환: (통과 종목 목록, 종목별 메타, 제외 사유)

    ## 단위 주의

    전종목시세의 `시가총액`은 **원 단위**이고 `상장주식수`가 직접 들어 있다
    (실측 확인: 시가총액 = 상장주식수 × 종가). 반면 지수 구성종목 CSV의
    `상장시가총액`은 **백만원 단위**다. 두 파일을 같은 코드로 읽으면 1,000,000배
    틀리므로 경로를 나눠 뒀다.

    ## 필터가 규정을 그대로 옮긴 것인지

    - `시장구분` ∈ KOSPI·KOSDAQ·KOSDAQ GLOBAL (KONEX 제외)
    - `소속부`에 관리종목·투자주의환기·SPAC·외국기업 미포함
    - `종가` ≥ 1,000원 (매수 불가 규정)
    - `상장주식수` ≥ 100,000주 (거래 불가 규정)
    - `거래량` > 0 (거래 미발생 종목 제외)
    - `거래대금` ≥ `min_trading_value` — **이건 규정이 아니라 우리 판단**이다.
      1억 계좌에서 단일 50%는 5천만원이고, 거래대금 30억 종목이면 일거래대금의
      1.7%라 슬리피지를 통제할 수 있다. 하한을 낮추면 꼬리는 두꺼워지지만
      슬리피지가 커진다.
    """
    import pandas as pd

    p = Path(path)
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(p, dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"{p} 인코딩을 판별할 수 없습니다")

    if len(df.columns) != 14:
        raise RuntimeError(
            f"{p.name} 은 전종목시세 형식이 아닙니다(컬럼 {len(df.columns)}개, 14개 기대): "
            f"{list(df.columns)}")
    df.columns = ["code", "name", "market", "section", "close", "chg", "pct",
                  "open", "high", "low", "volume", "value", "cap", "shares"]
    for col in ("close", "volume", "value", "cap", "shares"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["section"] = df["section"].fillna("")

    excluded: Dict[str, str] = {}

    def drop(mask, why: str) -> None:
        for code in df.loc[mask, "code"]:
            excluded.setdefault(code, why)

    drop(~df["market"].isin(ALLOWED_MARKETS), "KONEX 등 대상 외 시장")
    drop(df["section"].str.contains("|".join(BLOCKED_SECTIONS)), "관리종목·환기·SPAC·외국기업")
    drop(df["close"] < MIN_PRICE, f"종가 {MIN_PRICE:,.0f}원 미만(매수 불가)")
    drop(df["shares"] < MIN_SHARES, f"발행주식수 {MIN_SHARES:,.0f}주 미만(거래 불가)")
    drop(~(df["volume"] > 0), "당일 거래 미발생")
    drop(df["value"] < min_trading_value, f"거래대금 {min_trading_value / 1e8:.0f}억 미만(유동성)")

    ok = df[~df["code"].isin(excluded)].copy()
    meta: Dict[str, Any] = {
        "names": {r.code: str(r.name).strip() for r in df.itertuples()},
        "market": {r.code: str(r.market) for r in df.itertuples()},
        "section": {r.code: str(r.section) for r in df.itertuples()},
        # 금액은 모두 원 단위로 저장한다(CLAUDE.md 규약).
        "market_cap": {r.code: float(r.cap) for r in df.itertuples()
                       if r.cap == r.cap},
        "shares": {r.code: float(r.shares) for r in df.itertuples()
                   if r.shares == r.shares},
        "trading_value": {r.code: float(r.value) for r in df.itertuples()
                          if r.value == r.value},
    }
    logger.info(f"[Universe] 전종목 {len(df)} → 규정·유동성 통과 {len(ok)} "
                f"(제외 {len(excluded)}, 거래대금 하한 {min_trading_value / 1e8:.0f}억)")
    return sorted(ok["code"]), meta, excluded


def fetch_from_csv(labelled: List[Tuple[str, str]]
                   ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, float]]:
    """지수 구성종목 CSV를 읽는다(최소 자격 요건 판정용).

    `labelled`는 (라벨, 경로) 목록이다. pykrx 1.2.9의 목록성 조회는 KRX 계정
    로그인을 요구하도록 바뀌어 인증 없이는 빈 응답이 오므로, CSV 경로가 사실상
    기본 경로다.

    이 파일의 `상장시가총액`은 **백만원 단위**이므로 원으로 환산해 돌려준다.
    """
    indices: Dict[str, List[str]] = {}
    names: Dict[str, str] = {}
    caps: Dict[str, float] = {}

    for label, path in labelled:
        p = Path(path)
        df, code_col, name_col, cap_col, _ = _read_krx_csv(p)
        codes = df[code_col].tolist()
        indices[label] = sorted(set(codes))
        logger.info(f"[Universe] {label}: {len(indices[label])}종목 ({p.name})")

        if name_col is not None:
            for code, nm in zip(codes, df[name_col].astype(str)):
                names.setdefault(code, nm.strip())
        if cap_col is not None:
            for code, cap in zip(codes, df[cap_col]):
                try:
                    # 지수 CSV의 상장시가총액은 백만원 단위 → 원으로 환산
                    caps.setdefault(code, float(cap) * 1_000_000.0)
                except (TypeError, ValueError):
                    pass
    return indices, names, caps


async def verify_with_toss(symbols: List[str], bars: int = 5
                           ) -> Tuple[List[str], Dict[str, str]]:
    """토스에서 일봉이 실제로 오는 종목만 남긴다."""
    from api.toss_data import TossDataClient

    ok: List[str] = []
    bad: Dict[str, str] = {}
    async with TossDataClient() as cli:
        total = len(symbols)
        for i, sym in enumerate(symbols, 1):
            try:
                rows = await cli.fetch_candles(sym, count=bars)
                if rows is None or len(rows) == 0:
                    bad[sym] = "일봉 0건"
                else:
                    ok.append(sym)
            except Exception as exc:
                bad[sym] = f"{type(exc).__name__}: {str(exc)[:80]}"
            if i % 50 == 0 or i == total:
                logger.info(f"[Universe] 토스 확인 {i}/{total} · 사용가능 {len(ok)} · 제외 {len(bad)}")
    return ok, bad


def build_sectors(indices: Dict[str, List[str]], names: Dict[str, str]) -> Dict[str, str]:
    """섹터 매핑. 기존 `SECTOR_MAP`을 우선하고, 없으면 지수 소속을 대용으로 쓴다.

    350종목의 업종을 KRX에서 일괄로 받는 안정적인 경로가 없어서, 섹터 상한
    (`MAX_SECTOR_WEIGHT`)이 최소한 시장 구분으로는 작동하게 해 둔다. 업종 단위
    분산이 필요해지면 여기를 보강한다.
    """
    from config.settings import SECTOR_MAP

    out: Dict[str, str] = {}
    for label, syms in indices.items():
        for sym in syms:
            out[sym] = SECTOR_MAP.get(sym) or ("KOSDAQ" if label == "KOSDAQ150" else "KOSPI")
    _ = names
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="코스피200·코스닥150 구성종목 갱신")
    ap.add_argument("--date", type=str, default="", help="기준일 YYYYMMDD(기본 최근 영업일)")
    ap.add_argument("--kospi200", type=str, default="", help="코스피200 구성종목 CSV")
    ap.add_argument("--kosdaq150", type=str, default="", help="코스닥150 구성종목 CSV")
    ap.add_argument("--from-csv", nargs="+", default=None,
                    help="파일명으로 지수를 추정해 읽는다(kosdaq 포함 시 코스닥150)")
    ap.add_argument("--all-stocks", type=str, default="",
                    help="KRX 전종목시세 CSV. 대회 규정 필터를 적용해 매매 후보를 만든다. "
                         "지수 CSV를 함께 주면 최소 자격 요건 판정용 멤버십으로 남는다")
    ap.add_argument("--min-trading-value", type=float, default=3e9,
                    help="거래대금 하한(원). 기본 30억 — 1억 계좌의 단일 50%%(5천만원)가 "
                         "일거래대금의 1.7%%가 되는 수준")
    ap.add_argument("--no-verify", action="store_true", help="토스 시세 확인을 생략한다")
    ap.add_argument("--out", type=str, default="config/universe.json")
    args = ap.parse_args()

    s = get_settings()
    setup_logging(s.LOG_DIR, s.LOG_LEVEL)
    date = args.date or _latest_business_day()
    caps: Dict[str, float] = {}

    labelled: List[Tuple[str, str]] = []
    if args.kospi200:
        labelled.append(("KOSPI200", args.kospi200))
    if args.kosdaq150:
        labelled.append(("KOSDAQ150", args.kosdaq150))
    if args.from_csv:
        for path in args.from_csv:
            label = "KOSDAQ150" if "kosdaq" in Path(path).name.lower() else "KOSPI200"
            labelled.append((label, path))

    if labelled:
        indices, names, caps = fetch_from_csv(labelled)
        source = f"KRX CSV ({', '.join(Path(p).name for _, p in labelled)})"
    elif args.all_stocks:
        indices, names = {}, {}
        source = ""
    else:
        indices, names = fetch_from_pykrx(date)
        source = f"pykrx get_index_portfolio_deposit_file (기준일 {date})"

    # 전종목시세가 있으면 그것이 매매 후보를 정하고, 지수 목록은 최소 자격 요건
    # ("코스피200·코스닥150 중 5종목 이상") 판정용으로만 남는다.
    meta: Dict[str, Any] = {}
    rule_excluded: Dict[str, str] = {}
    if args.all_stocks:
        candidates, meta, rule_excluded = fetch_all_stocks(
            args.all_stocks, args.min_trading_value)
        names = {**meta.get("names", {}), **names}
        caps = {**meta.get("market_cap", {}), **caps}
        source = (f"KRX 전종목시세 {Path(args.all_stocks).name} "
                  f"(거래대금 {args.min_trading_value / 1e8:.0f}억 하한)"
                  + (f" + {source}" if source else ""))
        all_syms = candidates
    else:
        all_syms = sorted({s_ for lst in indices.values() for s_ in lst})
    logger.info(f"[Universe] 매매 후보 {len(all_syms)}종목")

    excluded: Dict[str, str] = {}
    if args.no_verify or not s.toss_ready():
        tradable = list(all_syms)
        if not s.toss_ready():
            logger.warning("[Universe] 토스 자격증명이 없어 시세 확인을 건너뜁니다")
    else:
        tradable, excluded = asyncio.run(verify_with_toss(all_syms))

    sectors = build_sectors(indices, names)
    # 전종목시세의 시장구분을 섹터 대용으로 보강한다(섹터 상한이 최소한 시장 단위로 작동).
    for sym, mkt in meta.get("market", {}).items():
        sectors.setdefault(sym, "KOSDAQ" if "KOSDAQ" in mkt else "KOSPI")

    payload: Dict[str, Any] = {
        "as_of": dt.date.today().isoformat(),
        "reference_date": date,
        "source": source,
        # 최소 자격 요건 판정용 지수 멤버십 (매매 제한이 아니다)
        "indices": {k: v for k, v in indices.items()},
        "tradable": sorted(tradable),
        "excluded": {**rule_excluded, **excluded},
        "names": {k: v for k, v in sorted(names.items()) if v},
        "sectors": {k: sectors[k] for k in sorted(sectors)},
        # 금액은 모두 원 단위 (CLAUDE.md 규약)
        "market_cap": {k: caps[k] for k in sorted(caps)},
        "shares": meta.get("shares", {}),
        "trading_value": meta.get("trading_value", {}),
        "market": meta.get("market", {}),
        "section": meta.get("section", {}),
        "filters": {
            "min_price": MIN_PRICE,
            "min_shares": MIN_SHARES,
            "min_trading_value": args.min_trading_value,
            "allowed_markets": list(ALLOWED_MARKETS),
            "blocked_sections": list(BLOCKED_SECTIONS),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f" 유니버스 갱신 완료 — {out}")
    print("=" * 78)
    for label, lst in indices.items():
        print(f"  {label:<14} {len(lst):>4}종목  (최소 자격 요건 판정용)")
    print(f"  {'매매 후보':<14} {len(all_syms):>4}종목")
    print(f"  {'사용가능':<14} {len(tradable):>4}종목  (토스 일봉 확인됨)")
    if rule_excluded:
        print()
        print(f"  규정·유동성 제외 {len(rule_excluded)}종목 — 사유별:")
        by_reason: Dict[str, int] = {}
        for why in rule_excluded.values():
            by_reason[why] = by_reason.get(why, 0) + 1
        for why, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"      {n:>5}  {why}")
    if excluded:
        print()
        print(f"  토스 시세 없음 {len(excluded)}종목")
        for sym, why in list(excluded.items())[:5]:
            print(f"      {sym} {names.get(sym, ''):<14} {why}")
    print("=" * 78)
    print(f"  적용: .env 에 UNIVERSE_FILE={out} 를 넣으십시오")
    print(f"  수집: python scripts/collect_data.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
