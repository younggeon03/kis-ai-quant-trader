"""대회 직전 점검 — 10/12 아침에 한 줄로 확인한다.

## 왜 스크립트인가

대회 설정이 **비어 있어도 시스템은 조용히 돈다.** 실제로 그랬다 —
`COMP_START`/`COMP_END`/`PHASE1_END`가 비어 있으면 `current_phase()`가 OFF를
돌려주고 국면 전환·초기화 권고·최소 요건 추적이 전부 작동하지 않는데, 로그에는
아무 경고도 안 나왔다. `MIN_TRADE_*`가 0이면 요건이 "없음"으로 취급된다.

그리고 **꺼진 줄 알았던 게이트가 살아 있었다.** `DAILY_LOSS_LIMIT=0.03`이
보유 평균 −3.6%에서 전량 청산을 걸고, `TAKE_PROFIT_PCT=0.09`가 상승을 9%에서
잘랐다. 둘 다 리플레이에 리스크 매니저가 없어 측정에 나타나지 않았다.

대회 첫날 아침에 사람이 `.env`를 눈으로 훑어 이런 것을 잡기는 어렵다. 그래서
**기대값과 실제값을 코드로 비교**한다.

```bash
python scripts/preflight.py            # 점검만
python scripts/preflight.py --live     # 실서버 전환 여부까지 요구한다
```

종료코드: 0 통과 · 1 경고(진행 가능) · 2 치명(진행하면 안 된다)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402

FATAL, WARN, OK = "FATAL", "WARN", "OK"


def check(settings, live: bool) -> List[Tuple[str, str, str, str]]:
    """(수준, 항목, 상태, 설명). 수준은 FATAL/WARN/OK."""
    s = settings
    out: List[Tuple[str, str, str, str]] = []

    def add(level_if_bad: str, name: str, ok: bool, detail: str, why: str = "") -> None:
        out.append((OK if ok else level_if_bad, name, detail, why))

    # --- 대회 식별 ---
    dates_ok = bool(s.COMP_START and s.COMP_END and s.PHASE1_END)
    add(FATAL, "대회 날짜", dates_ok,
        f"{s.COMP_START or '(비어 있음)'} ~ {s.COMP_END or '(비어 있음)'} "
        f"/ 1단계 종료 {s.PHASE1_END or '(비어 있음)'}",
        "비면 국면 전환·초기화 권고·요건 추적이 전부 작동하지 않는다")
    req_ok = (s.MIN_TRADE_COUNT > 0 and s.MIN_TRADE_AMOUNT > 0
              and s.MIN_TRADE_DAYS > 0 and s.MIN_TRADE_SYMBOLS > 0)
    add(FATAL, "최소 자격 요건", req_ok,
        f"{s.MIN_TRADE_COUNT}회 / {s.MIN_TRADE_AMOUNT / 1e8:.0f}억 / "
        f"{s.MIN_TRADE_DAYS}일 / {s.MIN_TRADE_SYMBOLS}종목",
        "0이면 '요건 없음'으로 취급돼 추적이 무의미하다")

    # --- 계좌 연결 ---
    if live:
        add(FATAL, "실서버 모드", not s.USE_MOCK_MODE,
            "PROD" if not s.USE_MOCK_MODE else "MOCK(가상체결)",
            "USE_MOCK_MODE=false 로 바꿔야 실제 주문이 나간다")
        add(FATAL, "KIS 자격증명", bool(s.KIS_APP_KEY and s.KIS_APP_SECRET),
            f"app_key {len(s.KIS_APP_KEY or '')}자 / secret {len(s.KIS_APP_SECRET or '')}자",
            "대회 계좌 발급 후 .env 에 직접 입력한다")
        acct = str(s.KIS_ACCOUNT_NO or "")
        add(FATAL, "계좌번호", bool(acct) and acct not in ("00000000-01", "12345678-01"),
            acct or "(비어 있음)", "예시값이 그대로면 주문이 거부된다")
    else:
        add(WARN, "실서버 모드", not s.USE_MOCK_MODE,
            "PROD" if not s.USE_MOCK_MODE else "MOCK — 10/12 계좌 발급 후 전환",
            "--live 로 점검하면 FATAL 로 판정한다")

    # --- 전략 ---
    add(WARN, "장중 전략", s.USE_INTRADAY,
        f"트리거 +{s.INTRADAY_TRIGGER_PCT:.0%} · 거래량 x{s.INTRADAY_VOL_MULT} · "
        f"스캔 {s.INTRADAY_SCAN_HHMM} · 마감 {s.INTRADAY_FLATTEN_HHMM}",
        "꺼져 있으면 일봉 스윙으로 돈다")
    # **사이징 검사는 전략별로 갈린다.** 분봉은 고정 2종목 x 42%, 스윙은 8슬롯
    # 변동성 타게팅으로 측정했다. 전략과 무관하게 한쪽을 요구하면 다른 쪽이
    # 통과할 수 없고, 실제로 스윙 분기와 모순됐다.
    gross = s.MAX_POSITIONS * s.FIXED_POSITION_WEIGHT
    if s.USE_INTRADAY:
        add(FATAL, "고정 사이징", s.SIZING_MODE == "fixed",
            f"{s.SIZING_MODE} · {s.MAX_POSITIONS}종목 x {s.FIXED_POSITION_WEIGHT:.0%}",
            "vol_target 이면 종목당 3%로 들어가 익스포저가 6%대가 된다")
        add(FATAL, "익스포저 여유", gross <= s.MAX_GROSS_EXPOSURE - 0.02,
            f"{s.MAX_POSITIONS} x {s.FIXED_POSITION_WEIGHT:.0%} = {gross:.0%} "
            f"(상한 {s.MAX_GROSS_EXPOSURE:.0%})",
            "여유가 없으면 부분 체결 중 두 번째 진입이 '주문가능금액 부족'으로 거부된다")
        add(FATAL, "단일 종목 규정", s.FIXED_POSITION_WEIGHT <= 0.50,
            f"{s.FIXED_POSITION_WEIGHT:.0%} (규정 50%)",
            "슬리피지로 넘치면 규정 위반이다")
    else:
        add(FATAL, "단일 종목 규정", s.MAX_POSITION_WEIGHT <= 0.50,
            f"{s.MAX_POSITION_WEIGHT:.0%} (규정 50%)",
            "슬리피지로 넘치면 규정 위반이다")

    # --- 리스크 해제 (장중 전략일 때만) ---
    if s.USE_INTRADAY:
        for name, value, limit, why in (
            ("익절", s.TAKE_PROFIT_PCT, 0.0, "9%면 상승을 거기서 자른다"),
            ("트레일링", s.TRAILING_ACTIVATE_PCT, 0.0, "2.5%면 되돌림에 바로 잘린다"),
        ):
            add(FATAL, f"{name} 해제", value == limit, f"{value}",
                why + " — 꼬리가 구조적으로 사라진다")
        for name, value, why in (
            ("MDD 소프트", s.MDD_SOFT_LIMIT, "신규 진입 차단"),
            ("MDD 하드", s.MDD_HARD_LIMIT, "전량 청산 + 쿨다운"),
            ("일일 손실", s.DAILY_LOSS_LIMIT, "HALT + 전량 청산 + 쿨다운"),
        ):
            add(FATAL, f"{name} 해제", value >= 1.0, f"{value}",
                f"{why} — 익스포저 {gross:.0%}에서 쉽게 닿는다")
        add(WARN, "신호 반전 해제", s.SIGNAL_SELL_THRESHOLD <= -1.0,
            f"{s.SIGNAL_SELL_THRESHOLD}",
            "일봉 점수로 장중 포지션을 청산한다(개념 불일치)")
        add(WARN, "급락 완화", s.CRASH_DROP_PCT >= 0.20,
            f"{s.CRASH_DROP_PCT:.0%}",
            "4.5%는 급등주의 정상 변동폭이다")
        add(WARN, "손절 유지", s.STOP_LOSS_PCT > 0,
            f"{s.STOP_LOSS_PCT:.0%}",
            "단일 종목 붕괴를 막는 마지막 선이다")
    else:
        # --- 일봉 스윙으로 돌아왔을 때 ---
        #
        # **이 분기가 없으면 스윙 복귀가 조용히 잘못된 설정으로 돈다.** 분봉용으로
        # 풀어 둔 게이트(익절 0 · 트레일링 0 · MDD 1.0 · 일일손실 1.0)는 스윙에서
        # 검증된 설정이 **아니다.** 절제 실험(21구간 x 30봉, 40종목 x 1,138봉):
        #
        # | 설정 | 9/28(구) | **9/30 v2(최신)** |
        # |---|---|---|
        # | 원 설정(손절 3.5% · 익절 9%) | −0.348 / −9.2% | −0.147 / −5.2% |
        # | **손절 12%만** | 0.544 / +15.0% | **+0.425 / +12.1%** |
        # | 익절 제거만 | 0.523 / +15.5% | **−0.163 / −6.9%** |
        # | 손절12 + 익절 제거 | 0.873 / +33.2% | **+0.263 / +7.0%** |
        # | 트레일링 제거만 | −0.255 / −7.2% | −0.089 / −3.7% |
        #
        # **두 실행이 갈렸고 원인은 확정되지 않았다.** 동의하는 것만 FATAL로 둔다 —
        # 손절 12%와 트레일링 유지. 익절은 갈리므로 WARN이고, 최신(v2)을 따라
        # **유지**를 권한다. 구버전만 보고 "익절 제거"를 FATAL로 걸어 둔 적이 있다.
        add(FATAL, "손절 12%(스윙)", s.STOP_LOSS_PCT >= 0.10,
            f"{s.STOP_LOSS_PCT:.0%}",
            "손절 3.5%는 유니버스 일간 변동성 3.3%와 거의 같아 무작위 진입에서도 "
            "10봉 내 62.6%가 노이즈로 손절됐다 — 두 실행이 모두 동의하는 최대 레버다")
        add(FATAL, "트레일링 유지(스윙)", s.TRAILING_ACTIVATE_PCT > 0,
            f"{s.TRAILING_ACTIVATE_PCT}",
            "트레일링 제거는 두 실행 모두 악화였다(−0.255 / −0.089). 분봉용으로 꺼 둔 "
            "채 스윙으로 내리면 그 설정이 된다")
        add(WARN, "익절 유지(스윙·근거 갈림)", s.TAKE_PROFIT_PCT > 0,
            f"{s.TAKE_PROFIT_PCT}",
            "익절 제거는 구버전에서 +0.349 기여, 최신 v2에서 −0.162 악화로 갈렸다. "
            "최신을 따라 9% 유지를 권한다")
        add(FATAL, "변동성 사이징(스윙)", s.SIZING_MODE == "vol_target",
            f"{s.SIZING_MODE}",
            "fixed 2종목 x 42%는 분봉 전용이다. 스윙은 8슬롯 변동성 타게팅으로 측정했다")
        add(WARN, "MDD 게이트(스윙)", s.MDD_SOFT_LIMIT < 1.0,
            f"소프트 {s.MDD_SOFT_LIMIT} · 하드 {s.MDD_HARD_LIMIT}",
            "30봉 구간에서는 발동하지 않는 것으로 측정됐지만, 분봉용 1.0을 그대로 두면 "
            "익스포저가 커질 때 보호가 없다")
        add(WARN, "일일 손실(스윙)", s.DAILY_LOSS_LIMIT < 1.0,
            f"{s.DAILY_LOSS_LIMIT}",
            "분봉용으로 풀어 둔 값이다")
        add(WARN, "신호 반전(스윙)", s.SIGNAL_SELL_THRESHOLD > -1.0,
            f"{s.SIGNAL_SELL_THRESHOLD}",
            "스윙에서는 일봉 점수 반전이 정상 청산 사유다 — 분봉용 −9.9는 그것을 끈다")
        add(WARN, "유니버스 단서(스윙)", len(s.UNIVERSE) <= 60,
            f"{len(s.UNIVERSE)}종목",
            "검증 수치(최신 v2: 손절12만 +0.425/+12.1%)는 **40종목**에서 측정했다"
            "(ablate_exits_v2.json config.symbols=40). 284종목 스윙은 미측정이고, "
            "동일가중 매수보유 벤치마크가 +61.8%로 모든 전략 설정을 이긴다")

    # --- 주문 ---
    add(FATAL, "지정가 비대칭", s.BUY_LIMIT_BAND > s.SELL_LIMIT_BAND,
        f"매수 +{s.BUY_LIMIT_BAND:.1%} / 매도 −{s.SELL_LIMIT_BAND:.1%}",
        "매수 0.4%로는 체결률이 61.5%뿐이고 못 사는 쪽이 가장 좋은 집단이다")

    # --- 유니버스 ---
    n = len(s.UNIVERSE)
    add(FATAL, "유니버스", n >= 50, f"{n}종목 (파일 {s.UNIVERSE_FILE or '없음'})",
        "비어 있거나 기본 40종목이면 확장 효과가 없다")
    # **종목코드가 숫자 6자리인지 본다.** 유니버스에 `0035S0`·`0010F0` 같은 코드가
    # 8개 섞여 있다(전부 실제 코스닥 기업이고 최근 상장이라 임시 코드로 보인다).
    # KIS 주문은 **단축코드 6자리**를 요구하므로 이 코드로는 주문이 거부될 수 있고,
    # 백테스트에서는 이력 하한 필터가 조용히 걸러 내므로 측정에 나타나지 않는다.
    odd = [c for c in s.UNIVERSE if not re.fullmatch(r"\d{6}", str(c))]
    add(WARN, "종목코드 형식", not odd,
        f"비표준 {len(odd)}개" + (f" ({', '.join(odd[:5])}…)" if odd else ""),
        "KIS 주문은 단축코드 6자리를 요구한다 — 실거래에서 거부될 수 있다. "
        "유니버스 재생성 시 제외하거나 주문 전 개별 확인이 필요하다")

    blob = s._universe_blob() or {}
    as_of = str(blob.get("as_of") or "")
    fresh = bool(as_of) and (not s.COMP_START or as_of >= "2026-10-01")
    add(WARN, "유니버스 신선도", fresh, f"as_of {as_of or '(없음)'}",
        "거래대금 하한은 '그날 돌던 종목을 남기는 필터'다 — 대회 직전에 재생성한다")
    return out


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser(description="대회 직전 점검")
    ap.add_argument("--live", action="store_true",
                    help="실서버 전환·KIS 자격증명까지 요구한다(대회 당일)")
    args = ap.parse_args()

    s = get_settings()
    rows = check(s, args.live)
    width = max(len(r[1]) for r in rows)

    print()
    print("=" * 96)
    print(f" 대회 직전 점검 {'(실서버 모드)' if args.live else '(준비 단계)'}")
    print("=" * 96)
    fatal = warn = 0
    for level, name, detail, why in rows:
        mark = {OK: " OK ", WARN: "WARN", FATAL: "FAIL"}[level]
        print(f" [{mark}] {name:<{width}}  {detail}")
        if level != OK:
            print(f"        {'':<{width}}  → {why}")
            if level == FATAL:
                fatal += 1
            else:
                warn += 1
    print("=" * 96)
    if fatal:
        print(f" 치명 {fatal}건 · 경고 {warn}건 — **이 상태로 대회에 들어가면 안 된다**")
    elif warn:
        print(f" 경고 {warn}건 — 확인한 뒤 진행한다")
    else:
        print(" 전부 통과")
    print("=" * 96)
    return 2 if fatal else (1 if warn else 0)


if __name__ == "__main__":
    raise SystemExit(main())
