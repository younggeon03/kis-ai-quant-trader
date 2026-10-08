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
    #
    # **세 가지 모드가 있고 요구 설정이 서로 모순된다.** 전략과 무관하게 한쪽을
    # 요구하면 다른 쪽이 통과할 수 없다(실제로 catalyst가 스윙 분기로 떨어져
    # "트레일링 유지"와 "vol_target"을 치명으로 요구받았다 — 둘 다 반대다).
    #
    # | 모드 | 진입 | 사이징 | 트레일링 | 청산 경로 | P(+100%) |
    # |---|---|---|---|---|---|
    # | intraday | 장중 돌파 | fixed 2 x 42% | 제거 | 당일 15:15 | **0.00%** |
    # | catalyst | 일봉 폭증(T+1 시가) | fixed 2 x 45% | 유지 | 일봉 6봉 | 1.40~1.67% |
    # | **burst** | **장중 폭증(당일)** | fixed 2 x 45% | 유지 | 일봉 6봉 | **2.09%** |
    # | swing | 일봉 모델 점수 | vol_target 8슬롯 | **유지** | 30봉 | 0.96% |
    #
    # `burst`는 `USE_INTRADAY=true` + `INTRADAY_FLATTEN=false`다. 장중에 진입하고
    # **청산은 일봉 경로**(손절 + 감쇠 3종 + 시간손절)가 맡는다. 그래서 출구 요구
    # 사항이 `intraday`가 아니라 `catalyst`와 같다 — 분류를 틀리면 "트레일링 해제"를
    # 치명으로 요구받는데 그것은 감쇠 청산과 정반대다.
    if s.USE_INTRADAY and s.INTRADAY_FLATTEN:
        mode = "intraday"
    elif s.USE_INTRADAY:
        mode = "burst"
    elif s.ENTRY_MODE == "catalyst":
        mode = "catalyst"
    else:
        mode = "swing"
    daily_exit = mode in ("catalyst", "burst")
    add(WARN, "전략 모드", mode == "burst",
        f"{mode} (USE_INTRADAY={s.USE_INTRADAY} · FLATTEN={s.INTRADAY_FLATTEN} · "
        f"ENTRY_MODE={s.ENTRY_MODE})",
        "측정 최선은 burst 2.09%다 (catalyst 1.40% · 스윙 0.96% · 분봉 단타 0.00%). "
        "대응 비교에서 장중 진입이 T+1보다 P(+100%) +0.74%p, McNemar p=0.039")
    intra_cond = (f"거래대금 x{s.INTRADAY_BURST_MULT:g}(장중 누적)"
                  if s.INTRADAY_ENTRY_MODE == "burst"
                  else f"트리거 +{s.INTRADAY_TRIGGER_PCT:.0%} · 거래량 x{s.INTRADAY_VOL_MULT}")
    add(WARN, "장중 전략", s.USE_INTRADAY,
        f"{s.INTRADAY_ENTRY_MODE} · {intra_cond} · 스캔 {s.INTRADAY_SCAN_HHMM} · "
        f"마감 {s.INTRADAY_FLATTEN_HHMM if s.INTRADAY_FLATTEN else 'OFF(다일 보유)'}",
        "꺼져 있으면 일봉 스윙 또는 catalyst 로 돈다")
    if s.USE_INTRADAY and not s.INTRADAY_FLATTEN:
        # **오버나이트 0건은 지금까지 설계 불변 조건이었다.** 끄면 청산이 전부
        # 일봉 경로(risk_manager)로 가므로, 거기에 시간손절과 감쇠 규칙이 실제로
        # 켜져 있는지 확인해야 한다. 아니면 포지션이 무기한 남는다.
        add(FATAL, "다일 보유 시 청산 경로", s.MAX_HOLDING_BARS > 0,
            f"시간손절 {s.MAX_HOLDING_BARS}봉 · 감쇠 [{s.EXIT_RULES or '없음'}]",
            "마감 정리를 끄면 청산이 전부 일봉 경로로 간다 — 시간손절이 0이면 "
            "포지션이 무기한 남는다")
    # **사이징 검사는 전략별로 갈린다.** 분봉은 고정 2종목 x 42%, 스윙은 8슬롯
    # 변동성 타게팅으로 측정했다. 전략과 무관하게 한쪽을 요구하면 다른 쪽이
    # 통과할 수 없고, 실제로 스윙 분기와 모순됐다.
    gross = s.MAX_POSITIONS * s.FIXED_POSITION_WEIGHT
    if mode in ("intraday", "catalyst", "burst"):
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

    # --- 리스크 해제 (intraday · catalyst 공통) ---
    # 셋 다 꼬리를 사는 전략이라 상승을 자르는 규칙을 전부 끈다.
    if mode in ("intraday", "catalyst", "burst"):
        add(FATAL, "익절 해제", s.TAKE_PROFIT_PCT == 0.0, f"{s.TAKE_PROFIT_PCT}",
            "9%면 상승을 거기서 자른다 — 꼬리가 구조적으로 사라진다")
        # **트레일링은 모드마다 반대다.** 분봉은 당일 청산이라 되돌림 청산이 겹쳐
        # 2.5%에 바로 잘리므로 꺼야 한다. catalyst는 6봉 보유라 "상승이 식으면
        # 청산"이 설계의 일부이고, 절제 실험에서 트레일링 제거가 두 실행 모두
        # 악화였다(−0.255 / −0.089). 구현은 `strategy/exit_rules.py`다.
        if mode == "intraday":
            add(FATAL, "트레일링 해제", s.TRAILING_ACTIVATE_PCT == 0.0,
                f"{s.TRAILING_ACTIVATE_PCT}",
                "당일 청산과 겹쳐 되돌림에 바로 잘린다 — 꼬리가 사라진다")
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
        if mode == "intraday":
            add(WARN, "급락 완화", s.CRASH_DROP_PCT >= 0.20,
                f"{s.CRASH_DROP_PCT:.0%}",
                "4.5%는 급등주의 정상 변동폭이다")
        add(WARN, "손절 유지", s.STOP_LOSS_PCT > 0,
            f"{s.STOP_LOSS_PCT:.0%}",
            "단일 종목 붕괴를 막는 마지막 선이다")

    if daily_exit:
        # --- 꼬리 전수 탐색의 승자 구조가 실제로 그 값인지 확인한다 ---
        #
        # 지문 22d557320d0e640d · 284종목 · 조건 6종 x 종목수 4 x 회전 5 · 시행 8만.
        # 승자는 **거래대금 10배 x 2종목 x 45% x 6봉 보유**로 P(+100%) 1.67%다
        # (2위 10봉 1.57% · 스윙 0.96% · 분봉 단타 0.00%).
        #
        # 값이 하나만 어긋나도 측정한 구조가 아니게 되는데, 로그에는 아무 경고도
        # 나오지 않는다. 그래서 기대값과 실제값을 코드로 비교한다.
        add(FATAL, "catalyst 폭증 하한", abs(s.CATALYST_BURST_MULT - 10.0) < 1e-9,
            f"x{s.CATALYST_BURST_MULT:g} (측정 최적 x10)",
            "8/12/20/30배를 분봉에서 쓸어 본 결과 하한은 효과가 단조롭지 않다 — "
            "일봉 폭증도 측정값에서 벗어나면 다른 구조다")
        # **주 청산이 `vol_alive`면 보유 상한은 안전장치다.** 고정 기간이 주
        # 청산일 때는 6봉이 최적이었지만(30봉은 1라운드가 되어 0.90%로 떨어진다),
        # 상태 기반에서는 6봉이 상태를 무력화한다 — 살아 있어도 6일에 잘린다.
        # 그래서 두 경우를 갈라서 본다.
        state_exit = "vol_alive" in str(s.EXIT_RULES or "")
        if state_exit:
            add(FATAL, "보유 상한", s.MAX_HOLDING_BARS >= 20,
                f"{s.MAX_HOLDING_BARS}봉 (안전 상한 — 주 청산은 vol_alive)",
                "6봉으로 두면 상태 기반이 무력화된다(살아 있어도 6일에 잘린다). "
                "측정은 상한 30봉에서 했고 실제 중앙 보유일수는 6일이다")
        else:
            add(FATAL, "보유 봉 수", s.MAX_HOLDING_BARS in (6, 10),
                f"{s.MAX_HOLDING_BARS}봉 (고정 기간 최적 6봉 · 2위 10봉)",
                "30봉이면 1라운드가 되어 P(+100%)가 0.90%로 떨어진다. "
                "회전은 중간이 최적이다")
        add(FATAL, "동시 보유 종목 수", s.MAX_POSITIONS == 2,
            f"{s.MAX_POSITIONS}종목 (측정 최적 2종목)",
            "1종목은 50%만 투자되고 40%가 현금으로 남아 0.93%, 3종목은 분산이 "
            "꼬리를 깎아 0.89%다")
        add(WARN, "진입 잠금", s.MIN_HOLDING_BARS == 0,
            f"{s.MIN_HOLDING_BARS}봉",
            "잠금이 있으면 시간손절과 겹쳐 회전이 의도보다 느려진다")

        # --- 감쇠 청산은 **꺼야 한다** (측정 결과) ---
        #
        # 거래량 감소·상승 감쇠·급격한 하락을 셋 다 넣고 쟀더니 전부 꼬리를 깎았다
        # (4,082건, 보유 중 실제 규칙 적용). 임계값 문제가 아니라 방향 문제다 —
        # 느슨하게 하면 "거의 안 걸리는 상태"로 수렴하고 조이면 단조적으로 나빠진다.
        #
        # | 설정 | 발동률 | P(+100%) |
        # |---|---|---|
        # | **청산 없음(6봉 시간손절만)** | — | **2.09%** |
        # | 손절 25% | 2.8% | 2.07% |
        # | 손절 12% | 21.5% | 1.89% |
        # | 거래량 감소 30% | 80.9% | 1.01% |
        #
        # 2종목 x 45%로 2배를 만들려면 평균 +111%가 필요해 소수의 2배 사건에
        # 전부 기댄다. 어떤 청산이든 그것을 자를 확률이 있다.
        rules = [r.strip() for r in str(s.EXIT_RULES or "").split(",") if r.strip()]
        bad = [r for r in rules if r != "vol_alive"]
        # **보유를 기간이 아니라 상태가 정한다.** `vol_alive`만 켠다 — 나머지
        # 셋(급락·상승감쇠·거래량 고점대비)은 전부 꼬리를 깎았다(2.09% → 1.01%).
        add(FATAL, "청산 규칙", rules == ["vol_alive"],
            ", ".join(rules) or "없음",
            "vol_alive 만 켠다. crash/trailing/vol_decay 는 전부 꼬리를 깎았고, "
            "vol_decay 는 분모가 보유 중 고점이라 발동률 80.9%다"
            + (f" / 잘못된 규칙: {bad}" if bad else ""))
        add(FATAL, "거래 생존 기준", abs(s.EXIT_VOL_ALIVE_MULT - 2.0) < 1e-9,
            f"20봉중앙 x{s.EXIT_VOL_ALIVE_MULT:g}",
            "측정 최적 x2. 올리면 포트 수치는 오르지만 **단일 2배 비율은 떨어지고** "
            "(0.94%→0.69%) 상승분은 라운드 환산 artifact다. x2가 라운드 5회로 "
            "검증된 기준선과 같아 왜곡이 없는 유일한 지점이다")

        add(FATAL, "손절 폭", s.STOP_LOSS_PCT >= 0.20,
            f"{s.STOP_LOSS_PCT:.0%}",
            "12%는 발동률 21.5%로 회복할 종목을 잘라 꼬리를 1.89%로 깎는다. "
            "이 사건 집단은 변동성이 커서 12%가 정상 변동폭 안이다 — 25% 권장")
        add(WARN, "손절 유지(붕괴 보험)", 0 < s.STOP_LOSS_PCT < 1.0,
            f"{s.STOP_LOSS_PCT:.0%}",
            "완전히 끄면 2종목 x 45%에서 한 종목 반토막이 계좌 −22.5%이고 "
            "거래정지는 6봉까지 들고 있게 된다. 25%는 꼬리를 0.02%p만 깎는다")

    if mode == "burst":
        # --- 장중 진입이 측정된 설정인지 ---
        #
        # 지문 8e0104239f174a6e · 2023-01-01~ · 258종목 · 커버리지 94.8% ·
        # 사건 4,082건. 배수 스윕에서 10/20/30배는 **통계적으로 구분되지 않았고**
        # (z≈1.2) 10배가 커버리지·표본이 가장 크다. 보유는 6~10봉이 평평하다
        # (3봉 1.58% · 6봉 2.09% · 10봉 2.19% · 15봉 1.70%).
        add(FATAL, "장중 진입 모드", s.INTRADAY_ENTRY_MODE == "burst",
            f"{s.INTRADAY_ENTRY_MODE}",
            "breakout(가격 +3% 돌파)은 P(+100%) 0.00%로 측정됐다")
        add(WARN, "장중 폭증 배수", abs(s.INTRADAY_BURST_MULT - 10.0) < 1e-9,
            f"x{s.INTRADAY_BURST_MULT:g}",
            "10/20/30배는 구분되지 않는다 — 10배가 커버리지 94.8%로 가장 신뢰할 만하다")
        add(WARN, "일봉 폭증 배수 일치", abs(s.INTRADAY_BURST_MULT
                                      - s.CATALYST_BURST_MULT) < 1e-9,
            f"장중 x{s.INTRADAY_BURST_MULT:g} / 일봉 x{s.CATALYST_BURST_MULT:g}",
            "두 경로가 다른 집단을 보면 측정과 실거래가 갈린다")
    if mode == "swing":
        # --- 일봉 스윙(모델 진입)으로 돌아왔을 때 ---
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
