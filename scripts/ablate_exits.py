"""출구 규칙 절제 실험 — 구간당 학습 1회로 여러 설정을 동시에 워크포워드 평가한다.

## 왜 별도 스크립트인가

`scripts/walkforward.py`를 설정마다 돌리면 **모델 학습이 설정 수만큼 반복된다.**
출구 규칙(손절·익절·트레일링·보유기간)은 학습에 전혀 쓰이지 않으므로 순수한 낭비다.
실측으로 구간당 학습이 전체 시간의 95%를 차지했다.

여기서는 구간마다

1. 한 번만 학습한다 (`train_on_bars`)
2. 피처·신호를 한 번만 계산한다 (`precompute` 결과를 설정 간 공유)
3. 그 위에서 설정별 백테스트만 반복한다 (빠름)

## 왜 한 번에 하나씩 바꾸는가

여러 변경을 묶어 넣으면 결과가 좋아져도 나빠져도 **무엇이 기여했는지 알 수 없다.**
그 상태에서 파라미터를 맞추면 그것이 곧 과적합이다. 그래서 각 설정은 기준선에서
한 항목씩만 다르게 두고, 누적 효과는 마지막 설정으로 따로 본다.

사용:

    python scripts/ablate_exits.py --source data/cache --symbols 40
    python scripts/ablate_exits.py --source data/cache --max-segments 3   # 빠른 확인
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()

from backtest.data import generate_synthetic_panel, load_panel_from_dir  # noqa: E402
from backtest.engine import BacktestEngine, align_panel  # noqa: E402
from backtest.validation import max_drawdown_from_returns, sharpe_ratio  # noqa: E402
from backtest.walkforward import MIN_WARMUP, slice_panel, train_on_bars  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.ai_model import EnsembleAlphaStrategy  # noqa: E402
from strategy.risk_manager import RiskManager  # noqa: E402
from utils.logger import logger, setup_logging  # noqa: E402

# 설정 delta에 넣으면 그 설정만 수급을 쓰게 하는 특수 키. 설정 필드가 아니라
# "어떤 모델·피처 캐시를 쓸지"를 고르는 스위치이므로 `variant_settings`에서 분리한다.
USE_FLOW_KEY = "_use_flow"

BASELINE = "원 설정(손절 3.5%/익절 9%)"
FULL = "손절12+익절+트레일 제거+보유10봉"

# ---------------------------------------------------------------- 출구 설정군
#
# **기준점을 원 설정으로 고정한다.** 손절 12% + 익절 제거가 기본값으로 채택된 뒤에는
# `{"delta": {}}`가 곧 채택된 설정이 되어, 변형 대부분이 기준선과 같아져 중복이 된다.
# 그래서 원 설정(3.5%/9%)을 명시 변형으로 두고 거기서부터의 차이를 본다.
#
# 이렇게 해야 기본값을 바꾼 뒤에도 "출구 규칙이 실제로 얼마나 기여했는가"를
# 같은 데이터로 다시 확인할 수 있다.
_ORIG = {"STOP_LOSS_PCT": 0.035, "TAKE_PROFIT_PCT": 0.09}

CONFIGS: List[Dict[str, Any]] = [
    {"name": BASELINE, "delta": dict(_ORIG)},
    {"name": "손절 12%만", "delta": {**_ORIG, "STOP_LOSS_PCT": 0.12}},
    {"name": "익절 제거만", "delta": {**_ORIG, "TAKE_PROFIT_PCT": 0.0}},
    {"name": "트레일링 제거만", "delta": {**_ORIG, "TRAILING_ACTIVATE_PCT": 0.0}},
    {"name": "보유 10봉만", "delta": {**_ORIG, "MAX_HOLDING_BARS": 10}},
    {"name": "손절12 + 익절제거 (현재 기본값)",
     "delta": {"STOP_LOSS_PCT": 0.12, "TAKE_PROFIT_PCT": 0.0}},
    {"name": "손절12 + 익절 + 트레일 제거",
     "delta": {"STOP_LOSS_PCT": 0.12, "TAKE_PROFIT_PCT": 0.0,
               "TRAILING_ACTIVATE_PCT": 0.0}},
    {"name": FULL,
     "delta": {"STOP_LOSS_PCT": 0.12, "TAKE_PROFIT_PCT": 0.0,
               "TRAILING_ACTIVATE_PCT": 0.0, "MAX_HOLDING_BARS": 10,
               "CRASH_DROP_PCT": 0.12}},
]

# ---------------------------------------------------------------- 구조 설정군
#
# 손절 12% + 익절 제거가 기본값으로 채택됐으므로 여기서의 기준선은
# "출구 수정이 적용된 현재 상태"다. **절대 수치는 데이터 지문마다 다르므로
# 같은 실행 안에서만 비교한다**(`panel_fingerprint` 참고).
#
# 출구 규칙과 달리 아래 항목들은 **서로 독립이 아니다.** 합성 데이터로 확인했다:
# 사이징만 키우면 MDD 소프트 리밋 7%에 즉시 닿고 회복선 4.2% 위에 수백 봉 머물러
# DERISK에 갇혀 익스포저가 오히려 11.6%로 떨어진다. MDD 한도를 15%/25%로 함께
# 올리면 62.8%가 된다. 그래서 단독 효과와 누적 조합을 같이 본다.
#
# `CRASH_DROP_PCT`를 따로 두는 이유: 기본값 4.5%는 손절 12%보다 먼저 걸리므로
# 사실상 4.5% 손절처럼 작동할 수 있다. 12%로 넓히는 것이 도움이 되는지는 미검증이다.
CRASH = {"CRASH_DROP_PCT": 0.12}
TOPK = {"USE_TOP_K": True, "MAX_POSITIONS": 5, "MIN_HOLDING_BARS": 3}
CONC = {"MAX_POSITION_WEIGHT": 0.35, "MAX_GROSS_EXPOSURE": 0.98,
        "MAX_SECTOR_WEIGHT": 0.60, "TARGET_VOLATILITY": 0.55,
        "KELLY_COLD_START_MULT": 0.90}
MDD = {"MDD_SOFT_LIMIT": 0.15, "MDD_HARD_LIMIT": 0.25, "DAILY_LOSS_LIMIT": 0.08}

STRUCTURE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "기준선(출구 수정 적용)", "delta": {}},
    {"name": "급락가드 12%만", "delta": dict(CRASH)},
    {"name": "Top-K만", "delta": dict(TOPK)},
    {"name": "MDD 완화만", "delta": dict(MDD)},
    {"name": "Top-K + MDD", "delta": {**TOPK, **MDD}},
    {"name": "Top-K + 집중(MDD 그대로)", "delta": {**TOPK, **CONC}},
    {"name": "Top-K + 집중 + MDD", "delta": {**TOPK, **CONC, **MDD}},
    {"name": "전체(+급락가드 12%)",
     "delta": {**CRASH, **TOPK, **CONC, **MDD}},
]

# ---------------------------------------------------------------- 사이징 설정군
#
# `structure` 실험에서 드러난 사실: **익스포저를 묶는 것은 종목 수도 MDD 게이트도 아니라
# 종목당 사이징이다.** 기준선이 8종목 37.3%면 종목당 4.7%인데 `MAX_POSITION_WEIGHT`는
# 18%다. 상한에 닿지도 않는다. 깎는 주체는
#
#   vol_w = TARGET_VOLATILITY / 자산변동성 = 0.18 / 0.53 = 0.34
#   kelly = KELLY_COLD_START_MULT        = 0.40
#   → 곱해서 0.14배
#
# 그래서 이 두 값만 단독으로 올려 본다. Top-K는 섞지 않는다(레짐 이중 적용 버그를
# 고친 뒤 별도 설정군으로 본다).
#
# `MAX_POSITION_WEIGHT` 0.45는 대회 규정 단일 종목 한도 50%에 여유를 둔 값이다
# (체결 슬리피지로 의도한 비중을 살짝 넘길 수 있다).
SIZING_CONFIGS: List[Dict[str, Any]] = [
    {"name": "기준선(출구 수정 적용)", "delta": {}},
    {"name": "TARGET_VOL 0.35", "delta": {"TARGET_VOLATILITY": 0.35}},
    {"name": "TARGET_VOL 0.55", "delta": {"TARGET_VOLATILITY": 0.55}},
    {"name": "KELLY 0.70", "delta": {"KELLY_COLD_START_MULT": 0.70}},
    {"name": "KELLY 0.90", "delta": {"KELLY_COLD_START_MULT": 0.90}},
    {"name": "VOL 0.35 + KELLY 0.70",
     "delta": {"TARGET_VOLATILITY": 0.35, "KELLY_COLD_START_MULT": 0.70}},
    {"name": "VOL 0.55 + KELLY 0.90",
     "delta": {"TARGET_VOLATILITY": 0.55, "KELLY_COLD_START_MULT": 0.90}},
    {"name": "VOL 0.55 + KELLY 0.90 + 비중 45%",
     "delta": {"TARGET_VOLATILITY": 0.55, "KELLY_COLD_START_MULT": 0.90,
               "MAX_POSITION_WEIGHT": 0.45}},
]

# ---------------------------------------------------------------- Top-K 설정군
#
# **이 실험이 익스포저 질문의 답이다.** 사이징 실험에서 두 가지가 드러났다.
#
#  - `TARGET_VOLATILITY` 3.1배 × `KELLY_COLD_START_MULT` 2.25배 = 공식상 7배인데
#    익스포저는 38.4% → 46.6%, 즉 1.2배만 움직였다.
#  - 종목당 상한을 18%→45%(2.5배)로 올렸더니 익스포저는 46.6% → 46.0%로 **안 움직이고**
#    거래가 −37%, MDD가 23.1%→31.4%로 뛰었다. 집중도만 올랐다.
#
# 즉 익스포저를 묶는 것은 사이징이 아니라 **신호 개수**다(`SIGNAL_BUY_THRESHOLD=0.22`
# 절대 임계값 + `USE_TOP_K=False`). 익스포저를 종목당 상한으로 나누면 동시 보유 하한이
# 나오는데, 상한 18%에서 2.1종목·45%에서 1.0종목으로 8슬롯 중 1~2개만 찬다.
#
# 이전 Top-K 측정(−0.813)은 레짐 이중 적용 버그를 측정한 것이었다. 버그는 고쳤고
# 여기서 처음으로 유효하게 잰다.
#
# **종목당 상한 상향은 단독으로는 무효이므로**(위 2번) Top-K와 묶은 변이를 따로 둔다.
# "한 번에 하나씩" 규칙과 충돌하지만, 단독 변이를 같은 실행에 함께 넣어 기여를 분리한다.
TOPK_CONFIGS: List[Dict[str, Any]] = [
    {"name": "기준선(임계값 방식)", "delta": {}},
    # --- Top-K 단독: 버그 수정이 −0.813을 되돌렸는지 ---
    {"name": "Top-K 5종목 · 잠금 없음",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 5, "MIN_HOLDING_BARS": 0}},
    {"name": "Top-K 5종목 · 잠금 3봉",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 5, "MIN_HOLDING_BARS": 3}},
    {"name": "Top-K 2종목 · 잠금 없음",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 2, "MIN_HOLDING_BARS": 0}},
    # --- 비중 상한 단독: 측정으로 무효가 확인됐지만 이 기준선에서 재확인 ---
    {"name": "비중 45% 단독", "delta": {"MAX_POSITION_WEIGHT": 0.45}},
    # --- 묶음: 대회 정책 후보. 2종목 × 45% = 익스포저 90% ---
    {"name": "Top-K 2종목 + 비중 45%",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 2, "MIN_HOLDING_BARS": 0,
               "MAX_POSITION_WEIGHT": 0.45}},
    # --- 핵심 가설: 선정·타이밍을 빼면 나아지는가 ---
    #
    # 익스포저당 수익이 벤치마크 0.62 대 전략 0.34이고 승구간이 8/21(38%)이다.
    # 격차가 "덜 담아서"가 아니라 **"잘못 골라서"** 생겼다는 의심이고, 이것이 그 검증이다.
    #
    # 구간 시작에 점수 상위 2종목을 90%로 사서 30봉 동안 손절 외에는 손대지 않는다.
    # `MIN_HOLDING_BARS=30`이 순위 교체를 잠그고, `MAX_HOLDING_BARS`를 평가창보다 크게
    # 둬 시간손절을 비활성화한다. 모델의 역할이 "언제 사고 팔까"에서 **"어느 종목이
    # 30일간 가장 많이 오를까"**로 바뀐다.
    #
    # 주의: 익절·트레일링 제거는 회전 전략에서 각각 −0.163·−0.089로 해로웠다.
    # 여기서는 "손대지 않는다"가 가설의 일부이므로 같이 끈다. 그래서 이 변이는
    # **단일 변수가 아니다** — 이기면 어느 요소가 기여했는지 후속 분해가 필요하다.
    {"name": "집중 매수보유 (2종목 90% · 손절만)",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 2,
               "MAX_POSITION_WEIGHT": 0.45, "MIN_HOLDING_BARS": 30,
               "MAX_HOLDING_BARS": 999,
               "TAKE_PROFIT_PCT": 0.0, "TRAILING_ACTIVATE_PCT": 0.0}},
]

# ---------------------------------------------------------------- 수급 설정군
#
# 수급 피처 16종이 실제로 기여하는지 본다. **`_use_flow`가 모델까지 가른다** —
# `train_on_bars`에 수급을 넘기지 않으면 모델이 수급 피처를 전부 NaN으로 보고
# 학습하고, 백테스트에서만 채우면 학습/추론 입력 분포가 달라져 해롭다.
# (실제로 이 경로가 끊겨 있어서 수급 피처가 계속 죽어 있었다.)
#
# 그래서 이 설정군은 수급 모드마다 모델을 따로 학습한다. 학습 비용이 2배다.
#
# 손절 3.5% 교차는 **기각됐으므로 뺐다.** 14구간까지 돌린 부분 결과에서
# "수급 사용 + 손절 3.5%"가 누적 −3.3%(평균 −0.21%)였다 — 수급은 좁은 손절을
# 구제하지 못한다. 설정을 4개에서 2개로 줄이면 학습은 수급 모드당 1회라 그대로지만
# 구간별 평가가 절반이 된다.
FLOW_CONFIGS: List[Dict[str, Any]] = [
    {"name": "수급 없음 (현재)", "delta": {USE_FLOW_KEY: False}},
    {"name": "수급 사용", "delta": {USE_FLOW_KEY: True}},
]

# ------------------------------------------------------- 유니버스 비교용(경량)
# 284종목은 40종목의 약 6배 느리다. 핵심 3개만 본다.
# **다른 유니버스의 수치와는 직접 비교할 수 없다**(데이터 지문이 다르다).
# 대신 "같은 실행의 벤치마크 대비 얼마나 붙었는가"로 판단한다.
UNIVERSE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "원 설정(손절 3.5%/익절 9%)", "delta": dict(_ORIG)},
    {"name": "손절 12%만 (현재 채택)", "delta": {}},
    {"name": "손절12 + 익절제거", "delta": {"TAKE_PROFIT_PCT": 0.0}},
]

# ------------------------------------------------------------- 익스포저 설정군
#
# **`topk` 실험이 앞선 진단을 반박해서 만든 설정군이다.**
#
# 나는 "익스포저를 올리는 유일한 구조적 레버는 Top-K"라고 판단했다. 측정은 반대였다
# (14구간, 지문 `1fd2cda54cd1aa77`):
#
# | 설정 | 익스포저 | 누적 |
# |---|---|---|
# | 비중 45% 단독 | **44.4%** | **+19.9%** |
# | 기준선(임계값 방식) | 40.5% | +10.2% |
# | Top-K 5종목 · 잠금 3봉 | 14.5% | +0.9% |
# | Top-K 2종목 · 잠금 없음 | **5.8%** | +2.1% |
#
# **Top-K는 익스포저를 올리는 게 아니라 무너뜨린다.** 목표 보유 수를 K로 묶고 순위에서
# 밀린 종목을 매 봉 청산하므로, 임계값 방식이 8슬롯까지 쌓아 들고 가던 포지션이
# 강제로 회전한다. 레짐 이중 적용 버그를 고친 뒤에도 그렇다 — 그 버그가 유일한
# 원인이 아니었다.
#
# 그리고 7개 설정에서 **익스포저와 누적수익의 상관이 +0.902**다. 익스포저 자체가
# 지배 변수다.
#
# `sizing` 실험이 "비중 상한 상향은 무효"라고 결론 낸 것은 **단독으로 재지 않았기
# 때문이다** — `VOL 0.55 + KELLY 0.90 + 비중 45%` 묶음으로만 쟀고, 해로운 배수 두 개가
# 비중의 기여를 덮었다. "한 번에 하나씩" 규칙을 어긴 대가를 또 치렀다.
#
# 그래서 여기서는 Top-K를 쓰지 않고 익스포저를 직접 올리는 두 레버를 각각 본다.
#  - `MAX_POSITION_WEIGHT`: 종목당 더 크게 (규정 상한 50%)
#  - `SIGNAL_BUY_THRESHOLD`: 문턱을 낮춰 **더 많은 종목이 통과**하게
EXPOSURE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "기준선", "delta": {}},
    # 섹터 상한을 함께 올리지 않으면 종목당 상한은 35%에서 잘린다.
    # 이 변이는 그 사실의 확인용이다 — "비중 45%"와 수치가 같게 나와야 한다.
    {"name": "비중 45% (섹터 35% 유지)", "delta": {"MAX_POSITION_WEIGHT": 0.45}},
    # --- 실효 상한을 올리는 변이: 비중과 섹터를 함께 올린다 ---
    {"name": "비중 45% + 섹터 45%",
     "delta": {"MAX_POSITION_WEIGHT": 0.45, "MAX_SECTOR_WEIGHT": 0.45}},
    {"name": "비중 50% + 섹터 50%",
     "delta": {"MAX_POSITION_WEIGHT": 0.50, "MAX_SECTOR_WEIGHT": 0.50}},
    # --- 보유 종목 수 레버: 문턱을 낮춰 더 많은 종목이 통과하게 한다 ---
    # Top-K와 달리 강제 회전이 없다. Top-K는 순위에서 밀린 종목을 매 봉 청산해
    # 익스포저를 5.8~14.5%로 무너뜨렸다.
    {"name": "임계값 0.15", "delta": {"SIGNAL_BUY_THRESHOLD": 0.15}},
    {"name": "임계값 0.10", "delta": {"SIGNAL_BUY_THRESHOLD": 0.10}},
    # --- 두 레버가 독립인지 ---
    {"name": "비중50+섹터50 + 임계값 0.15",
     "delta": {"MAX_POSITION_WEIGHT": 0.50, "MAX_SECTOR_WEIGHT": 0.50,
               "SIGNAL_BUY_THRESHOLD": 0.15}},
    {"name": "비중50+섹터50 + 임계값 0.10",
     "delta": {"MAX_POSITION_WEIGHT": 0.50, "MAX_SECTOR_WEIGHT": 0.50,
               "SIGNAL_BUY_THRESHOLD": 0.10}},
    # --- 집중 보유 가설, Top-K 없이 재설계 ---
    #
    # `topk` 실험의 "집중 매수보유"는 **측정 수단이 오염돼 있었다.** 2종목을 Top-K로
    # 강제했는데 Top-K 자체가 익스포저를 무너뜨려, "2종목 90%로 들고 버틴다"는 가설이
    # 실제로는 13.0%만 들고 있었다. 가설이 아니라 수단을 측정한 셈이다.
    #
    # 그래서 Top-K를 쓰지 않고 실효 상한만 규정 한도까지 올린 뒤 **청산을 손절로만**
    # 제한한다. 익절·트레일링·시간손절·신호반전을 모두 끈다(`MAX_HOLDING_BARS`를
    # 평가창 30봉보다 크게, `SIGNAL_SELL_THRESHOLD`를 도달 불가능한 값으로).
    # 위 "비중 50% + 섹터 50%"와 **청산 규칙만** 다르므로, 둘을 비교하면
    # "손대지 않는다"의 기여가 단독으로 분리된다.
    {"name": "비중50+섹터50 + 손절만(집중 보유)",
     "delta": {"MAX_POSITION_WEIGHT": 0.50, "MAX_SECTOR_WEIGHT": 0.50,
               "TAKE_PROFIT_PCT": 0.0, "TRAILING_ACTIVATE_PCT": 0.0,
               "MAX_HOLDING_BARS": 999, "SIGNAL_SELL_THRESHOLD": -9.9}},
]

CONFIG_SETS: Dict[str, List[Dict[str, Any]]] = {
    "flow": FLOW_CONFIGS,
    "universe": UNIVERSE_CONFIGS,
    "exits": CONFIGS,
    "structure": STRUCTURE_CONFIGS,
    "sizing": SIZING_CONFIGS,
    "topk": TOPK_CONFIGS,
    "exposure": EXPOSURE_CONFIGS,
}


def stratified_sample(symbols, n: int, settings, seed: int = 16) -> List[str]:
    """지수 소속/비소속 비율을 유지해 n종목을 뽑는다.

    ## 왜 층화인가

    284종목 전량 측정은 36종목의 약 6배이고 28구간이라 **15시간대**다. 가용 메모리가
    2.4GB인 환경에서 이미 한 번 중단됐다(5/28).

    그냥 앞에서 n개를 자르면(`--symbols`) `universe.json`의 정렬 순서에 따라 한쪽
    집단으로 쏠린다. 유니버스 확장의 요지가 **지수 외 종목의 꼬리**(변동성 1.38배,
    30봉 P(+100%) 1.13%→2.92%)이므로, 그 집단이 빠지면 측정할 대상 자체가 사라진다.
    그래서 두 집단의 비율을 유지한 채 각각에서 뽑는다.

    ## 한계

    **다른 유니버스의 절대 수치와 비교할 수 없다**(지문이 다르다). 판정은 같은 실행
    안의 벤치마크 대비 거리로만 한다. 시드를 고정해 재현은 보장한다.
    """
    members: set = set()
    for lst in settings.index_members().values():
        members.update(lst)
    inside = [s for s in symbols if s in members]
    outside = [s for s in symbols if s not in members]
    if not members:
        # 지수 정보가 없으면 층화할 근거가 없다. 단순 무작위로 떨어진다.
        rng = np.random.default_rng(seed)
        picked = sorted(rng.choice(list(symbols), size=n, replace=False).tolist())
        logger.warning(f"[Ablate] 지수 멤버십 정보 없음 — 단순 무작위 {len(picked)}종목")
        return picked

    share = len(inside) / len(symbols)
    n_in = min(len(inside), max(1, round(n * share)))
    n_out = min(len(outside), max(1, n - n_in))
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(inside, size=n_in, replace=False).tolist()
                    + rng.choice(outside, size=n_out, replace=False).tolist())
    logger.info(
        f"[Ablate] 층화 추출 {len(picked)}종목 — 지수 소속 {n_in}/{len(inside)} · "
        f"지수 외 {n_out}/{len(outside)} (전체 비율 {share:.1%}, 시드 {seed})")
    return picked


# 꼬리 지표를 판정할 임계값. 지난 대회 10위권이 +100%였다.
TAIL_THRESHOLDS = (0.20, 0.50, 1.00)


def tail_stats(seg_returns: np.ndarray) -> Dict[str, Any]:
    """구간 수익률의 **분포**를 요약한다.

    ## 왜 누적수익과 Sharpe로는 안 되는가

    `total_return`은 31구간을 이어붙여 복리로 누적한 값이고 `sharpe`는 평균/분산이다.
    둘 다 **평균 성과**를 재는데, 대회는 **단 한 창(30거래일)**이고 심사는 절대수익
    top10이다. 보상이 계단함수이므로 최적화 대상은 `E[R]`이 아니라 `P(R > 임계)`다.

    **Sharpe는 방향이 반대다.** 분산을 벌점하는데 꼬리 전략은 분산을 사려고 한다.
    실측으로도 드러났다 — 19구간 측정에서 8개 설정 전부 구간 최대가 +14.1% 이하였고
    +20%를 넘긴 구간이 **0/19**였다. 누적수익으로는 설정 간 순위가 갈렸지만, 목표
    기준으로는 전부 똑같이 실패였다. 그 사실이 누적수익·Sharpe에는 보이지 않는다.

    그래서 구간 하나를 **대회 1회**로 보고 그 분포의 오른쪽을 본다.
    """
    r = np.asarray(seg_returns, dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {}
    out: Dict[str, Any] = {
        "seg_median": float(np.median(r)),
        "seg_p75": float(np.percentile(r, 75)),
        "seg_p90": float(np.percentile(r, 90)),
        "seg_max": float(r.max()),
        "seg_min": float(r.min()),
    }
    for th in TAIL_THRESHOLDS:
        out[f"hits_{int(th * 100)}"] = int((r >= th).sum())
    return out


def build_summary(variants, pieces, trades, expo, seg_rows) -> Dict[str, Any]:
    """지금까지 끝난 구간만으로 설정별 집계를 만든다. 체크포인트와 최종 저장이 공용한다."""
    summary: Dict[str, Any] = {}
    for name, _sv, _uf in variants:
        arr = np.concatenate(pieces[name]) if pieces[name] else np.array([])
        if not len(arr):
            continue
        # 구간별 수익률 = 대회 1회의 결과. 꼬리 판정의 표본이다.
        per_seg = np.array([r.get(name, np.nan) for r in seg_rows], dtype=float)
        summary[name] = {
            "sharpe": sharpe_ratio(arr),
            "total_return": float(np.prod(1.0 + arr) - 1.0),
            "mdd": max_drawdown_from_returns(arr),
            "trades": trades[name],
            "exposure": float(np.mean(expo[name])) if expo[name] else float("nan"),
            "positive_segments": sum(1 for r in seg_rows if r.get(name, 0.0) > 0),
            "segments": len(seg_rows),
            **tail_stats(per_seg),
        }
    return summary


def write_report(out_path: Path, seg_rows, summary, fp, cfg,
                 complete: bool, planned: int = 0) -> None:
    """부분 결과라도 디스크에 남긴다.

    ## 왜 구간마다 쓰는가 (겪은 일)

    예전에는 모든 구간이 끝난 뒤에만 JSON을 썼다. 가용 메모리가 2.4GB인 환경에서 측정이
    **세 번 중단됐고**(8/31 · 5/28 · 14/31) 매번 수 시간이 전손됐다. 진행 로그에도
    첫·마지막 설정만 찍혀 중간 설정은 복구할 수 없었다.

    장시간 측정에서 중단은 예외가 아니라 정상 경로다. 그래서 구간마다 덮어쓰고
    `complete` 플래그로 완주 여부를 구분한다. **`complete: false`인 파일의 수치는
    구간 수가 다르므로 완주한 실행과 직접 비교하면 안 된다.**
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "complete": bool(complete),
        "segments_done": len(seg_rows),
        "segments_planned": int(planned) or len(seg_rows),
        "segments": seg_rows,
        "summary": summary,
        "data_fingerprint": fp,
        "config": cfg,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(out_path)   # 쓰는 중에 죽어도 기존 체크포인트가 깨지지 않는다


def variant_settings(base, delta: Dict[str, Any]):
    """기준 설정을 복사해 몇 개 항목만 덮어쓴다. `_use_flow`는 설정이 아니라 스위치다."""
    s = base.model_copy(deep=True)
    for key, value in delta.items():
        if key == USE_FLOW_KEY:
            continue
        setattr(s, key, value)
    return s


def panel_fingerprint(panel: Dict[str, Any]) -> Dict[str, Any]:
    """정렬된 패널의 지문. **다른 실행 결과와 비교 가능한지 판단하는 근거다.**

    이걸 넣은 이유가 있다. 출구 실험(09-28)과 사이징 실험(09-29)에서 **같은 설정이
    Sharpe 0.873 / +33.2% 와 0.263 / +7.0% 로 갈렸다.** 21구간 전부 불일치했고,
    모델 학습은 결정적이며 코드 변경도 등가임을 확인했다. 남는 설명은 그 사이
    데이터 수집이 40종목을 재수집해 파일을 덮어썼다는 것이다.

    수익률은 수정주가 배수에 불변이라 벤치마크는 그대로였고, 그래서 **조용히**
    비교 불가가 됐다. 지문을 남겨 두면 다음에는 바로 드러난다.
    """
    import hashlib

    h = hashlib.md5()
    for sym in sorted(panel):
        df = panel[sym]
        h.update(sym.encode())
        h.update(np.ascontiguousarray(
            df["close"].to_numpy(dtype=float)).tobytes())
    first = panel[sorted(panel)[0]]
    return {
        "hash": h.hexdigest()[:16],
        "symbols": len(panel),
        "bars": int(min(len(d) for d in panel.values())),
        "ts_from": str(first["ts"].iloc[0]),
        "ts_to": str(first["ts"].iloc[-1]),
    }


def benchmark_returns(panel: Dict[str, Any], start: int, end: int) -> np.ndarray:
    """동일가중 매수보유 — 같은 구간의 비교 기준."""
    rows = []
    for df in panel.values():
        close = df["close"].to_numpy(dtype=float)[start:end]
        if len(close) < 2:
            continue
        rows.append(np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1.0))
    if not rows:
        return np.array([])
    width = min(len(r) for r in rows)
    return np.nanmean(np.vstack([r[:width] for r in rows]), axis=0)


def run_variant(panel, model, settings, start: int, end: int,
                featured: Optional[Dict], flow_panel: Optional[Dict]
                ) -> Tuple[Optional[Tuple[np.ndarray, int, float]], Optional[Dict]]:
    """한 설정으로 [start, end) 구간을 평가한다. `featured`는 설정 간 공유한다."""
    if end - start < 10:
        return None, featured
    strategy = EnsembleAlphaStrategy(settings, model=model)
    sliced = slice_panel(panel, 0, end)
    if featured is None:
        featured = strategy.precompute(sliced, flow_panel)
    else:
        # 출구 규칙은 피처·신호에 영향을 주지 않으므로 계산 결과를 그대로 재사용한다.
        cached = featured
        strategy.precompute = lambda _p, _f=None: cached  # type: ignore[assignment]
    engine = BacktestEngine(sliced, strategy=strategy,
                            risk_manager=RiskManager(settings), settings=settings,
                            initial_capital=settings.INITIAL_CAPITAL,
                            flow_panel=flow_panel)
    try:
        result = engine.run(warmup=max(start, MIN_WARMUP), verbose=False)
    except ValueError:
        return None, featured
    equity = result.curve["equity"].to_numpy(dtype=float)
    if len(equity) < 5:
        return None, featured
    rets = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1.0)
    if "exposure" in result.curve.columns:
        mean_expo = float(np.nanmean(result.curve["exposure"].to_numpy(dtype=float)))
    else:
        mean_expo = float("nan")
    return (rets, len(result.trades), mean_expo), featured


def main() -> int:
    ap = argparse.ArgumentParser(description="출구 규칙 절제 실험")
    ap.add_argument("--source", type=str, default="")
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--sample", type=int, default=0,
                    help="유니버스에서 N종목을 지수 소속/비소속 비율을 유지해 층화 추출"
                         "(284종목 전량이 시간상 불가할 때 쓴다)")
    ap.add_argument("--sample-seed", type=int, default=16,
                    help="층화 추출 시드. 같은 시드면 같은 종목이 뽑힌다")
    ap.add_argument("--symbols", type=int, default=0,
                    help="앞에서 N종목만. 0=전체. 기본이 40이면 유니버스를 284로 "
                         "설정해 뒀는데도 조용히 40종목만 측정된다")
    ap.add_argument("--train-window", type=int, default=500)
    ap.add_argument("--step", type=int, default=30, help="평가 구간 길이(대회 30거래일)")
    ap.add_argument("--max-segments", type=int, default=0, help="앞에서 N구간만")
    ap.add_argument("--set", dest="config_set", type=str, default="exits",
                    choices=sorted(CONFIG_SETS), help="측정할 설정군")
    ap.add_argument("--flow-dir", type=str, default="",
                    help="수급 데이터 디렉토리. 기본은 <source>/flow. 비활성화된 데이터를 "
                         "파일 이동 없이 켜려면 경로를 직접 지정한다")
    ap.add_argument("--no-flow", action="store_true", help="수급 데이터를 로드하지 않는다")
    ap.add_argument("--flow-default", action="store_true",
                    help="설정에 _use_flow가 없으면 기본으로 수급을 쓴다")
    ap.add_argument("--out", type=str, default="reports/ablate_exits.json")
    args = ap.parse_args()

    base = get_settings()
    setup_logging(base.LOG_DIR, base.LOG_LEVEL)
    symbols = base.UNIVERSE[: args.symbols] if args.symbols else base.UNIVERSE
    if args.sample and args.sample < len(symbols):
        symbols = stratified_sample(symbols, args.sample, base, args.sample_seed)

    flow_panel: Dict[str, Any] = {}
    if args.source:
        panel = load_panel_from_dir(args.source, symbols)
        if not args.no_flow:
            from strategy.flow_features import load_flow_dir
            flow_dir = Path(args.flow_dir) if args.flow_dir \
                else Path(args.source) / "flow"
            flow_panel = load_flow_dir(flow_dir, symbols)
            logger.info(f"[Ablate] 수급 {len(flow_panel)}종목 로드 ({flow_dir})"
                        if flow_panel else
                        f"[Ablate] 수급 데이터 없음 ({flow_dir}) — 가격 피처만 사용")
    else:
        panel = generate_synthetic_panel(
            symbols, n_bars=args.days, seed=base.MOCK_SEED,
            sector_map={s: base.sector_of(s) for s in symbols})
    if not panel:
        logger.error("데이터를 찾지 못했습니다.")
        return 1

    panel = align_panel(panel, base.MIN_HISTORY_BARS)
    n_bars = min(len(df) for df in panel.values())
    ts_series = panel[list(panel)[0]]["ts"].astype(str).tolist()
    fp = panel_fingerprint(panel)
    logger.info(f"[Ablate] 데이터 지문 {fp['hash']} ({fp['symbols']}종목 · {fp['bars']}봉 · "
                f"{fp['ts_from']}~{fp['ts_to']}) — 지문이 다른 실행과는 비교할 수 없다")
    logger.info(f"[Ablate] {len(panel)}종목 · {n_bars}봉 · 학습창 {args.train_window} · "
                f"평가 {args.step}봉 · 설정군 {args.config_set} "
                f"({len(CONFIG_SETS[args.config_set])}개)")

    chosen = CONFIG_SETS[args.config_set]
    # (이름, 설정, 수급 사용 여부). 수급은 모델·피처를 갈라야 하므로 따로 들고 간다.
    variants = [(c["name"], variant_settings(base, c["delta"]),
                 bool(c["delta"].get(USE_FLOW_KEY, args.flow_default)))
                for c in chosen]
    needs_flow = {v[2] for v in variants}
    if True in needs_flow and not flow_panel:
        logger.error("[Ablate] 수급을 쓰는 설정이 있는데 수급 패널이 비어 있습니다. "
                     "--flow-dir 를 확인하십시오.")
        return 1
    pieces: Dict[str, List[np.ndarray]] = {v[0]: [] for v in variants}
    trades: Dict[str, int] = {v[0]: 0 for v in variants}
    expo: Dict[str, List[float]] = {v[0]: [] for v in variants}
    seg_rows: List[Dict[str, Any]] = []
    bench: List[np.ndarray] = []

    started = time.monotonic()
    index = 0
    test_start = max(args.train_window, MIN_WARMUP)
    # 체크포인트를 쓰려면 저장 경로와 설정 메타를 루프 진입 전에 확정해야 한다.
    out_path = Path(args.out)
    run_cfg = {"train_window": args.train_window, "step": args.step,
               "symbols": len(panel), "n_bars": n_bars,
               "config_set": args.config_set, "flow_symbols": len(flow_panel)}
    planned = max(0, (n_bars - test_start) // args.step)
    logger.info(f"[Ablate] 구간 {planned}개 예정 · 체크포인트 → {out_path}")
    while test_start + args.step <= n_bars:
        if args.max_segments and index >= args.max_segments:
            break
        test_end = min(test_start + args.step, n_bars)
        train_start = max(0, test_start - args.train_window)
        train_end = max(train_start + 1,
                        test_start - base.LABEL_HORIZON - base.CV_EMBARGO)
        index += 1
        # 수급 모드마다 **모델을 따로 학습한다.** 학습과 평가가 같은 입력을 봐야
        # 하기 때문이다(한쪽만 수급을 쓰면 학습/추론 분포가 달라진다).
        models: Dict[bool, Any] = {}
        try:
            for use_flow in sorted(needs_flow):
                models[use_flow] = train_on_bars(
                    panel, np.arange(train_start, train_end), base,
                    flow_panel if use_flow else None)
        except Exception as exc:
            logger.warning(f"[Ablate] 구간 {index} 학습 실패: {exc}")
            test_start += args.step
            continue

        feat_cache: Dict[bool, Optional[Dict]] = {k: None for k in needs_flow}
        row: Dict[str, Any] = {
            "index": index,
            "from": ts_series[test_start] if test_start < len(ts_series) else "",
            "to": ts_series[min(test_end, len(ts_series)) - 1] if ts_series else "",
        }
        b = benchmark_returns(panel, test_start, test_end)
        if len(b):
            bench.append(b)
            row["benchmark"] = float(np.prod(1.0 + b) - 1.0)

        for name, sv, use_flow in variants:
            out, feat_cache[use_flow] = run_variant(
                panel, models[use_flow], sv, test_start, test_end,
                feat_cache[use_flow], flow_panel if use_flow else None)
            if out is None:
                continue
            rets, n_trades, mean_expo = out
            pieces[name].append(rets)
            trades[name] += int(n_trades)
            if np.isfinite(mean_expo):
                expo[name].append(mean_expo)
            row[name] = float(np.prod(1.0 + rets) - 1.0)

        seg_rows.append(row)
        # **모든 설정을 찍는다.** 예전에는 첫·마지막만 찍어서, 측정이 중단되면 중간
        # 설정의 결과를 로그로도 복구할 수 없었다(수급 실험에서 핵심 비교를 잃었다).
        detail = " · ".join(
            f"{name} {row.get(name, 0.0) * 100:+.2f}%" for name, _sv, _uf in variants)
        logger.info(
            f"[Ablate] 구간 {index}/{planned} {row['from']}~{row['to']} "
            f"벤치 {row.get('benchmark', 0.0) * 100:+.2f}% · {detail} "
            f"({time.monotonic() - started:.0f}s)")

        # 구간마다 부분 결과를 남긴다. 중단되어도 여기까지는 보존된다.
        write_report(out_path, seg_rows,
                     build_summary(variants, pieces, trades, expo, seg_rows),
                     fp, run_cfg, complete=False, planned=planned)
        test_start += args.step

    # ------------------------------------------------------------------ 집계
    print()
    print("=" * 98)
    print(f" 절제 실험 [{args.config_set}] — {len(seg_rows)}구간 x {args.step}봉 out-of-sample")
    print(f" 데이터 지문 {fp['hash']} · {fp['symbols']}종목 x {fp['bars']}봉 "
          f"({fp['ts_from']}~{fp['ts_to']}) · 수급 {len(flow_panel)}종목")
    print(" 지문이 다른 실행의 수치와는 비교할 수 없다.")
    print("=" * 98)
    summary = build_summary(variants, pieces, trades, expo, seg_rows)
    bench_seg = np.array([r.get("benchmark", np.nan) for r in seg_rows], dtype=float)

    # ---- 꼬리 기준 (대회 판정) ----
    # 구간 하나 = 대회 1회(30거래일)다. 심사가 절대수익 top10이므로 이 분포의
    # 오른쪽이 판정 기준이고, 아래 평균 기준은 참고용으로만 찍는다.
    print(f" [꼬리 기준] 구간 1개 = 대회 1회. 지난 대회 10위권 +100%")
    print(f" {'설정':<30} {'중앙값':>8} {'p75':>8} {'p90':>8} {'최대':>9} "
          f"{'+20%':>6} {'+50%':>6} {'+100%':>6} {'익스포저':>9}")
    print("-" * 98)
    bt = tail_stats(bench_seg)
    if bt:
        print(f" {'벤치마크(동일가중)':<30} {bt['seg_median'] * 100:>+7.1f}% "
              f"{bt['seg_p75'] * 100:>+7.1f}% {bt['seg_p90'] * 100:>+7.1f}% "
              f"{bt['seg_max'] * 100:>+8.1f}% {bt['hits_20']:>6d} "
              f"{bt['hits_50']:>6d} {bt['hits_100']:>6d} {'100.0%':>9}")
    # 꼬리가 두꺼운 순으로 정렬한다 — p90이 1차, 최대가 2차 기준.
    ranked = sorted((n for n, _s, _u in variants if n in summary),
                    key=lambda n: (summary[n].get("seg_p90", -9),
                                   summary[n].get("seg_max", -9)), reverse=True)
    for name in ranked:
        s = summary[name]
        print(f" {name:<30} {s.get('seg_median', 0) * 100:>+7.1f}% "
              f"{s.get('seg_p75', 0) * 100:>+7.1f}% {s.get('seg_p90', 0) * 100:>+7.1f}% "
              f"{s.get('seg_max', 0) * 100:>+8.1f}% {s.get('hits_20', 0):>6d} "
              f"{s.get('hits_50', 0):>6d} {s.get('hits_100', 0):>6d} "
              f"{s['exposure'] * 100:>8.1f}%")
    print("=" * 98)

    # ---- 평균 기준 (참고) ----
    bench_all = np.concatenate(bench) if bench else np.array([])
    print(" [평균 기준 — 참고용] Sharpe는 분산을 벌점하므로 꼬리 목표와 방향이 반대다")
    if len(bench_all):
        print(f" {'벤치마크(동일가중)':<30} Sharpe {sharpe_ratio(bench_all):>6.3f}  "
              f"누적 {(float(np.prod(1.0 + bench_all)) - 1.0) * 100:>+8.1f}%  "
              f"MDD {max_drawdown_from_returns(bench_all) * 100:>5.1f}%")
    print("-" * 98)
    print(f" {'설정':<30} {'Sharpe':>7} {'누적수익':>10} {'MDD':>7} "
          f"{'거래':>6} {'양수구간':>9}")
    print("-" * 98)
    for name, _sv, _uf in variants:
        s = summary.get(name)
        if s is None:
            continue
        print(f" {name:<30} {s['sharpe']:>7.3f} {s['total_return'] * 100:>+9.1f}% "
              f"{s['mdd'] * 100:>6.1f}% {s['trades']:>6d} "
              f"{s['positive_segments']:>5d}/{len(seg_rows)}")
    print("=" * 98)

    write_report(out_path, seg_rows, summary, fp, run_cfg,
                 complete=True, planned=planned)
    logger.success(f"[Ablate] 저장 → {out_path.resolve()} (완주 {len(seg_rows)}구간)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
