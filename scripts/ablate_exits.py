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
# 레짐 이중 적용 버그를 고친 뒤 Top-K 자체를 다시 본다. 이전 측정(−0.813)은
# 버그를 측정한 것이었다. `MIN_HOLDING_BARS`(슬롯 잠금)와 `MAX_POSITIONS`를 교차한다.
TOPK_CONFIGS: List[Dict[str, Any]] = [
    {"name": "기준선(임계값 방식)", "delta": {}},
    {"name": "Top-K 8종목 · 잠금 없음",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 8, "MIN_HOLDING_BARS": 0}},
    {"name": "Top-K 8종목 · 잠금 3봉",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 8, "MIN_HOLDING_BARS": 3}},
    {"name": "Top-K 5종목 · 잠금 없음",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 5, "MIN_HOLDING_BARS": 0}},
    {"name": "Top-K 5종목 · 잠금 3봉",
     "delta": {"USE_TOP_K": True, "MAX_POSITIONS": 5, "MIN_HOLDING_BARS": 3}},
]

# ---------------------------------------------------------------- 수급 설정군
#
# 수급 피처 16종이 실제로 기여하는지 본다. **`_use_flow`가 모델까지 가른다** —
# `train_on_bars`에 수급을 넘기지 않으면 모델이 수급 피처를 전부 NaN으로 보고
# 학습하고, 백테스트에서만 채우면 학습/추론 입력 분포가 달라져 해롭다.
# (실제로 이 경로가 끊겨 있어서 수급 피처가 계속 죽어 있었다.)
#
# 그래서 이 설정군은 수급 모드마다 모델을 따로 학습한다. 학습 비용이 2배다.
FLOW_CONFIGS: List[Dict[str, Any]] = [
    {"name": "수급 없음 (현재)", "delta": {USE_FLOW_KEY: False}},
    {"name": "수급 사용", "delta": {USE_FLOW_KEY: True}},
    {"name": "수급 없음 + 손절 3.5%",
     "delta": {USE_FLOW_KEY: False, "STOP_LOSS_PCT": 0.035}},
    {"name": "수급 사용 + 손절 3.5%",
     "delta": {USE_FLOW_KEY: True, "STOP_LOSS_PCT": 0.035}},
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

CONFIG_SETS: Dict[str, List[Dict[str, Any]]] = {
    "flow": FLOW_CONFIGS,
    "universe": UNIVERSE_CONFIGS,
    "exits": CONFIGS,
    "structure": STRUCTURE_CONFIGS,
    "sizing": SIZING_CONFIGS,
    "topk": TOPK_CONFIGS,
}


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
        # 진행 로그는 설정군의 첫(기준선)·마지막(누적) 항목을 보여준다.
        # 상수 이름을 쓰면 설정군을 바꿀 때 키가 어긋나 0.00%로 찍힌다.
        first_name, last_name = chosen[0]["name"], chosen[-1]["name"]
        logger.info(
            f"[Ablate] 구간 {index} {row['from']}~{row['to']} "
            f"벤치 {row.get('benchmark', 0.0) * 100:+.2f}% · "
            f"{first_name} {row.get(first_name, 0.0) * 100:+.2f}% · "
            f"{last_name} {row.get(last_name, 0.0) * 100:+.2f}% "
            f"({time.monotonic() - started:.0f}s)")
        test_start += args.step

    # ------------------------------------------------------------------ 집계
    print()
    print("=" * 98)
    print(f" 절제 실험 [{args.config_set}] — {len(seg_rows)}구간 x {args.step}봉 out-of-sample")
    print(f" 데이터 지문 {fp['hash']} · {fp['symbols']}종목 x {fp['bars']}봉 "
          f"({fp['ts_from']}~{fp['ts_to']}) · 수급 {len(flow_panel)}종목")
    print(" 지문이 다른 실행의 수치와는 비교할 수 없다.")
    print("=" * 98)
    bench_all = np.concatenate(bench) if bench else np.array([])
    if len(bench_all):
        print(f" 동일가중 매수보유(벤치마크)          "
              f"Sharpe {sharpe_ratio(bench_all):>6.3f}  "
              f"누적 {(float(np.prod(1.0 + bench_all)) - 1.0) * 100:>+8.1f}%  "
              f"MDD {max_drawdown_from_returns(bench_all) * 100:>5.1f}%")
    print("-" * 98)
    header = (f" {'설정':<30} {'Sharpe':>7} {'누적수익':>10} {'MDD':>7} "
              f"{'거래':>6} {'익스포저':>9} {'양수구간':>9}")
    print(header)
    print("-" * 98)
    summary: Dict[str, Any] = {}
    for name, _sv, _uf in variants:
        arr = np.concatenate(pieces[name]) if pieces[name] else np.array([])
        if not len(arr):
            continue
        seg_pos = sum(1 for r in seg_rows if r.get(name, 0.0) > 0)
        sh = sharpe_ratio(arr)
        tot = float(np.prod(1.0 + arr) - 1.0)
        mdd = max_drawdown_from_returns(arr)
        ex = float(np.mean(expo[name])) if expo[name] else float("nan")
        summary[name] = {"sharpe": sh, "total_return": tot, "mdd": mdd,
                         "trades": trades[name], "exposure": ex,
                         "positive_segments": seg_pos, "segments": len(seg_rows)}
        print(f" {name:<30} {sh:>7.3f} {tot * 100:>+9.1f}% {mdd * 100:>6.1f}% "
              f"{trades[name]:>6d} {ex * 100:>8.1f}% {seg_pos:>5d}/{len(seg_rows)}")
    print("=" * 98)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"segments": seg_rows, "summary": summary,
         "data_fingerprint": fp,
         "config": {"train_window": args.train_window, "step": args.step,
                    "symbols": len(panel), "n_bars": n_bars,
                    "config_set": args.config_set,
                    "flow_symbols": len(flow_panel)}},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.success(f"[Ablate] 저장 → {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
