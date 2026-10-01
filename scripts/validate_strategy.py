"""전략 과적합 검증 — CPCV 경로 분포 + Deflated Sharpe + PBO.

    python scripts/validate_strategy.py                       # 합성 데이터로 검증
    python scripts/validate_strategy.py --source data/cache   # 실데이터로 검증
    python scripts/validate_strategy.py --sweep               # 파라미터 탐색의 과적합 확률까지

파라미터를 바꿀 때마다 이 스크립트를 돌린다. 단일 백테스트의 Sharpe는 기간을 어떻게
잡았느냐에 따라 쉽게 달라지지만, 경로 분포와 DSR은 그것을 드러낸다.

판정 기준
    DSR ≥ 0.95  통계적으로 유의. 실전에 넣어도 되는 근거가 된다
    DSR < 0.95  운으로 설명 가능. 파라미터를 되돌리거나 표본을 늘린다
    PBO ≥ 0.5   그 파라미터 탐색은 무의미하다. 학습 구간 1등이 검증에서 중간 이하로 떨어진다
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.data import generate_synthetic_panel, load_panel_from_dir  # noqa: E402
from backtest.engine import BacktestEngine, align_panel  # noqa: E402
from backtest.validation import (  # noqa: E402
    CombinatorialPurgedCV,
    ValidationReport,
    deflated_sharpe_ratio,
    max_drawdown_from_returns,
    pbo_cscv,
    sharpe_ratio,
)
from config.settings import get_settings  # noqa: E402
from strategy.ai_model import AlphaModel, EnsembleAlphaStrategy  # noqa: E402
from strategy.features import build_panel_features  # noqa: E402
from strategy.risk_manager import RiskManager  # noqa: E402
from utils.logger import logger, setup_logging  # noqa: E402

# 학습·구간평가 헬퍼는 워크포워드와 공용이다(중복 구현을 두지 않는다)
from backtest.walkforward import segment_returns as _segment_returns  # noqa: E402
from backtest.walkforward import train_on_bars as _train_on  # noqa: E402


def run_cpcv(panel: Dict[str, pd.DataFrame], settings, n_groups: int, n_test: int,
             trial_sharpes: Optional[List[float]] = None) -> ValidationReport:
    n_bars = min(len(df) for df in panel.values())
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test=n_test,
                               horizon=settings.LABEL_HORIZON, embargo=settings.CV_EMBARGO)
    splits = list(cv.split(n_bars))
    logger.info(f"[Validate] {n_bars}봉 · {n_groups}그룹 · 분할 {len(splits)}개 "
                f"· 경로 {cv.n_paths}개 예상")

    group_returns: Dict[Tuple[int, int], np.ndarray] = {}
    split_sharpes: List[float] = []
    started = time.monotonic()

    for i, split in enumerate(splits, 1):
        try:
            model = _train_on(panel, split.train_idx, settings)
        except Exception as exc:
            logger.warning(f"[Validate] 분할 {i}/{len(splits)} 학습 실패: {exc}")
            continue

        pieces = []
        for group in split.test_groups:
            lo, hi = split.group_slices[group]
            returns = _segment_returns(panel, model, settings, lo, hi)
            if returns is None:
                continue
            group_returns[(split.split_id, group)] = returns
            pieces.append(returns)

        if pieces:
            joined = np.concatenate(pieces)
            split_sharpes.append(sharpe_ratio(joined))
            logger.info(f"[Validate] 분할 {i}/{len(splits)} 그룹{split.test_groups} "
                        f"Sharpe={split_sharpes[-1]:.3f} "
                        f"({time.monotonic() - started:.0f}s 경과)")

    paths = cv.assemble_paths(group_returns)
    path_sharpes = [sharpe_ratio(p) for p in paths]
    path_mdds = [max_drawdown_from_returns(p) for p in paths]

    combined = np.concatenate(paths) if paths else np.array([0.0])

    # DSR은 "서로 다른 설정을 몇 개나 시험했는가"를 보정하는 값이다.
    # CPCV 분할별 Sharpe는 같은 설정의 기간별 변동이므로 여기에 넣으면 안 된다.
    # 시험한 설정들의 Sharpe를 외부에서 받았을 때만 DSR을 계산한다.
    if trial_sharpes:
        dsr = deflated_sharpe_ratio(combined, trial_sharpes)
        n_configs = len(trial_sharpes)
    else:
        dsr = deflated_sharpe_ratio(combined, [0.0, 0.0])   # 기준선 0 → PSR과 같아진다
        dsr["threshold_sr"] = 0.0
        dsr["n_trials"] = 0
        n_configs = 0

    report = ValidationReport(path_sharpes=path_sharpes, path_mdds=path_mdds,
                              dsr=dsr, n_groups=n_groups, n_splits=len(splits))
    report.n_configs = n_configs
    report.split_sharpes = split_sharpes
    return report


def run_sweep(panel: Dict[str, pd.DataFrame], settings) -> Dict[str, float]:
    """파라미터 조합별 수익률 행렬을 만들어 PBO를 계산한다.

    지금 고민 중인 축(진입 임계값, 목표 변동성)을 바꿔가며, 그 탐색 자체가
    과적합인지 본다. PBO가 0.5에 가까우면 어떤 조합을 고르든 의미가 없다.
    """
    thresholds = [0.18, 0.22, 0.26]
    vol_targets = [0.12, 0.15, 0.18]
    combos = list(itertools.product(thresholds, vol_targets))

    n_bars = min(len(df) for df in panel.values())
    train_end = int(n_bars * 0.6)
    logger.info(f"[Sweep] 조합 {len(combos)}개 · 학습 {train_end}봉 · 평가 {n_bars - train_end}봉")

    model = _train_on(panel, np.arange(train_end), settings)

    columns: List[np.ndarray] = []
    labels: List[str] = []
    for threshold, vol in combos:
        orig = (settings.SIGNAL_BUY_THRESHOLD, settings.TARGET_VOLATILITY)
        settings.SIGNAL_BUY_THRESHOLD, settings.TARGET_VOLATILITY = threshold, vol
        try:
            returns = _segment_returns(panel, model, settings, train_end, n_bars)
        finally:
            settings.SIGNAL_BUY_THRESHOLD, settings.TARGET_VOLATILITY = orig
        if returns is None:
            continue
        columns.append(returns)
        labels.append(f"th={threshold} vol={vol}")
        logger.info(f"[Sweep] {labels[-1]:<20} Sharpe={sharpe_ratio(returns):.3f} "
                    f"MDD={max_drawdown_from_returns(returns) * 100:.2f}%")

    if len(columns) < 2:
        return {"pbo": float("nan"), "median_oos_rank": float("nan"), "n_combinations": 0}

    width = min(len(c) for c in columns)
    matrix = np.column_stack([c[:width] for c in columns])
    result = pbo_cscv(matrix, n_splits=8)
    result["labels"] = labels
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="전략 과적합 검증")
    parser.add_argument("--source", type=str, default="", help="로컬 시세 디렉토리")
    parser.add_argument("--days", type=int, default=1200, help="합성 데이터 봉 수")
    parser.add_argument("--symbols", type=int, default=12, help="사용할 종목 수")
    parser.add_argument("--n-groups", type=int, default=5, help="CPCV 그룹 수")
    parser.add_argument("--n-test", type=int, default=2, help="테스트로 쓸 그룹 수")
    parser.add_argument("--sweep", action="store_true", help="파라미터 탐색 PBO까지 계산")
    parser.add_argument("--trial-sharpes", type=str, default="",
                        help="개발 중 시험한 설정들의 Sharpe(쉼표 구분). 주면 DSR을 계산한다")
    parser.add_argument("--out", type=str, default="reports/validation.json")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_DIR, settings.LOG_LEVEL)

    symbols = settings.UNIVERSE[: args.symbols] if args.symbols else settings.UNIVERSE
    if args.source:
        panel = load_panel_from_dir(args.source, symbols)
        if not panel:
            logger.error(f"[Validate] {args.source} 에서 데이터를 찾지 못했습니다.")
            return 1
    else:
        panel = generate_synthetic_panel(
            symbols, n_bars=args.days, seed=settings.MOCK_SEED,
            sector_map={c: settings.sector_of(c) for c in symbols})
    panel = align_panel(panel, settings.MIN_HISTORY_BARS)

    trials: List[float] = []
    if args.trial_sharpes:
        trials = [float(x) for x in args.trial_sharpes.replace(" ", "").split(",") if x]
        logger.info(f"[Validate] 시험한 설정 {len(trials)}개를 선택편향 보정에 반영합니다")

    report = run_cpcv(panel, settings, args.n_groups, args.n_test, trials or None)
    if args.sweep:
        report.pbo = run_sweep(panel, settings)

    print()
    print(report.pretty())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    logger.info(f"[Validate] 리포트 저장 → {out}")

    key = "dsr" if report.n_configs > 0 else "psr"
    return 0 if report.dsr.get(key, 0.0) >= 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())
