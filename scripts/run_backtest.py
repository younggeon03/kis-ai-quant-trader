"""백테스트 실행 스크립트.

    python scripts/run_backtest.py                 # 합성 데이터 1000봉
    python scripts/run_backtest.py --days 2000 --capital 100000000
    python scripts/run_backtest.py --no-model      # 기술적 팩터만(모델 기여도 비교)
    python scripts/run_backtest.py --source data/cache

결과: 콘솔 성과 리포트 + reports/ 디렉토리에 자산곡선 CSV / 체결내역 CSV / PNG 저장.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()

from backtest.data import generate_synthetic_panel, load_panel_from_dir  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402
from backtest.metrics import plot_report  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.ai_model import EnsembleAlphaStrategy  # noqa: E402
from strategy.risk_manager import RiskManager  # noqa: E402
from utils.logger import logger, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="KIS AI Quant 백테스트")
    parser.add_argument("--days", type=int, default=1000, help="합성 데이터 봉 수")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--capital", type=float, default=None, help="초기 자본")
    parser.add_argument("--source", type=str, default="", help="로컬 시세 디렉토리")
    parser.add_argument("--symbols", type=int, default=0, help="사용할 종목 수(0=전체)")
    parser.add_argument("--no-model", action="store_true", help="AI 모델 없이 기술적 팩터만 사용")
    parser.add_argument("--no-flow", action="store_true", help="수급 피처 제외(기여도 비교용)")
    parser.add_argument("--out", type=str, default="reports", help="리포트 출력 디렉토리")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_DIR, settings.LOG_LEVEL)

    symbols = settings.UNIVERSE[: args.symbols] if args.symbols else settings.UNIVERSE
    flow_panel = {}
    if args.source:
        panel = load_panel_from_dir(args.source, symbols)
        # 같은 디렉토리의 flow/ 하위에서 수급 데이터를 함께 읽는다
        from strategy.flow_features import load_flow_dir

        flow_panel = load_flow_dir(Path(args.source) / "flow", symbols)
        if flow_panel:
            logger.info(f"[Backtest] 수급 데이터 {len(flow_panel)}종목 로드")
        if not panel:
            logger.error(f"[Backtest] {args.source} 에서 데이터를 찾지 못했습니다.")
            return 1
    else:
        panel = generate_synthetic_panel(
            symbols,
            n_bars=args.days,
            seed=args.seed if args.seed is not None else settings.MOCK_SEED,
            sector_map={s: settings.sector_of(s) for s in symbols},
        )
    logger.info(f"[Backtest] 데이터 {len(panel)}종목 × {min(len(d) for d in panel.values())}봉")

    strategy = EnsembleAlphaStrategy(settings)
    if not args.no_model:
        strategy.load_model()
    else:
        logger.info("[Backtest] --no-model: 기술적 팩터 + 횡단면 랭크만 사용")

    if args.no_flow:
        flow_panel = {}
        logger.info("[Backtest] --no-flow: 수급 피처를 쓰지 않습니다")

    # in-sample 백테스트 경고 — 이번 작업에서 실제로 밟은 함정이다.
    # 학습 구간과 평가 구간이 겹치면 모델이 미래를 이미 본 것이라 Sharpe가 무의미하게 높다.
    train_lo, train_hi = getattr(strategy.model, "train_range", ("", ""))
    if train_lo and panel:
        first = min(str(df["ts"].iloc[0]) for df in panel.values())
        last = max(str(df["ts"].iloc[-1]) for df in panel.values())
        if not (last < train_lo or first > train_hi):
            logger.warning("=" * 78)
            logger.warning(f"  [경고] 학습 구간({train_lo}~{train_hi})과 "
                           f"평가 구간({first}~{last})이 겹칩니다.")
            logger.warning("  이 백테스트는 in-sample이며 성과가 크게 부풀려집니다.")
            logger.warning("  실제 성과는 scripts/walkforward.py 또는 "
                           "scripts/validate_strategy.py 로 측정하세요.")
            logger.warning("=" * 78)

    engine = BacktestEngine(
        panel,
        strategy=strategy,
        risk_manager=RiskManager(settings),
        settings=settings,
        initial_capital=args.capital,
        flow_panel=flow_panel,
    )
    result = engine.run(verbose=not args.quiet)

    print()
    print(result.summary())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "nomodel" if args.no_model else "ensemble"
    result.curve.to_csv(out / f"equity_curve_{tag}.csv", index=False, encoding="utf-8-sig")
    trades = result.trades_frame()
    if not trades.empty:
        trades.to_csv(out / f"trades_{tag}.csv", index=False, encoding="utf-8-sig")
    (out / f"metrics_{tag}.json").write_text(
        json.dumps(result.metrics.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    png = plot_report(result.curve, str(out / f"equity_{tag}.png"),
                      title=f"KIS AI Quant — {tag} (Sharpe {result.metrics.sharpe:.2f}, "
                            f"MDD {result.metrics.max_drawdown * 100:.1f}%)")
    logger.success(f"[Backtest] 리포트 저장 → {out.resolve()}")
    if png:
        logger.info(f"[Backtest] 차트 → {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
