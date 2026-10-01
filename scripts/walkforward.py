"""워크포워드 재학습 시뮬레이션.

    python scripts/walkforward.py                            # 합성 데이터
    python scripts/walkforward.py --source data/cache        # 실데이터
    python scripts/walkforward.py --retrain-days 21          # 월 1회 재학습
    python scripts/walkforward.py --train-window 750         # 학습 창 3년

각 구간에서 모델은 그 시점까지의 과거만 본다. 이어 붙인 결과가 실전에 가장 가까운 숫자다.
일반 백테스트보다 낮게 나오는 것이 정상이며, 그 차이가 곧 "한 번 학습한 모델로 전 구간을
평가했을 때 생기는 낙관"이다.
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
from backtest.engine import align_panel  # noqa: E402
from backtest.walkforward import run_walkforward  # noqa: E402
from config.settings import get_settings  # noqa: E402
from utils.logger import logger, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="워크포워드 재학습 시뮬레이션")
    parser.add_argument("--source", type=str, default="", help="로컬 시세 디렉토리")
    parser.add_argument("--days", type=int, default=1500, help="합성 데이터 봉 수")
    parser.add_argument("--symbols", type=int, default=20, help="사용할 종목 수")
    parser.add_argument("--train-window", type=int, default=500, help="학습 창 길이(봉)")
    parser.add_argument("--retrain-days", type=int, default=63,
                        help="재학습 주기(봉). 63≈분기, 21≈월")
    parser.add_argument("--flow-dir", type=str, default="",
                        help="수급 디렉토리(기본 <source>/flow)")
    parser.add_argument("--no-flow", action="store_true", help="수급 피처 제외")
    parser.add_argument("--out", type=str, default="reports/walkforward.json")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_DIR, settings.LOG_LEVEL)

    symbols = settings.UNIVERSE[: args.symbols] if args.symbols else settings.UNIVERSE
    flow_panel = {}
    if args.source:
        panel = load_panel_from_dir(args.source, symbols)
        if not panel:
            logger.error(f"[WalkForward] {args.source} 에서 데이터를 찾지 못했습니다.")
            return 1
        # 수급은 **학습과 평가에 같은 패널**을 넘겨야 한다. 한쪽만 쓰면 학습/추론
        # 입력 분포가 달라진다(backtest/walkforward.py::train_on_bars 참고).
        if not args.no_flow:
            from strategy.flow_features import load_flow_dir
            flow_dir = Path(args.flow_dir) if args.flow_dir                 else Path(args.source) / "flow"
            flow_panel = load_flow_dir(flow_dir, symbols)
            logger.info(f"[WalkForward] 수급 {len(flow_panel)}종목 로드 ({flow_dir})"
                        if flow_panel else
                        f"[WalkForward] 수급 없음 ({flow_dir}) — 가격 피처만 사용")
    else:
        panel = generate_synthetic_panel(
            symbols, n_bars=args.days, seed=settings.MOCK_SEED,
            sector_map={c: settings.sector_of(c) for c in symbols})
    panel = align_panel(panel, settings.MIN_HISTORY_BARS)

    n_bars = min(len(df) for df in panel.values())
    expected = max(0, (n_bars - max(args.train_window, 120)) // args.retrain_days)
    logger.info(f"[WalkForward] {len(panel)}종목 · {n_bars}봉 · "
                f"학습창 {args.train_window} · 재학습 주기 {args.retrain_days} "
                f"→ 약 {expected}회 재학습")

    try:
        result = run_walkforward(panel, settings,
                                 train_window=args.train_window,
                                 step=args.retrain_days,
                                 flow_panel=flow_panel)
    except ValueError as exc:
        logger.error(f"[WalkForward] {exc}")
        return 1

    if not result.segments:
        logger.error("[WalkForward] 평가된 구간이 없습니다.")
        return 1

    print()
    print(result.segment_table())
    print()
    print(result.summary())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    logger.info(f"[WalkForward] 리포트 저장 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
