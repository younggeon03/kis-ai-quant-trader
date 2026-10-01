"""AI 알파 모델 학습 스크립트.

    python scripts/train_model.py                    # 합성 데이터 1500봉으로 학습
    python scripts/train_model.py --days 2500        # 더 긴 기간
    python scripts/train_model.py --source data/cache  # 로컬 CSV/Parquet 사용

계좌 발급 후에는 `--source`에 KIS에서 받아 저장한 일봉 디렉토리를 지정하면
동일한 파이프라인으로 재학습된다(코드 수정 불필요).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import enable_utf8_console  # noqa: E402

enable_utf8_console()

import numpy as np  # noqa: E402

from backtest.data import generate_synthetic_panel, load_panel_from_dir  # noqa: E402
from config.settings import get_settings  # noqa: E402
from strategy.ai_model import AlphaModel  # noqa: E402
from strategy.features import build_panel_features  # noqa: E402
from utils.logger import logger, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 알파 모델 학습")
    parser.add_argument("--days", type=int, default=1500, help="합성 데이터 봉 수")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--source", type=str, default="", help="로컬 시세 디렉토리(미지정 시 합성)")
    parser.add_argument("--symbols", type=int, default=0, help="사용할 종목 수(0=전체)")
    parser.add_argument("--no-meta", action="store_true", help="2차 모델(메타라벨링) 생략")
    parser.add_argument(
        "--no-flow", action="store_true",
        help="수급 피처 제외(기술 피처만). 수급 단독 효과를 측정할 때 쓴다")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_DIR, settings.LOG_LEVEL)

    symbols = settings.UNIVERSE[: args.symbols] if args.symbols else settings.UNIVERSE
    flow_panel = {}
    if args.source:
        panel = load_panel_from_dir(args.source, symbols)
        # 같은 디렉토리의 flow/ 하위에서 수급 데이터를 함께 읽는다.
        #
        # `--no-flow`는 **측정용 스위치다.** 수급 피처의 단독 효과를 보려면 같은
        # 유니버스에서 on/off 두 번을 돌려 AUC를 비교해야 한다. 유니버스가 다른 두
        # 모델의 AUC는 라벨 모집단이 달라 비교할 수 없으므로, 축을 하나만 움직인다.
        #
        # 실전 배포 모델은 **반드시 수급을 켜고 학습한다.** 추론은 수급이 채워진
        # 상태로 들어오므로, 끈 채로 배포하면 train/serve 불일치가 된다
        # (`walkforward.py::train_on_bars`가 `flow_panel`을 빼먹어 수급 중요도가
        #  16종 전부 0으로 측정된 버그와 같은 종류다).
        if args.no_flow:
            logger.info("[Train] --no-flow — 수급 피처 제외(기술 피처만)")
        else:
            from strategy.flow_features import load_flow_dir

            flow_panel = load_flow_dir(Path(args.source) / "flow", symbols)
            if flow_panel:
                logger.info(f"[Train] 수급 데이터 {len(flow_panel)}종목 로드")
            else:
                logger.warning(
                    f"[Train] {args.source}/flow 에서 수급 데이터를 찾지 못했습니다 "
                    "— 기술 피처만으로 학습합니다")
        if not panel:
            logger.error(f"[Train] {args.source} 에서 데이터를 찾지 못했습니다.")
            return 1
        logger.info(f"[Train] 로컬 시세 {len(panel)}종목 로드")
    else:
        panel = generate_synthetic_panel(
            symbols,
            n_bars=args.days,
            seed=args.seed if args.seed is not None else settings.MOCK_SEED,
            sector_map={s: settings.sector_of(s) for s in symbols},
        )
        logger.info(f"[Train] 합성 시세 생성 {len(panel)}종목 × {args.days}봉")

    logger.info("[Train] 피처 생성 중...")
    featured = build_panel_features(panel, flow_panel)
    if not featured:
        logger.error("[Train] 피처 생성 실패(데이터 길이 부족)")
        return 1

    model = AlphaModel(settings)
    report = model.fit(featured)
    print()
    print(report.pretty())
    model.save()

    # --- 2차 모델(메타라벨링) ---
    if not args.no_meta:
        from strategy.meta_model import MetaModel

        logger.info("[Train] 2차 모델(메타라벨링) 학습 중...")
        primary: dict = {}
        for symbol, df in featured.items():
            cols = [c for c in model.columns if c in df.columns]
            X = np.zeros((len(df), len(model.columns)), dtype=np.float32)
            lookup = {c: i for i, c in enumerate(model.columns)}
            for c in cols:
                X[:, lookup[c]] = np.nan_to_num(df[c].to_numpy(dtype=float), nan=0.0)
            primary[symbol] = model.predict_proba(X)

        meta = MetaModel(settings)
        try:
            Xm, ym, tm, cm = MetaModel.build_dataset(
                featured, primary, model.columns,
                buy_threshold=settings.SIGNAL_BUY_THRESHOLD,
                neutral=model.base_rate)
            meta_report = meta.fit(Xm, ym, tm, cm, verbose=False)
            print()
            print(meta_report.pretty())
            meta.save()
        except ValueError as exc:
            logger.warning(f"[Train] 메타 모델 생략: {exc}")

    if report.cv_auc < 0.52:
        logger.warning(
            "[Train] CV AUC가 0.52 미만입니다. 라벨 지평(LABEL_HORIZON)이나 배리어 배수를 "
            "조정하거나 학습 기간을 늘려보세요."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
