"""후보 3개를 공통 평가기(day_eval.evaluate)로 재확인 — 캐시된 피처·라벨(같은 함수로 만든 값)을 먼저 올린다."""
import pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from scripts.research import day_eval as de
from scripts.research.agents import day_ml as dm
f, l = pickle.load(open(ROOT / "reports/agents/day_ml_cache.pkl", "rb")); dm._FEAT.update(f); dm._LAB.update(l)
de.evaluate(de.baseline_engine, "기준: 엔진 09:04 ATR10 1위")
for n in ("candidate_1", "candidate_2", "candidate_3"):
    de.evaluate(getattr(dm, n), n)
