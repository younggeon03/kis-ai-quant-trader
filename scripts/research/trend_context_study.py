"""숲 보기 측정(11/16) — 채택 진입(4분x3·전일↑·VWAP 위) · 종가 −3% 청산 위에서 일봉 이평 맥락을 진입 필터와 청산에 넣어 본다.

    PYTHONIOENCODING=utf-8 UNIVERSE_FILE=config/universe_10b.json python scripts/research/trend_context_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.research.combo_study as cs  # noqa: E402

OUT = ROOT / "reports" / "combo_trend_index.json"
cs.COMP_BARS = 5
cs.MAX_GROSS = 0.98
sys.argv = ["combo_study.py", "--ma-filters", "--open-window", "4", "--slots", "2", "--out", str(OUT)]
rc = cs.main()
if OUT.exists():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    print()
    print(f" {'규칙(5일 창, 2자리x49%)':<52}{'사건':>7}{'평균':>7}{'중앙':>7}{'손실':>7}{'≤-5%':>7}{'≤-10%':>7}{'≥+10%':>7}{'≥+20%':>7}")
    for x in d["rows"]:
        n = x["rule"]
        if n.startswith("[진입필터] 기준") or n.startswith("[숲필터]") or (
                "채택진입] 청산" in n and ("이평 → 익일" in n or "이평만" in n or "−3% → 익일" in n)):
            print(f" {n[:52]:<52}{x['events']:>7,}{x['mean']*100:+6.1f}%{x['median']*100:+6.1f}%{x['neg']*100:6.1f}%"
                  f"{x['loss5']*100:6.1f}%{x['loss10']*100:6.1f}%{x['up10']*100:6.1f}%{x['up20']*100:6.1f}%")
raise SystemExit(rc)
