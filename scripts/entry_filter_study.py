"""진입 필터 측정(11/15) — 채택 진입·종가 −3% 청산 위에서 과열 제외·시장 강도·발동 수 필터를 지금 서버 칸과 같은 설정으로 센다.

`combo_study.main()` 을 그대로 돌리되 5거래일 창·2자리x49%(실거래)로 두고, 끝나면 15:19 누적/하루 전체 거래대금 비율의
중앙값(실거래 15:19 판정의 분모 근사용)을 함께 찍는다. 노트북 전용(하루 전체 분봉 `data/cache/fullday`).

    PYTHONIOENCODING=utf-8 UNIVERSE_FILE=config/universe_10b.json python scripts/eod_exit_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.combo_study as cs  # noqa: E402

OUT = ROOT / "reports" / "combo_entry_index.json"
cs.COMP_BARS = 5
cs.MAX_GROSS = 0.98
sys.argv = ["combo_study.py", "--ma-filters", "--open-window", "4", "--slots", "2", "--out", str(OUT)]
rc = cs.main()
r = np.array(cs.RATIO_1519)
if len(r):
    print(f" 15:19 누적 / 하루 전체 거래대금: 중앙 {np.median(r):.3f} · 평균 {r.mean():.3f} · 10% {np.percentile(r, 10):.3f} · 90% {np.percentile(r, 90):.3f} (n={len(r):,})")
if OUT.exists():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    d["ratio_1519"] = {"median": float(np.median(r)), "mean": float(r.mean()), "n": int(len(r))} if len(r) else None
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f" {'규칙':<46}{'평균':>7}{'중앙':>7}{'손실':>7}{'≤-5%':>7}{'≤-10%':>7}{'≥+10%':>7}{'≥+20%':>7}{'q05':>7}{'q95':>7}")
    for x in d["rows"]:
        n = x["rule"]
        if "[진입필터]" in n:
            print(f" {n[:46]:<46}{x['mean']*100:+6.1f}%{x['median']*100:+6.1f}%{x['neg']*100:6.1f}%{x['loss5']*100:6.1f}%"
                  f"{x['loss10']*100:6.1f}%{x['up10']*100:6.1f}%{x['up20']*100:6.1f}%{x['q05']*100:+6.1f}%{x['q95']*100:+6.1f}%")
raise SystemExit(rc)
