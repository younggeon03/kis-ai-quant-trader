"""관제 화면의 웹 공개판을 만든다.

실행 중인 엔진에서 상태를 떠오고, 백테스트 리포트와 모델 리포트를 합쳐
`docs/dashboard_web.html` 한 파일로 굽는다(외부 요청 없이 그대로 열린다).

    python scripts/build_web_dashboard.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _curve(path: Path, n: int = 180) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    step = max(1, len(rows) // n)
    return [
        {"t": rows[i]["ts"], "e": round(float(rows[i]["equity"])),
         "d": round(float(rows[i]["drawdown"]) * 100, 3)}
        for i in range(0, len(rows), step)
    ]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build(url: str = "http://127.0.0.1:8500/api/state") -> Path:
    data: dict = {"captured_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            st = json.load(resp)
        data["live"] = {
            "portfolio": st["portfolio"],
            "risk": st["risk"],
            "strategy": {
                "regime": st["strategy"]["regime"],
                "top_signals": st["strategy"]["top_signals"][:8],
                "backend": st["strategy"]["backend"],
            },
            "system": {k: st["system"].get(k) for k in (
                "mode", "session", "loops", "errors", "universe", "open_orders",
                "progress", "elapsed_sec", "mdd_soft", "mdd_hard")},
            "history": st["history"],
            "recent_fills": st["recent_fills"][-8:],
        }
    except Exception as exc:  # 엔진이 안 떠 있으면 직전 스냅샷을 재사용한다
        print(f"[build] 엔진 상태를 가져오지 못했습니다({exc}). 이전 스냅샷을 씁니다.")
        prev = _load_json(ROOT / "docs" / "snapshot.json")
        data["live"] = prev.get("live", {})

    data["backtest"] = _load_json(ROOT / "reports" / "metrics_ensemble.json")
    data["backtest_nomodel"] = _load_json(ROOT / "reports" / "metrics_nomodel.json")
    data["model"] = _load_json(ROOT / "models" / "alpha_gbdt_report.json")
    for key, name in (("curve", "equity_curve_ensemble.csv"),
                      ("curve_nomodel", "equity_curve_nomodel.csv")):
        path = ROOT / "reports" / name
        data[key] = _curve(path) if path.exists() else []

    (ROOT / "docs" / "snapshot.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")

    template = (ROOT / "docs" / "dashboard_web.template.html").read_text(encoding="utf-8")
    out = ROOT / "docs" / "dashboard_web.html"
    out.write_text(template.replace("/*__DATA__*/{}", json.dumps(data, ensure_ascii=False)),
                   encoding="utf-8")
    print(f"[build] {out} ({out.stat().st_size:,} bytes)")
    return out


if __name__ == "__main__":
    build()
