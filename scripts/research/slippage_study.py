"""장 시작 매수의 실측 슬리피지(11/15) — 원장 체결가 vs 측정이 가정한 체결가(판정 분의 시가).

측정(`combo_study.py`)은 09:0N 분봉 **시가**에 사고, 판정 입력은 09:00~09:0(N-1) 분봉 거래대금 합이다.
실거래는 09:0N 뒤 복수시세(10호출·10~20초)로 판정하고 +2% 지정가로 추격한다. 둘 사이 차이를 원장과 토스 1분봉으로 잰다.

    ALL_PROXY=socks5://127.0.0.1:1080 PYTHONIOENCODING=utf-8 python scripts/research/slippage_study.py <ledger.jsonl> <out_dir>

토스 수집은 서버 IP(SSH 터널)로 나가야 한다 — 먼저 `curl --socks5-hostname 127.0.0.1:1080 https://api.ipify.org` 확인.
"""
from __future__ import annotations

import asyncio
import collections
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.toss_data import TossDataClient  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.collect_opening import fetch_opening  # noqa: E402
from utils.logger import enable_utf8_console  # noqa: E402


def read_ledger(path: Path):
    buys = collections.defaultdict(list)
    scans = {}
    for line in path.open(encoding="utf-8"):
        if '"fill"' not in line and '"opening_scan"' not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ts = str(r.get("ts", ""))
        day = ts[:10].replace("-", "")
        if day < "20261020":
            continue
        if r.get("event") == "opening_scan":
            scans[day] = (ts[11:19], r)
        elif r.get("event") == "fill" and r.get("side") == "BUY" and ts[11:13] == "09" and ts[14:16] < "20":
            buys[(day, r["symbol"])].append((ts[11:19], float(r["price"]), int(r["qty"])))
    return buys, scans


async def main() -> int:
    enable_utf8_console()
    ledger = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    buys, scans = read_ledger(ledger)
    s = get_settings()
    rows = []
    async with TossDataClient(s) as cli:
        for (day, sym), fills in sorted(buys.items()):
            f = out / f"{sym}_{day}.csv"
            if f.exists():
                df = pd.read_csv(f, dtype={"ts": str})
            else:
                df = await fetch_opening(cli, sym, day, "0915")
                df.to_csv(f, index=False)
            if df is None or df.empty:
                print(f"{day} {sym}: 분봉 없음")
                continue
            df = df.assign(hm=df["ts"].astype(str).str[8:12])
            scan_t, scan_rec = scans.get(day, (None, {}))
            n_win = int(scan_rec.get("window_min", 5))          # 그날 판정 창(분) — 측정의 ow
            judge_hm = f"09{n_win:02d}"
            bar = df[df["hm"] == judge_hm]
            q = sum(x[2] for x in fills)
            avg = sum(x[1] * x[2] for x in fills) / q
            ref_open = float(bar["open"].iloc[0]) if not bar.empty else float("nan")
            ref_close = float(bar["close"].iloc[0]) if not bar.empty else float("nan")
            prev_bar = df[df["hm"] == f"09{n_win - 1:02d}"]
            prev_close = float(prev_bar["close"].iloc[0]) if not prev_bar.empty else float("nan")
            # 판정 입력 치우침: 측정 = 09:00~09:0(N-1) 거래대금 합 / 실거래 ≈ 거기에 판정 분의 (경과초/60) 비율만큼 더해짐
            win = df[(df["hm"] >= "0900") & (df["hm"] < judge_hm)]
            tv_win = float((win["close"] * win["volume"]).sum())
            tv_judge = float((bar["close"] * bar["volume"]).sum()) if not bar.empty else 0.0
            scan_sec = (int(scan_t[3:5]) - n_win) * 60 + int(scan_t[6:8]) if scan_t else 0
            tv_live = tv_win + tv_judge * min(1.0, max(0.0, scan_sec / 60.0))
            first_fill = fills[0][0]
            rows.append({"day": day, "symbol": sym, "window": n_win, "scan": scan_t, "first_fill": first_fill,
                         "fill_avg": avg, "ref_open": ref_open, "ref_close": ref_close, "prev_close": prev_close,
                         "slip_vs_open": avg / ref_open - 1.0 if ref_open > 0 else float("nan"),
                         "slip_vs_prev_close": avg / prev_close - 1.0 if prev_close > 0 else float("nan"),
                         "tv_ratio_live_over_meas": tv_live / tv_win if tv_win > 0 else float("nan"),
                         "sec_after_judge": scan_sec})
    if not rows:
        print("체결 없음")
        return 1
    t = pd.DataFrame(rows)
    print()
    print(f" {'날':<10}{'종목':<8}{'창':>3}{'판정':>10}{'첫체결':>10}{'체결평균':>10}{'판정분 시가':>11}{'슬리피지':>9}{'전분 종가比':>10}{'입력치우침':>9}")
    for r in t.itertuples():
        print(f" {r.day:<10}{r.symbol:<8}{r.window:>3}{r.scan or '-':>10}{r.first_fill:>10}{r.fill_avg:>10,.0f}{r.ref_open:>11,.0f}"
              f"{r.slip_vs_open * 100:+8.2f}%{r.slip_vs_prev_close * 100:+9.2f}%{(r.tv_ratio_live_over_meas - 1) * 100:+8.1f}%")
    print(f" 슬리피지(판정 분 시가 대비): 평균 {t.slip_vs_open.mean() * 100:+.2f}% · 중앙 {t.slip_vs_open.median() * 100:+.2f}% · "
          f"최악 {t.slip_vs_open.max() * 100:+.2f}% (n={len(t)})")
    print(f" 판정 입력 치우침(실거래 누적 ÷ 측정 창 합 − 1): 평균 {(t.tv_ratio_live_over_meas.mean() - 1) * 100:+.1f}% · "
          f"판정 뒤 복수시세까지 {t.sec_after_judge.mean():.0f}초")
    t.to_json(out / "slippage.json", orient="records", force_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
