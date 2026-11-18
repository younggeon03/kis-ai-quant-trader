"""실시간 관제 화면 (FastAPI).

매매 루프와 **같은 이벤트 루프**에서 asyncio 태스크로 함께 돈다(별도 프로세스 불필요).
상태는 엔진이 주입한 콜백으로 조회하므로, 화면이 죽어도 매매는 영향받지 않는다.

  GET /              관제 화면
  GET /architecture  시스템 구조 다이어그램(docs/architecture.html)
  GET /health        헬스체크 (프로세스 감시용)
  GET /api/state     전체 상태 JSON
  GET /metrics       Prometheus 텍스트 포맷

화면 규칙
  · 등락 색은 국내 시장 관습을 따른다 — 상승 빨강, 하락 파랑.
  · 초록은 가격이 아니라 '시스템이 살아 있다'는 신호로만 쓴다.
  · 화면의 주인공은 낙폭 게이지다. 대회 심사의 핵심이 MDD 방어이기 때문이다.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from utils.logger import logger

StateProvider = Callable[[], Dict[str, Any]]

_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>KIS AI Quant Trader 관제</title>
<style>
  :root{
    --ink:#0B0F14; --surface:#101822; --raise:#16202C; --rule:#1F2D3D;
    --text:#E4EBF3; --dim:#7D90A6; --faint:#4C5D71;
    --rise:#F04452;        /* 상승 — 국내 관습 */
    --fall:#3182F6;        /* 하락 */
    --caution:#F5A623;     /* 경계 구간 */
    --live:#00C48C;        /* 시스템 생존 신호 (가격 아님) */
    --halt:#FF4D4F;
    --sp:clamp(14px,1.6vw,22px);
  }
  *{box-sizing:border-box;}
  html,body{margin:0;background:var(--ink);color:var(--text);}
  body{
    font-family:'Pretendard','Apple SD Gothic Neo','Malgun Gothic',system-ui,-apple-system,sans-serif;
    font-feature-settings:"tnum" 1;
    font-variant-numeric:tabular-nums;
    -webkit-font-smoothing:antialiased;
    line-height:1.45;
  }
  .wrap{max-width:1560px;margin:0 auto;padding:0 var(--sp) 48px;}

  /* ─────────────── 헤더 ─────────────── */
  header{
    display:flex;align-items:center;gap:18px;flex-wrap:wrap;
    padding:16px 0 14px;border-bottom:1px solid var(--rule);
  }
  .mark{display:flex;align-items:baseline;gap:10px;}
  .mark b{font-size:15px;font-weight:700;letter-spacing:-.01em;}
  .mark span{font-size:12px;color:var(--faint);}
  .pulse{width:7px;height:7px;border-radius:50%;background:var(--live);
         box-shadow:0 0 0 0 rgba(0,196,140,.55);animation:beat 2.4s infinite;}
  .pulse.off{background:var(--halt);animation:none;}
  @keyframes beat{70%{box-shadow:0 0 0 7px rgba(0,196,140,0)}100%{box-shadow:0 0 0 0 rgba(0,196,140,0)}}
  @media (prefers-reduced-motion:reduce){.pulse{animation:none}}
  .meta{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--dim);}
  .meta>span{white-space:nowrap;}
  .meta b{color:var(--text);font-weight:600;}
  .spacer{flex:1;}
  button{
    font:inherit;font-size:12.5px;color:var(--dim);background:transparent;
    border:1px solid var(--rule);border-radius:2px;padding:5px 11px;cursor:pointer;
  }
  button:hover{color:var(--text);border-color:var(--faint);}
  button:focus-visible{outline:2px solid var(--live);outline-offset:2px;}
  button[aria-pressed="true"]{color:var(--caution);border-color:var(--caution);}
  a.link{color:var(--dim);font-size:12.5px;text-decoration:none;border-bottom:1px solid var(--rule);padding-bottom:2px;}
  a.link:hover{color:var(--text);}

  /* ─────────────── 히어로: 자산 ─────────────── */
  .hero{display:grid;grid-template-columns:minmax(260px,320px) 1fr;gap:var(--sp);
        padding:22px 0 20px;border-bottom:1px solid var(--rule);align-items:center;}
  .equity-label{font-size:12.5px;color:var(--dim);margin-bottom:2px;}
  .equity{font-size:clamp(34px,4.4vw,52px);font-weight:700;letter-spacing:-.025em;line-height:1.05;}
  .equity small{font-size:.42em;font-weight:500;color:var(--dim);margin-left:6px;letter-spacing:0;}
  .delta{display:flex;align-items:baseline;gap:18px;margin-top:12px;font-size:14px;flex-wrap:wrap;}
  .delta>span{white-space:nowrap;}
  .delta i{font-style:normal;color:var(--dim);font-size:12px;margin-right:5px;}
  .delta.note{margin-top:8px;font-size:12.5px;color:var(--dim);}
  .delta.note b{color:var(--fg);font-weight:600;}
  .chart{position:relative;}
  .chart svg{display:block;width:100%;height:150px;}
  .chart .empty{position:absolute;inset:0;display:grid;place-items:center;color:var(--faint);font-size:13px;}
  .chart figcaption{display:flex;justify-content:space-between;font-size:11.5px;color:var(--faint);margin-top:6px;}

  /* ─────────────── 낙폭 게이지 (화면의 주인공) ─────────────── */
  .gauge{padding:20px 0 22px;border-bottom:1px solid var(--rule);}
  .gauge-head{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap;}
  .state{font-size:13px;font-weight:700;padding:4px 12px;border-radius:2px;letter-spacing:.02em;}
  .state.normal{background:rgba(0,196,140,.12);color:var(--live);}
  /* 대회 영역 — 기간·국면·초기화 권고·최소 요건 */
  .comp{margin:14px 0;padding:14px 16px;border:1px solid var(--line);border-radius:12px;
        background:var(--panel);display:grid;gap:12px;}
  .comp-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
  .comp-phase{font-weight:700;padding:3px 10px;border-radius:999px;font-size:12px;}
  .comp-phase.p1{background:rgba(255,59,59,.14);color:var(--up);}
  .comp-phase.p2{background:rgba(0,196,140,.12);color:var(--live);}
  .comp-note{color:var(--dim);font-size:12px;}
  .comp-left{margin-left:auto;font-variant-numeric:tabular-nums;font-size:12px;color:var(--dim);}
  /* 초기화 권고: 되돌릴 수 없는 1회성 행위라 눈에 띄어야 한다 */
  .reset{padding:10px 12px;border-radius:10px;font-size:13px;line-height:1.5;}
  .reset.act{background:rgba(255,59,59,.12);border:1px solid rgba(255,59,59,.35);color:var(--up);}
  .reset.hold{background:rgba(0,0,0,.14);border:1px solid var(--line);color:var(--dim);}
  .reqs{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;}
  .req{border:1px solid var(--line);border-radius:10px;padding:8px 10px;}
  .req-top{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);}
  .req-val{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums;margin:2px 0 6px;}
  .req-bar{height:4px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;}
  .req-bar i{display:block;height:100%;border-radius:999px;background:var(--dim);}
  .req.ok .req-bar i{background:var(--live);}
  .req.ok .req-val{color:var(--live);}
  .req.bad .req-bar i{background:var(--up);}
  .req.bad .req-val{color:var(--up);}
  .state.derisk{background:rgba(245,166,35,.14);color:var(--caution);}
  .state.halt{background:rgba(255,77,79,.16);color:var(--halt);}
  .gauge-note{font-size:13px;color:var(--dim);}
  .track{position:relative;height:40px;}
  .zones{position:absolute;inset:0 0 18px;display:flex;border-radius:2px;overflow:hidden;}
  .zones div{height:100%;}
  .z1{background:rgba(0,196,140,.16);} .z2{background:rgba(245,166,35,.2);} .z3{background:rgba(255,77,79,.24);}
  .fill{position:absolute;left:0;top:0;bottom:18px;background:rgba(228,235,243,.10);
        border-right:2px solid var(--text);transition:width .45s cubic-bezier(.4,0,.2,1);}
  .fill.warn{border-right-color:var(--caution);} .fill.bad{border-right-color:var(--halt);}
  .ticks{position:absolute;inset:auto 0 0;height:18px;}
  .tick{position:absolute;top:0;transform:translateX(-50%);font-size:11px;color:var(--faint);white-space:nowrap;}
  .tick b{color:var(--dim);font-weight:500;}
  .tick::before{content:"";position:absolute;left:50%;top:-22px;width:1px;height:22px;background:var(--rule);}
  .readout{position:absolute;top:-2px;transform:translateX(-50%);font-size:12.5px;font-weight:700;
           white-space:nowrap;transition:left .45s cubic-bezier(.4,0,.2,1);}

  /* ─────────────── 패널 그리드 ─────────────── */
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:0 var(--sp);}
  .cols.thirds{grid-template-columns:1.1fr .95fr .95fr;}
  section{padding:20px 0 6px;border-bottom:1px solid var(--rule);min-width:0;}
  h2{font-size:12.5px;font-weight:600;color:var(--dim);margin:0 0 12px;display:flex;
     align-items:center;gap:8px;}
  h2 em{font-style:normal;color:var(--faint);font-weight:400;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  th{font-size:11.5px;font-weight:500;color:var(--faint);text-align:right;padding:0 0 7px;}
  th:first-child{text-align:left;}
  td{padding:7px 0;border-top:1px solid rgba(31,45,61,.55);text-align:right;}
  td:first-child{text-align:left;font-weight:600;}
  tbody tr:hover td{background:rgba(255,255,255,.017);}
  .up{color:var(--rise);} .down{color:var(--fall);} .flat{color:var(--dim);}
  .nil{color:var(--faint);font-size:12.5px;padding:18px 0;text-align:left;}
  .buy{color:var(--live);font-weight:600;}
  .sub{color:var(--faint);font-size:11.5px;font-weight:400;margin-left:5px;}

  /* 막대 시각화 */
  .bar{position:relative;height:4px;width:92px;margin-left:auto;margin-top:5px;
       background:rgba(255,255,255,.055);border-radius:2px;overflow:hidden;}
  .bar i{position:absolute;top:0;bottom:0;display:block;border-radius:2px;}
  .pnlbar{position:relative;height:4px;width:92px;margin-left:auto;margin-top:5px;
          background:rgba(255,255,255,.055);border-radius:2px;}
  .pnlbar i{position:absolute;top:0;bottom:0;left:50%;border-radius:2px;}
  .axis{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--faint);}

  /* 유량 스파크바 */
  .rps{display:flex;align-items:flex-end;gap:2px;height:52px;position:relative;margin-bottom:8px;}
  .rps{justify-content:flex-end;}
  .rps i{flex:0 0 7px;background:var(--fall);opacity:.7;min-height:1px;border-radius:1px 1px 0 0;}
  .rps i.over{background:var(--halt);opacity:1;}
  .limit{position:absolute;left:0;right:0;border-top:1px dashed var(--caution);}
  .limit span{position:absolute;right:0;top:-16px;font-size:10.5px;color:var(--caution);}

  /* 진행 */
  .prog{height:4px;background:rgba(255,255,255,.055);border-radius:2px;overflow:hidden;margin:10px 0 6px;}
  .prog i{display:block;height:100%;background:var(--live);opacity:.8;transition:width .5s ease;}
  dl{display:grid;grid-template-columns:auto 1fr;gap:7px 16px;margin:0;font-size:13px;}
  dt{color:var(--dim);font-size:12.5px;} dd{margin:0;text-align:right;}

  /* 체결 테이프 */
  .tape td{font-family:ui-monospace,'SF Mono','Cascadia Mono','D2Coding',monospace;font-size:12.5px;}
  .tape td:first-child{color:var(--faint);}
  .why{color:var(--dim);font-size:11.5px;}

  /* 지수 종목 순위판 */
  .rank-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:12px;}
  .rank-head h2{margin:0;}
  .rank-head label{font-size:12.5px;color:var(--dim);display:flex;align-items:center;gap:6px;cursor:pointer;}
  .rank-head select{font:inherit;font-size:12.5px;color:var(--text);background:var(--surface);
                    border:1px solid var(--rule);border-radius:2px;padding:3px 6px;}
  .tag{font-size:10.5px;color:var(--faint);border:1px solid var(--rule);border-radius:2px;padding:0 4px;margin-left:5px;font-weight:400;}
  .tag.held{color:var(--live);border-color:var(--live);}
  .nobuy{color:var(--caution);font-weight:700;}
  .toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:var(--halt);
         color:#fff;font-size:13px;padding:9px 18px;border-radius:2px;display:none;}
  @media(max-width:980px){
    .cols,.cols.thirds,.hero{grid-template-columns:1fr;}
    .hero{gap:20px;}
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="mark"><span class="pulse" id="pulse"></span><b>KIS AI Quant Trader</b><span>관제</span></div>
    <div class="meta" id="meta"></div>
    <div class="spacer"></div>
    <a class="link" href="/architecture" target="_blank" rel="noopener">시스템 구조</a>
    <button id="hold" aria-pressed="false">갱신 멈춤</button>
  </header>

  <div class="comp" id="comp" hidden>
    <div class="comp-head">
      <span class="comp-phase" id="compPhase">—</span>
      <span class="comp-note" id="compLabel"></span>
      <span class="comp-left" id="compLeft"></span>
    </div>
    <div class="reset" id="compReset" hidden></div>
    <div class="reqs" id="compReqs"></div>
  </div>

  <div class="hero">
    <div>
      <div class="equity-label">총자산</div>
      <div class="equity" id="equity">—</div>
      <div class="delta" id="delta"></div>
      <div class="delta note" id="brokerEq"></div>
    </div>
    <figure class="chart" style="margin:0">
      <svg id="spark" viewBox="0 0 1000 150" preserveAspectRatio="none" aria-label="자산 추이"></svg>
      <div class="empty" id="sparkEmpty">자산 곡선은 첫 루프가 돌면 그려집니다</div>
      <figcaption><span id="sparkFrom"></span><span id="sparkTo"></span></figcaption>
    </figure>
  </div>

  <div class="gauge">
    <div class="gauge-head">
      <span class="state normal" id="state">—</span>
      <span class="gauge-note" id="stateNote"></span>
    </div>
    <div class="track" id="track">
      <div class="zones"><div class="z1" id="z1"></div><div class="z2" id="z2"></div><div class="z3" id="z3"></div></div>
      <div class="fill" id="fill" style="width:0%"></div>
      <div class="readout" id="readout" style="left:0%">0.00%</div>
      <div class="ticks" id="ticks"></div>
    </div>
  </div>

  <div class="cols">
    <section><h2>보유 포지션 <em id="posCount"></em></h2><table id="positions"></table></section>
    <section><h2>장 시작 판정 <em id="openNote"></em></h2><table id="openTbl"></table>
      <h2 style="margin-top:18px">장중 발동 <em id="slideNote"></em></h2><table id="slideTbl"></table></section>
  </div>

  <section>
    <div class="rank-head">
      <h2>지수 종목 순위 <em id="rankNote"></em></h2>
      <label><input type="checkbox" id="rankSmall"> 시총 1조 이하만</label>
      <label>지수 <select id="rankIdx"><option value="">전체</option><option>KOSPI200</option><option>KOSDAQ150</option></select></label>
      <label>표시 <select id="rankTop"><option>10</option><option selected>20</option><option>30</option></select></label>
    </div>
    <h2>관심 종목 <em id="pickNote"></em></h2>
    <table id="pickTbl"></table>
    <h2>거래대금 · 등락률 동시 상위 <em>두 순위 중 나쁜 쪽이 작은 순 10종목 · 지수 전체 · 상승 종목만</em></h2>
    <table id="crossTbl"></table>
  </section>
  <div class="cols">
    <section><h2>거래대금 상위 <em>장 시작부터 누적</em></h2><table id="tvTbl"></table></section>
    <section><h2>등락률 상위 <em>전일 종가 대비</em></h2><table id="chgTbl"></table></section>
  </div>

  <div class="cols thirds">
    <section><h2>요청 유량 <em id="rpsNote"></em></h2>
      <div class="rps" id="rps"><div class="limit" id="limitLine"><span></span></div></div>
      <dl id="rpsDl"></dl>
    </section>
    <section><h2>진행 <em id="progLabel"></em></h2>
      <div class="prog"><i id="progBar" style="width:0%"></i></div>
      <dl id="progDl"></dl>
    </section>
    <section><h2>리스크 판정 기록</h2><dl id="riskDl"></dl></section>
  </div>

  <section><h2>체결 <em id="fillCount"></em></h2><table class="tape" id="fills"></table></section>
</div>
<div class="toast" id="toast">엔진에 연결할 수 없습니다</div>

<script>
const $ = id => document.getElementById(id);
const num = (n,d=0) => (n===null||n===undefined||isNaN(n)) ? '—'
  : Number(n).toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d});
const signed = (n,d=2) => (n===null||n===undefined||isNaN(n)) ? '—'
  : (n>0?'+':'') + Number(n).toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d});
// 국내 관습: 오르면 빨강, 내리면 파랑
const tone = v => v > 0 ? 'up' : (v < 0 ? 'down' : 'flat');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

let holding = false;
$('hold').addEventListener('click', e => {
  holding = !holding;
  e.target.setAttribute('aria-pressed', String(holding));
  e.target.textContent = holding ? '갱신 재개' : '갱신 멈춤';
  if (!holding) refresh();
});

function rows(headers, data, empty) {
  if (!data.length) return `<tr><td class="nil" colspan="${headers.length}">${empty}</td></tr>`;
  return '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>'
       + data.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('');
}

function drawSpark(history) {
  const svg = $('spark');
  if (!history || history.length < 2) { $('sparkEmpty').style.display = 'grid'; svg.innerHTML=''; return; }
  $('sparkEmpty').style.display = 'none';
  const W = 1000, H = 150, pad = 8;
  const eq = history.map(h => h.equity);
  let lo = Math.min(...eq), hi = Math.max(...eq);
  const head = ((hi - lo) || Math.max(1, hi * 0.0004)) * 0.35;  // 위아래 여백
  lo -= head; hi += head;
  const span = hi - lo;
  const x = i => (i / (history.length - 1)) * W;
  const y = v => pad + (1 - (v - lo) / span) * (H - pad * 2 - 26);
  const line = history.map((h,i) => `${i?'L':'M'}${x(i).toFixed(1)},${y(h.equity).toFixed(1)}`).join('');
  const area = `${line}L${W},${H-26}L0,${H-26}Z`;
  const last = history[history.length-1], first = history[0];
  const rising = last.equity >= first.equity;
  const col = rising ? 'var(--rise)' : 'var(--fall)';
  // 낙폭 리본: 자산 곡선 아래 얇은 띠로 위험을 같은 시간축에 붙여 읽는다
  const maxDd = Math.max(0.5, ...history.map(h => h.dd));
  const dd = history.map((h,i) => `${i?'L':'M'}${x(i).toFixed(1)},${(H-24+(h.dd/maxDd)*22).toFixed(1)}`).join('');
  svg.innerHTML = `
    <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${col}" stop-opacity=".13"/>
      <stop offset="100%" stop-color="${col}" stop-opacity="0"/>
    </linearGradient></defs>
    <path d="${area}" fill="url(#g)"/>
    <path d="${line}" fill="none" stroke="${col}" stroke-width="1.6" vector-effect="non-scaling-stroke"
          stroke-linejoin="round"/>
    <line x1="0" y1="${H-26}" x2="${W}" y2="${H-26}" stroke="var(--rule)" stroke-width="1"
          vector-effect="non-scaling-stroke"/>
    <path d="${dd}L${W},${H-24}L0,${H-24}Z" fill="var(--halt)" fill-opacity=".28"/>
    <circle cx="${x(history.length-1)}" cy="${y(last.equity)}" r="3" fill="${col}"/>`;
  $('sparkFrom').textContent = first.t + ' · ' + num(lo) + '원';
  $('sparkTo').textContent = '최대낙폭 ' + num(maxDd,2) + '% · ' + num(hi) + '원';
}

function drawGauge(risk, limits) {
  const soft = limits.soft, hard = limits.hard;
  const max = Math.max(hard * 1.35, (risk.drawdown_pct || 0) * 1.15, hard + 3);
  const pct = v => Math.min(100, Math.max(0, v / max * 100));
  $('z1').style.width = pct(soft) + '%';
  $('z2').style.width = (pct(hard) - pct(soft)) + '%';
  $('z3').style.width = (100 - pct(hard)) + '%';

  const dd = risk.drawdown_pct || 0;
  const fill = $('fill');
  fill.style.width = pct(dd) + '%';
  fill.className = 'fill' + (dd >= hard ? ' bad' : dd >= soft ? ' warn' : '');
  const read = $('readout');
  read.style.left = Math.min(94, Math.max(6, pct(dd))) + '%';
  read.textContent = num(dd, 2) + '%';
  read.className = 'readout ' + (dd >= hard ? 'up' : dd >= soft ? 'flat' : '');
  read.style.color = dd >= hard ? 'var(--halt)' : dd >= soft ? 'var(--caution)' : 'var(--text)';

  $('ticks').innerHTML =
    `<span class="tick" style="left:${pct(soft)}%"><b>${num(soft,0)}%</b> 진입 차단</span>` +
    `<span class="tick" style="left:${pct(hard)}%"><b>${num(hard,0)}%</b> 전량 청산</span>`;

  const st = (risk.state || 'NORMAL');
  $('state').className = 'state ' + st.toLowerCase();
  $('state').textContent = {NORMAL:'정상 매매', DERISK:'익스포저 축소', HALT:'매매 중단'}[st] || st;
  const gap = Math.max(0, soft - dd);
  $('stateNote').textContent = st === 'HALT'
    ? '킬스위치 발동 — 전량 청산 후 쿨다운' + (risk.cooldown_until ? ' (해제 ' + String(risk.cooldown_until).slice(11,16) + ')' : '')
    : st === 'DERISK'
      ? '신규 진입을 막고 익스포저를 절반으로 줄이는 중'
      : `진입 차단선까지 ${num(gap,2)}%p 남음 · 거부 ${num(risk.blocked)}건`;
}

function drawRps(hist, rl) {
  const limit = rl.base_rate || 2;
  const cap = Math.max(limit * 1.6, ...(hist.length ? hist : [0]), 1);
  const bars = (hist.length ? hist : [0]).slice(-45);
  $('rps').innerHTML = `<div class="limit" style="bottom:${(limit/cap*100).toFixed(1)}%"><span>한도 ${num(limit,1)}/s</span></div>`
    // 1초 측정 창 경계 때문에 한도를 아주 살짝 넘는 값은 정상이다. 실제 초과만 표시한다.
    + bars.map(v => `<i class="${v > limit * 1.25 ? 'over' : ''}" style="height:${Math.max(1, v/cap*100).toFixed(1)}%"></i>`).join('');
  $('rpsNote').textContent = rl.penalized ? '감축 중' : '';
  $('rpsDl').innerHTML = [
    ['현재', num(rl.recent_rps,1) + ' /s'],
    ['관측 피크', num(rl.peak_rps,1) + ' /s'],
    ['대기열', num(rl.queue_depth) + '건 · 평균 ' + num(rl.avg_wait_ms,0) + 'ms'],
    ['서버 거부', `<span class="${rl.server_rejects>0?'up':''}">${num(rl.server_rejects)}건</span>`],
  ].map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

function renderCompetition(c){
  const box = $('comp');
  if (!c || !c.active) { box.hidden = true; return; }
  box.hidden = false;

  const p1 = c.phase === 'PHASE1';
  const badge = $('compPhase');
  badge.textContent = p1 ? '1단계 · 꼬리 베팅' : '2단계 · 본 전략';
  badge.className = 'comp-phase ' + (p1 ? 'p1' : 'p2');
  $('compLabel').textContent = c.label || '';
  $('compLeft').innerHTML = `${esc(c.period||'')} · 남은 <b>${num(c.days_remaining)}</b>영업일 · `
    + `보유 목표 <b>${num(c.top_k)}</b>종목`;

  // 초기화 권고. 10/16이 마지막 기회이고 코드가 실행하지 않으므로 사람이 봐야 한다.
  const rs = $('compReset');
  if (c.reset && c.reset.window_open) {
    rs.hidden = false;
    rs.className = 'reset ' + (c.reset.should_reset ? 'act' : 'hold');
    const head = c.reset.should_reset
      ? `<b>수익률 초기화 권고</b> — KIS 화면에서 직접 실행하십시오`
      : `<b>초기화 보류</b>`;
    rs.innerHTML = `${head}<br>누적 <b>${c.reset.total_return_pct>0?'+':''}`
      + `${c.reset.total_return_pct}%</b> · 마감 ${esc(c.reset.deadline||'')}<br>`
      + `<span style="color:var(--dim)">${esc(c.reset.reason||'')}</span>`;
  } else { rs.hidden = true; }

  // 최소 요건 — 못 채우면 수익률과 무관하게 실격이다.
  const reqs = c.requirements;
  const wrap = $('compReqs');
  if (!reqs || !reqs.requirements) { wrap.innerHTML = ''; return; }
  wrap.innerHTML = reqs.requirements.map(q => {
    const pct = Math.round((q.ratio||0)*100);
    const cur = q.unit === '원' ? (q.current/1e8).toFixed(2)+'억' : num(q.current)+q.unit;
    const req = q.unit === '원' ? (q.required/1e8).toFixed(2)+'억' : num(q.required)+q.unit;
    return `<div class="req ${q.met?'ok':'bad'}">
      <div class="req-top"><span>${esc(q.label)}</span><span>${q.met?'충족':pct+'%'}</span></div>
      <div class="req-val">${esc(cur)}</div>
      <div class="req-top"><span style="color:var(--dim)">기준 ${esc(req)}</span></div>
      <div class="req-bar"><i style="width:${pct}%"></i></div>
    </div>`;
  }).join('');
}

// 지수 종목 순위판 — 엔진이 60초마다 복수시세로 만든 표(매매에는 쓰지 않는다)
let lastRank = null;
const eok = v => (v ? num(v / 1e8, 0) + '억' : '—');
const rankName = x => esc(x.name || x.symbol) + `<span class="sub">${esc(x.symbol)}</span>`
  + (x.held ? '<span class="tag held">보유</span>' : '') + (x.small ? '<span class="tag">소형</span>' : '');
const chgCell = v => v === null || v === undefined ? '—' : `<span class="${tone(v)}">${signed(v)}%</span>`;
function renderRank(rk) {
  if (rk) lastRank = rk;
  rk = lastRank;
  if (!rk) return;
  const small = $('rankSmall').checked, idx = $('rankIdx').value, top = +$('rankTop').value;
  const pool = (rk.rows || []).filter(r => (!small || r.small) && (!idx || r.index === idx));
  $('rankNote').textContent = rk.time ? `${rk.time} 기준 · ${(rk.rows || []).length}종목 · 60초마다` : '장중에 갱신됩니다';
  const idxShort = s => ({KOSPI200: '코스피200', KOSDAQ150: '코스닥150'}[s] || s || '—');
  $('tvTbl').innerHTML = rows(['#', '종목', '거래대금', '등락률', '현재가', '시총'],
    pool.slice(0, top).map((r, i) => [String(i + 1), rankName(r), eok(r.turnover), chgCell(r.chg_pct),
                                       num(r.price), eok(r.mcap)]), '데이터 없음');
  const byChg = pool.filter(r => r.chg_pct !== null && r.chg_pct !== undefined)
                    .sort((a, b) => b.chg_pct - a.chg_pct);
  $('chgTbl').innerHTML = rows(['#', '종목', '등락률', '거래대금', '현재가', '시총'],
    byChg.slice(0, top).map((r, i) => [String(i + 1), rankName(r), chgCell(r.chg_pct), eok(r.turnover),
                                       num(r.price), eok(r.mcap)]), '데이터 없음');
  const picks = rk.picks || [];
  $('pickNote').textContent = picks.length ? `${picks.length}종목 · config/dash_picks.json` : '';
  $('pickTbl').innerHTML = rows(['종목', '지수', '현재가', '등락률', '거래대금', '거래대금 순위', '등락률 순위', '시총'],
    picks.map(r => [rankName(r), esc(idxShort(r.index)), num(r.price), chgCell(r.chg_pct), eok(r.turnover),
                    r.tv_rank ? r.tv_rank + '위' : '—', r.chg_rank ? r.chg_rank + '위' : '—', eok(r.mcap)]),
    '관심 종목이 없습니다');
  $('crossTbl').innerHTML = rows(['#', '종목', '지수', '현재가', '등락률', '거래대금', '거래대금 순위', '등락률 순위', '시총'],
    (rk.cross || []).map((r, i) => [String(i + 1), rankName(r), esc(idxShort(r.index)), num(r.price), chgCell(r.chg_pct), eok(r.turnover),
                    r.tv_rank + '위', r.chg_rank + '위', eok(r.mcap)]), '장중에 갱신됩니다');
}
['rankSmall', 'rankIdx', 'rankTop'].forEach(id => $(id).addEventListener('change', () => renderRank()));

async function refresh() {
  if (holding) return;
  let s;
  try {
    s = await (await fetch('/api/state', {cache:'no-store'})).json();
    $('toast').style.display = 'none';
    $('pulse').classList.remove('off');
  } catch (e) {
    $('toast').style.display = 'block';
    $('pulse').classList.add('off');
    return;
  }

  const p = s.portfolio||{}, r = s.risk||{}, st = s.strategy||{}, sys = s.system||{},
        rl = sys.rate_limiter||{}, ws = sys.websocket||{}, pr = sys.progress||{};

  renderCompetition(s.competition||{});

  const el = Math.round(sys.elapsed_sec||0);
  $('meta').innerHTML = [
    `<b>${esc(sys.mode||'—')}</b>`,
    sys.no_buy ? '<span class="nobuy">엔진 매수 차단</span>' : '엔진 매수 중',
    sys.no_sell ? '<span class="nobuy">엔진 매도 차단(손절·마감 매도 없음)</span>' : '엔진 매도 중',
    `장 ${esc({REGULAR:'정규장',PRE_AUCTION:'장전 동시호가',CLOSING_AUCTION:'마감 동시호가',AFTER_HOURS:'시간외',CLOSED:'휴장'}[sys.session]||sys.session||'—')}`,
    `유니버스 <b>${num(sys.universe)}</b>종목`,
    `루프 <b>${num(sys.loops)}</b>`,
    `가동 <b>${Math.floor(el/3600)}시간 ${Math.floor(el%3600/60)}분</b>`,
    sys.errors ? `<b class="up">오류 ${num(sys.errors)}</b>` : '오류 없음',
    ...(Object.keys(sys.stale_quotes||{}).length
        ? [`<b class="up">시세 멈춤 ${Object.entries(sys.stale_quotes).map(([k,v])=>`${k} ${v}초`).join(', ')}</b>`] : []),
  ].map(x => `<span>${x}</span>`).join('');

  $('equity').innerHTML = num(p.equity) + '<small>원</small>';
  $('delta').innerHTML = [
    ['누적', p.total_return_pct], ['당일', p.daily_pnl_pct],
  ].map(([k,v]) => `<span><i>${k}</i><b class="${tone(v)}">${signed(v)}%</b></span>`).join('')
   + `<span><i>현금</i>${num(p.cash)}</span><span><i>익스포저</i>${num((p.gross_exposure||0)*100,1)}%</span>`;

  // 증권사가 매긴 순자산(MTS·대회 순위 기준)과 엔진 자산의 차이 — 장 밖에는 0이어야 한다
  const be = p.broker_equity, bts = (p.broker_equity_ts||'').slice(11,16);
  $('brokerEq').innerHTML = (be===null||be===undefined) ? '' :
    (Math.abs(p.broker_gap||0) < 1
      ? `<span>증권사 순자산과 일치 <b>${num(be)}원</b></span><span><i>${bts} 대조</i></span>`
      : `<span>증권사 순자산 <b>${num(be)}원</b></span><span>차이 <b>${signed(p.broker_gap,0)}원</b></span><span><i>${bts} 대조</i></span>`);

  drawSpark(s.history||[]);
  drawGauge(r, {soft: sys.mdd_soft ?? 7, hard: sys.mdd_hard ?? 12});

  const pos = p.positions||[];
  $('posCount').textContent = pos.length ? `${pos.length}종목 · 평가손익 ${signed(pos.reduce((a,x)=>a+(x.unrealized||0),0),0)}원` : '';
  $('positions').innerHTML = rows(
    ['종목','수량','평단','현재가','평가금액','수익률','손절가'],
    pos.map(x => {
      const w = Math.min(100, Math.abs(x.pnl_pct||0) / 6 * 50);
      return [
        esc(x.name || x.symbol) + `<span class="sub">${esc(x.name ? x.symbol : (x.sector||''))}</span>`,
        num(x.qty), num(x.avg_price), num(x.last_price), num((x.qty||0) * (x.last_price||0)),
        `<span class="${tone(x.pnl_pct)}">${signed(x.pnl_pct)}%</span>
         <div class="pnlbar"><span class="axis"></span>
           <i class="" style="${(x.pnl_pct>=0
              ? `left:50%;width:${w}%;background:var(--rise)`
              : `left:${50-w}%;width:${w}%;background:var(--fall)`)}"></i></div>`,
        num(x.stop) + (x.trailing ? '<span class="sub">추적</span>' : ''),
      ];
    }), '보유 종목이 없습니다');

  // 오늘의 진입 판정 — 장 시작 판정(09:0N 한 번)과 장중 발동, 각각 어떻게 됐는지
  const tg = s.triggers||{}, op = tg.opening, sl = (tg.slide||[]).slice().reverse();
  const nm = x => esc(x.name || x.symbol) + (x.name ? `<span class="sub">${esc(x.symbol)}</span>` : '');
  const oc = o => `<span class="${o==='매수 주문' ? 'buy' : 'flat'}">${esc(o||'')}</span>`;
  $('openNote').textContent = !op ? '' : op.skipped ? op.skipped
    : `${op.time} · ${op.checked}종목 · ${op.window}분 x${op.mult} 이상 ${op.rows.length}건`;
  $('openTbl').innerHTML = rows(['종목','배수','결과'],
    (op && op.rows || []).slice(0,10).map(x => [nm(x), 'x' + num(x.ratio,1), oc(x.outcome)]),
    !op ? '09:05 판정 전' : op.skipped ? '오늘은 판정하지 않았습니다' : '발동 없음');
  const nbuy = sl.filter(x => x.outcome === '매수 주문').length;
  $('slideNote').textContent = sl.length ? `${sl.length}건 · 매수 ${nbuy}건` : '';
  $('slideTbl').innerHTML = rows(['종목','시각','배수','결과'],
    sl.slice(0,12).map(x => [nm(x), esc(String(x.time||'').slice(0,5)), 'x' + num(x.ratio,1), oc(x.outcome)]),
    '발동 없음');

  renderRank(s.rank);
  drawRps(s.rps_history||[], rl);

  $('progLabel').textContent = pr.label || '';
  $('progBar').style.width = (pr.pct||0) + '%';
  $('progDl').innerHTML = [
    ['진행률', num(pr.pct,1) + '%'],
    ['미체결 주문', num(sys.open_orders) + '건'],
    ['이벤트 대기', num(sys.event_queue) + '건'],
    ['실시간 연결', ws.connected ? '<span style="color:var(--live)">연결됨</span> ' + esc(ws.slots||'')
                                 : '<span class="flat">REST 폴링</span>'],
  ].map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

  $('riskDl').innerHTML = [
    ['승률 / 손익비', num(r.win_rate,1) + '% / ' + num(r.payoff,2)],
    ['켈리 비중', num(r.kelly_weight,3)],
    ['진입 거부', num(r.blocked) + '건'],
    ['킬스위치', num(r.kill_switches) + '회'],
  ].map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')
   + (r.recent_reasons||[]).slice(-3).map(x => `<dt class="why">최근</dt><dd class="why">${esc(x)}</dd>`).join('');

  const f = (s.recent_fills||[]).slice().reverse();
  $('fillCount').textContent = f.length ? `최근 ${f.length}건` : '';
  $('fills').innerHTML = rows(
    ['시각','종목','구분','수량','체결가','실현손익','사유'],
    f.map(x => [
      esc(String(x.ts||'').slice(11,19)), esc(x.symbol),
      `<span class="${x.side==='BUY'?'up':'down'}">${x.side==='BUY'?'매수':'매도'}</span>`,
      num(x.qty), num(x.price),
      x.realized ? `<span class="${tone(x.realized)}">${signed(x.realized,0)}</span>` : '—',
      `<span class="why">${esc(x.reason||'')}</span>`,
    ]), '체결 없음');
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def create_app(state_provider: StateProvider, settings=None):
    """관제 화면 FastAPI 앱 생성."""
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

    app = FastAPI(title="KIS AI Quant Trader", docs_url=None, redoc_url=None)
    diagram = Path(__file__).resolve().parent.parent / "docs" / "architecture.html"

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _HTML

    @app.get("/architecture")
    async def architecture():
        if diagram.exists():
            return FileResponse(diagram, media_type="text/html")
        return HTMLResponse(
            "<body style='background:#0B0F14;color:#7D90A6;font-family:sans-serif;padding:40px'>"
            "시스템 구조 다이어그램이 아직 없습니다. "
            "<code>node bin/archify.mjs deliver architecture docs/architecture.spec.json "
            "docs/architecture.html</code> 로 생성하세요.</body>",
            status_code=404,
        )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        state = state_provider()
        sys_state = state.get("system", {})
        return {
            "status": "ok" if sys_state.get("running") else "starting",
            "mode": sys_state.get("mode"),
            "loops": sys_state.get("loops"),
            "errors": sys_state.get("errors"),
            "risk_state": (state.get("risk") or {}).get("state"),
        }

    @app.get("/api/state")
    async def api_state() -> Dict[str, Any]:
        state = state_provider()
        if settings is not None:
            state.setdefault("system", {})
            state["system"]["mdd_soft"] = settings.MDD_SOFT_LIMIT * 100
            state["system"]["mdd_hard"] = settings.MDD_HARD_LIMIT * 100
            for pos in (state.get("portfolio") or {}).get("positions") or []:
                pos.setdefault("name", settings.name_of(str(pos.get("symbol", ""))))
        return state

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        state = state_provider()
        p = state.get("portfolio", {})
        r = state.get("risk", {})
        rl = (state.get("system", {}) or {}).get("rate_limiter", {})
        lines = [
            "# HELP kis_equity 총자산(원)",
            "# TYPE kis_equity gauge",
            f"kis_equity {p.get('equity', 0)}",
            "# HELP kis_drawdown_pct 현재 낙폭(%)",
            "# TYPE kis_drawdown_pct gauge",
            f"kis_drawdown_pct {p.get('drawdown_pct', 0)}",
            "# HELP kis_daily_pnl_pct 당일 손익(%)",
            "# TYPE kis_daily_pnl_pct gauge",
            f"kis_daily_pnl_pct {p.get('daily_pnl_pct', 0)}",
            "# HELP kis_positions 보유 종목 수",
            "# TYPE kis_positions gauge",
            f"kis_positions {r.get('positions', 0)}",
            "# HELP kis_rate_limit_rejects 서버 유량 거부 누적",
            "# TYPE kis_rate_limit_rejects counter",
            f"kis_rate_limit_rejects {rl.get('server_rejects', 0)}",
            "# HELP kis_kill_switches 킬스위치 발동 누적",
            "# TYPE kis_kill_switches counter",
            f"kis_kill_switches {r.get('kill_switches', 0)}",
        ]
        sysd = state.get("system", {}) or {}
        lines += [
            "# HELP kis_event_dropped 손실 허용 채널에서 버린 이벤트(시세) 누적",
            "# TYPE kis_event_dropped counter",
            f"kis_event_dropped {sysd.get('event_dropped', 0)}",
            "# HELP kis_rate_limit_wait_ms 유량 제어 평균 대기(ms)",
            "# TYPE kis_rate_limit_wait_ms gauge",
            f"kis_rate_limit_wait_ms {rl.get('avg_wait_ms', 0)}",
        ]
        if sysd.get("clock_skew_s") is not None:
            lines += ["# HELP kis_clock_skew_seconds 증권사 서버 시각 - 로컬 시각",
                      "# TYPE kis_clock_skew_seconds gauge",
                      f"kis_clock_skew_seconds {sysd['clock_skew_s']}"]
        lines += list(sysd.get("http_latency_prom") or [])
        return "\n".join(lines) + "\n"

    return app


class DashboardServer:
    """uvicorn 서버를 백그라운드 태스크로 구동/종료한다."""

    def __init__(self, state_provider: StateProvider, host: str = "127.0.0.1",
                 port: int = 8500, settings=None):
        self.state_provider = state_provider
        self.host = host
        self.port = port
        self.settings = settings
        self._server = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        try:
            import uvicorn
        except ImportError:
            logger.warning("[Dashboard] uvicorn 미설치로 관제 화면을 건너뜁니다.")
            return False
        try:
            app = create_app(self.state_provider, self.settings)
            config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning",
                                    access_log=False)
            self._server = uvicorn.Server(config)
            self._task = asyncio.create_task(self._server.serve(), name="dashboard")
            await asyncio.sleep(0.4)  # 기동 대기
            logger.success(f"[Dashboard] http://{self.host}:{self.port} 에서 확인하세요")
            return True
        except Exception as exc:
            logger.warning(f"[Dashboard] 기동 실패(매매는 계속 진행): {exc}")
            return False

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._task.cancel()
            self._task = None
