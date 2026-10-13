#!/usr/bin/env bash
# 노트북(Git Bash)에서 클라우드 서버(오라클·네이버 클라우드, Ubuntu)로 배포한다.
#
#   bash deploy/oracle/deploy.sh <서버 공인 IP> setup    # 코드·데이터 복사 + 서버 설정(엔진은 켜지 않음)
#   bash deploy/oracle/deploy.sh <서버 공인 IP> switch   # 노트북 엔진 정지 → 원장·상태 복사 → 서버 엔진 시작
#   bash deploy/oracle/deploy.sh <서버 공인 IP> code     # 코드만 갱신하고 서버 엔진 재시작(대회 중 수정 반영)
#   bash deploy/oracle/deploy.sh <서버 공인 IP> status   # 서버 엔진 상태·최근 로그
#
# **같은 계좌에 엔진 둘을 동시에 돌리지 않는다** — 주문이 두 번 나간다. switch 는 노트북 엔진을
# 먼저 멈추고 확인한 뒤에야 서버 엔진을 켠다. 장중(09:00~15:30)에는 switch 하지 않는다.
set -euo pipefail
IP="${1:?서버 IP}"
MODE="${2:?setup|switch|code|status}"
# 오라클은 ubuntu, 네이버 클라우드(NCP)는 root 로 접속한다: BANKIS_SSH_USER=root bash deploy.sh ...
KEY="${BANKIS_SSH_KEY:-$HOME/.ssh/bankis_oracle}"
HOST="${BANKIS_SSH_USER:-ubuntu}@$IP"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

push_code() {
  # 커밋된 코드만 보낸다(작업 중인 수정이 섞이지 않게).
  git archive --format=tar HEAD | "${SSH[@]}" "mkdir -p ~/bankis && tar -x -C ~/bankis"
}
push_data() {
  tar -cf - .env models config/universe.json data/cache/opening | "${SSH[@]}" "tar -x -C ~/bankis"
}
laptop_engine_running() {
  powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*main.py*' -and \$_.CommandLine -notlike '*deploy*' }) { exit 0 } else { exit 1 }"
}

case "$MODE" in
  setup)
    push_code
    push_data
    "${SSH[@]}" "bash ~/bankis/deploy/oracle/setup.sh"
    "${SSH[@]}" "cd ~/bankis && .venv/bin/python main.py --health-only 2>&1 | tail -5"
    echo "setup 완료 — 서버 엔진은 아직 꺼져 있다. 장 마감 후 'switch' 로 전환한다."
    ;;
  switch)
    hhmm=$(date +%H%M)
    if [ "$hhmm" \> "0830" ] && [ "$hhmm" \< "1540" ]; then
      echo "장중(08:30~15:40)에는 전환하지 않는다 — 진입·청산 도중 엔진이 바뀐다"; exit 1
    fi
    echo "노트북 엔진 정지(재시작 안 함)..."
    powershell.exe -NoProfile -Command "New-Item -ItemType File -Force '$ROOT\\logs\\STOP' | Out-Null; Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*main.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }; Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*run_live.ps1*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    sleep 3
    if laptop_engine_running; then echo "노트북 엔진이 아직 돈다 — 중단"; exit 1; fi
    push_code
    push_data
    tar -cf - logs/ledger.jsonl state 2>/dev/null | "${SSH[@]}" "tar -x -C ~/bankis"
    "${SSH[@]}" "sudo systemctl enable --now bankis.service && sleep 5 && systemctl is-active bankis.service"
    echo "전환 완료 — 노트북 엔진은 멈췄고 서버 엔진이 돈다. 확인: bash deploy/oracle/deploy.sh $IP status"
    ;;
  code)
    push_code
    "${SSH[@]}" "sudo systemctl restart bankis.service && sleep 5 && systemctl is-active bankis.service"
    ;;
  status)
    "${SSH[@]}" "systemctl is-active bankis.service; tail -n 15 ~/bankis/logs/trader.log 2>/dev/null || tail -n 15 ~/bankis/logs/service.log"
    ;;
  *) echo "모드: setup | switch | code | status"; exit 1 ;;
esac
