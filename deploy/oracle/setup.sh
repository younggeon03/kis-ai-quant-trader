#!/usr/bin/env bash
# 오라클 무료 서버(Ubuntu 24.04, Ampere A1 ARM) 최초 설정 — 서버에서 한 번 실행한다.
#
#   bash ~/bankis/deploy/oracle/setup.sh
#
# 하는 일: 시간대를 한국으로, uv 로 Python 3.13 가상환경을 만들고 고정 버전 패키지를 설치,
# systemd 서비스를 등록한다(엔진이 죽거나 서버가 재부팅돼도 다시 뜬다).
# 서비스는 **등록만 하고 켜지 않는다** — 노트북 엔진을 멈춘 뒤 deploy.sh 가 켠다.
# 같은 계좌에 엔진 둘이 돌면 주문이 두 번 나간다.
set -euo pipefail
APP="$HOME/bankis"

sudo timedatectl set-timezone Asia/Seoul
sudo apt-get update -y
sudo apt-get install -y curl ca-certificates build-essential chrony logrotate

# --- 시간 동기화 — 09:10 판정·진입이 서버 시계로 돈다. 엔진도 증권사 Date 헤더로 오차를 잰다 ---
sudo systemctl enable --now chrony

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$APP"
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r deploy/oracle/requirements-lock.txt

mkdir -p logs state data/cache models

sudo tee /etc/systemd/system/bankis.service >/dev/null <<EOF
[Unit]
Description=BanKis16 trading engine
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP
Environment=TZ=Asia/Seoul
Environment=PYTHONIOENCODING=utf-8
ExecStart=$APP/.venv/bin/python main.py
Restart=always
RestartSec=30
# 종료(SIGTERM) 때 엔진이 미체결 취소·상태 저장을 끝낼 시간을 준다(main.shutdown)
TimeoutStopSec=60
# 메모리 상한 — 실측 RSS 약 250MB. 누수가 나면 시스템 전체가 아니라 이 서비스만 재시작된다
MemoryMax=2G
LimitNOFILE=65536
StandardOutput=append:$APP/logs/service.log
StandardError=append:$APP/logs/service.err

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload

# 메모리 여유(무료 A1 12GB면 충분하지만 혹시 모를 급증 대비 스왑 2GB)
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# --- 비밀파일 권한 — KIS 앱키·계좌가 들어 있다 ---
[ -f "$APP/.env" ] && chmod 600 "$APP/.env"

# --- 로그 순환 — 하루 단위 30일 보관·압축. copytruncate: 엔진을 재시작하지 않고 자른다 ---
sudo tee /etc/logrotate.d/bankis >/dev/null <<EOF
$APP/logs/*.log $APP/logs/*.err {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# --- 커널 — 죽은 연결을 2시간이 아니라 약 2분 안에 정리(증권사 세션·WebSocket) ---
sudo tee /etc/sysctl.d/90-bankis.conf >/dev/null <<EOF
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6
net.core.somaxconn = 1024
EOF
sudo sysctl --system >/dev/null

# --- 원장·상태 백업 — 대회 근거자료. 장 마감 뒤 매일 압축 보관(30일) ---
mkdir -p "$HOME/backups"
( crontab -l 2>/dev/null | grep -v bankis-backup;
  echo "40 15 * * 1-5 tar -czf \$HOME/backups/bankis-\$(date +\%Y\%m\%d).tgz -C $APP logs/ledger.jsonl state 2>/dev/null; find \$HOME/backups -name 'bankis-*.tgz' -mtime +30 -delete  # bankis-backup"
) | crontab -

.venv/bin/python -c "import numpy, pandas, sklearn, lightgbm; print('패키지 OK', numpy.__version__, sklearn.__version__, lightgbm.__version__)"
# 키 로그인이 확인된 상태(이 스크립트가 키로 실행되고 있다)이므로 비밀번호 로그인을 끈다.
# NCP 는 22번을 전체에 열어 두므로 비밀번호 대입 공격을 막아야 한다.
if [ -s "$HOME/.ssh/authorized_keys" ]; then
  echo "PasswordAuthentication no" | sudo tee /etc/ssh/sshd_config.d/99-bankis.conf >/dev/null
  sudo systemctl reload ssh 2>/dev/null || sudo systemctl reload sshd 2>/dev/null || true
  echo "SSH 비밀번호 로그인 끔(키로만 접속)"
fi
echo "설정 완료 — 서비스는 deploy.sh 가 켠다(bankis.service)"
