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
sudo apt-get install -y curl ca-certificates build-essential

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

.venv/bin/python -c "import numpy, pandas, sklearn, lightgbm; print('패키지 OK', numpy.__version__, sklearn.__version__, lightgbm.__version__)"
# 키 로그인이 확인된 상태(이 스크립트가 키로 실행되고 있다)이므로 비밀번호 로그인을 끈다.
# NCP 는 22번을 전체에 열어 두므로 비밀번호 대입 공격을 막아야 한다.
if [ -s "$HOME/.ssh/authorized_keys" ]; then
  echo "PasswordAuthentication no" | sudo tee /etc/ssh/sshd_config.d/99-bankis.conf >/dev/null
  sudo systemctl reload ssh 2>/dev/null || sudo systemctl reload sshd 2>/dev/null || true
  echo "SSH 비밀번호 로그인 끔(키로만 접속)"
fi
echo "설정 완료 — 서비스는 deploy.sh 가 켠다(bankis.service)"
