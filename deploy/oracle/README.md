# 오라클 무료 서버로 옮기기

노트북을 꺼도 엔진이 돌게 오라클 클라우드 무료 서버(Always Free)에 올린다.
**계정·서버 생성(1~4단계)은 사용자가, 나머지는 `deploy.sh`가 한다.**

## 1. 계정 만들기
- https://www.oracle.com/cloud/free/ → 가입(신용카드 인증 — 무료 범위에서는 청구되지 않는다)
- 홈 리전: **South Korea Central (Seoul)** 또는 **South Korea North (Chuncheon)**

## 2. 서버(인스턴스) 만들기
콘솔 → Compute → Instances → **Create instance**
- Image: **Canonical Ubuntu 24.04**
- Shape: **Ampere → VM.Standard.A1.Flex**, **OCPU 2 · Memory 12GB** (무료 범위)
  - "Out of capacity"가 나면 다른 가용 도메인(AD)을 고르거나 잠시 뒤 다시 시도한다
  - 1GB짜리 `VM.Standard.E2.1.Micro`는 이 엔진에 메모리가 모자란다
- Networking: 기본값(공인 IPv4 자동 할당 **켜기**)
- **Add SSH keys → Paste public keys** 에 아래 한 줄을 붙여 넣는다

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGo4vtgdb4V8lhZ09tXmyAF1xxxA5xCP3xAKZKMHfqc6 bankis-oracle
```

(비밀키는 노트북 `~/.ssh/bankis_oracle`에만 있다. 공개키는 위 한 줄이다.)

## 3. 공인 IP 확인
인스턴스 상세 → **Public IP address** 를 알려 준다. 방화벽(포트) 설정은 필요 없다 —
SSH(22)만 쓰고, 대시보드는 SSH 터널로 본다.

## 4. 알려 줄 것
- 공인 IP

## 5. 배포 (Claude 가 한다)
```bash
bash deploy/oracle/deploy.sh <IP> setup    # 코드·데이터 복사, Python 3.13·패키지 설치, 서비스 등록
bash deploy/oracle/deploy.sh <IP> switch   # 장 마감 후: 노트북 엔진 정지 → 원장·상태 복사 → 서버 엔진 시작
bash deploy/oracle/deploy.sh <IP> status   # 상태·최근 로그
bash deploy/oracle/deploy.sh <IP> code     # 대회 중 코드 수정 반영(커밋된 코드만)
```

## 알아 둘 것
- **같은 계좌에 엔진 둘을 동시에 돌리지 않는다** — 주문이 두 번 나간다. `switch`가 노트북 엔진을
  먼저 멈추고, 장중(08:30~15:40)에는 전환을 거부한다.
- KIS는 IP 등록이 필요 없다(토스와 다르다). 서버 IP가 바뀌어도 매매에는 문제없다.
- 서버 엔진은 systemd 서비스라 죽거나 서버가 재부팅돼도 30초 뒤 다시 뜬다.
- 대시보드: `ssh -i ~/.ssh/bankis_oracle -L 8500:127.0.0.1:8500 ubuntu@<IP>` 후 브라우저 `http://127.0.0.1:8500`
- 무료 A1 인스턴스는 **유휴 회수 정책**이 있다(7일간 CPU·네트워크·메모리 사용률이 아주 낮으면
  회수될 수 있다). 엔진이 장중 계속 돌므로 해당될 가능성은 낮지만, 계정을 유료(Pay As You Go)로
  올려 두면 회수 대상에서 빠지고 무료 범위 사용은 여전히 0원이다.
