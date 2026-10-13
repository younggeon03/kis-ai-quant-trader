# 네이버 클라우드(NCP)로 옮기기

배포 스크립트(`deploy.sh`·`setup.sh`)는 오라클과 같다. NCP는 **root 로 접속**하고, 처음에는
**비밀번호로** 들어가서 노트북 공개키를 한 번 등록한다. 그 뒤로는 Claude 가 키로 배포한다.

## 1. 가입·결제수단
- https://www.ncloud.com → 회원가입 → 결제수단 등록(신규 가입 크레딧이 있으면 함께 받는다)

## 2. 서버 만들기
콘솔 → Services → Compute → **Server** → **서버 생성**
- 플랫폼: **VPC** (VPC·Subnet 이 없으면 생성 화면에서 기본값으로 만든다 — Subnet 은 **Public**)
- 이미지: **Ubuntu Server 24.04** (없으면 22.04 — 설정 스크립트가 Python 3.13 을 따로 깐다)
- 서버 타입: **Standard**, 스펙 **2vCPU · 8GB** (최소 4GB. 2GB 이하는 메모리가 모자란다)
- 스토리지: 기본(50GB)
- 인증키: **새 인증키 생성** → `.pem` 파일을 내려받아 보관(관리자 비밀번호 확인에 쓴다)
- 네트워크 접근 설정(ACG): 기본 ACG 에 **인바운드 TCP 22**(SSH)를 허용
  - 접속 소스는 `0.0.0.0/0` 로 둔다 — 노트북 IP 가 자주 바뀐다. 대신 4단계 뒤 비밀번호 로그인을 끈다

## 3. 공인 IP
Server → **Public IP** → 공인 IP 신청 → 방금 만든 서버에 **할당**

## 4. 노트북 공개키 등록 (한 번만)
1. Server 목록 → 서버 선택 → **서버 관리 및 설정 변경 → 관리자 비밀번호 확인** → `.pem` 업로드 →
   root 비밀번호 확인
2. 노트북 **PowerShell** 에서(공인 IP 를 넣고, 비밀번호를 물으면 위 비밀번호 입력):

```powershell
type $env:USERPROFILE\.ssh\bankis_oracle.pub | ssh -o StrictHostKeyChecking=accept-new root@<공인IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

3. 공인 IP 를 Claude 에게 알려 준다

## 5. 배포 (Claude 가 한다)
```bash
BANKIS_SSH_USER=root bash deploy/oracle/deploy.sh <IP> setup
BANKIS_SSH_USER=root bash deploy/oracle/deploy.sh <IP> switch   # 장 마감 후
BANKIS_SSH_USER=root bash deploy/oracle/deploy.sh <IP> status
```
setup 이 끝나면 SSH 비밀번호 로그인을 끈다(키로만 접속 — 22번이 전체에 열려 있으므로).

## 알아 둘 것
- 같은 계좌에 엔진 둘을 동시에 돌리지 않는다 — `switch` 가 노트북 엔진을 먼저 멈추고 장중에는 거부한다
- KIS 는 IP 등록이 필요 없다. 서버는 systemd 로 돌아 죽거나 재부팅돼도 30초 뒤 다시 뜬다
- 요금은 서버 스펙·공인 IP 사용 시간으로 청구된다. 대회 후(11/20) **서버와 공인 IP 를 반납**해야 과금이 멈춘다
