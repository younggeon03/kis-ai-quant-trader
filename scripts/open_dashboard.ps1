# 서버 관제 화면을 내 PC 브라우저로 연다 — SSH 터널(로컬 8500 → 서버 127.0.0.1:8500).
#
#   더블클릭: scripts\open_dashboard.cmd
#
# 관제 화면은 서버 자신(127.0.0.1)에만 열려 있고 로그인이 없다. 포트를 인터넷에 열면 계좌가 그대로 보이므로
# 열지 않고, 이미 쓰는 SSH 키로 터널만 만든다(추가 비용 0). 터널이 이미 열려 있으면 브라우저만 띄운다.
# 서버 IP는 저장소에 두지 않는다 — 환경변수 BANKIS_SERVER 또는 deploy\server.local(한 줄, git 제외)에서 읽는다.
param(
    [string]$Server = "",
    [string]$User = "root",
    [string]$Key = "$env:USERPROFILE\.ssh\bankis_oracle",
    [int]$Port = 8500
)

if (-not $Server) { $Server = $env:BANKIS_SERVER }
if (-not $Server) {
    $local = Join-Path $PSScriptRoot "..\deploy\server.local"
    if (Test-Path $local) { $Server = (Get-Content $local -TotalCount 1).Trim() }
}
if (-not $Server) {
    Write-Host "서버 IP가 없습니다 — deploy\server.local 에 IP 한 줄을 넣거나 BANKIS_SERVER 를 설정하세요."
    Start-Sleep -Seconds 8
    exit 1
}

function Test-Port([int]$p) {
    try { $c = New-Object Net.Sockets.TcpClient; $c.Connect("127.0.0.1", $p); $ok = $c.Connected; $c.Close(); return $ok }
    catch { return $false }
}

if (-not (Test-Port $Port)) {
    Start-Process ssh -ArgumentList '-i', $Key, '-N', '-L', "${Port}:127.0.0.1:8500",
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '-o', 'ExitOnForwardFailure=yes',
        '-o', 'StrictHostKeyChecking=accept-new', "$User@$Server" -WindowStyle Hidden
    for ($i = 0; $i -lt 20 -and -not (Test-Port $Port); $i++) { Start-Sleep -Milliseconds 500 }
}
if (Test-Port $Port) {
    Start-Process "http://localhost:$Port/"
} else {
    Write-Host "터널을 열지 못했습니다 — 서버 상태나 SSH 키($Key)를 확인하세요."
    Start-Sleep -Seconds 8
}
