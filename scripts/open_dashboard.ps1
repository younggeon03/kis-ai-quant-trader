# 서버 관제 화면을 내 PC 브라우저로 연다 — SSH 터널(로컬 8500 → 서버 127.0.0.1:8500).
#
#   더블클릭: scripts\open_dashboard.cmd
#
# 관제 화면은 서버 자신(127.0.0.1)에만 열려 있고 로그인이 없다. 포트를 인터넷에 열면 계좌가 그대로 보이므로
# 열지 않고, 이미 쓰는 SSH 키로 터널만 만든다(추가 비용 0). 터널이 이미 열려 있으면 브라우저만 띄운다.
param(
    [string]$Server = "$BANKIS_SERVER",
    [string]$User = "root",
    [string]$Key = "$env:USERPROFILE\.ssh\bankis_oracle",
    [int]$Port = 8500
)

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
