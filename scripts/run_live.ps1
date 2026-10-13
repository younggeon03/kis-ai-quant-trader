# 대회 실거래 실행기 — main.py 가 예기치 않게 끝나면 30초 뒤 다시 띄운다.
#
# 왜 필요한가: Claude Code 백그라운드 작업은 메모리가 부족하면 정리되고, 엔진이 예외로 죽으면
# 그날 매매가 멈춘다. 이 스크립트는 독립 프로세스로 돌고, 재시작하면 엔진이 브로커 잔고·상태
# 파일(진입일·요건 매매 표시)로 복구한다.
#
# 멈추려면: logs\STOP 파일을 만든다(예: New-Item logs\STOP). 엔진이 끝난 뒤 다시 띄우지 않는다.
# 바로 끄려면 run_live 창(또는 powershell 프로세스)과 python main.py 프로세스를 종료한다.
$ErrorActionPreference = "Continue"
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$log = Join-Path $proj "logs\run_live.log"
$stop = Join-Path $proj "logs\STOP"
if (Test-Path $stop) { Remove-Item $stop -Force }
for ($i = 1; $i -le 1000; $i++) {
    Add-Content $log ("{0} 엔진 시작 #{1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $i)
    $out = Join-Path $proj ("logs\live_console_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $p = Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $proj `
        -RedirectStandardOutput $out -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Add-Content $log ("{0} 엔진 종료 #{1} 코드={2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $i, $p.ExitCode)
    if (Test-Path $stop) { Add-Content $log "STOP 파일 — 다시 띄우지 않는다"; break }
    Start-Sleep -Seconds 30
}
