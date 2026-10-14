@echo off
rem Double-click: open the server dashboard in the browser via SSH tunnel (see open_dashboard.ps1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_dashboard.ps1" %*
