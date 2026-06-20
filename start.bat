@echo off
title Fx4Fun v4.1 STABLE
color 0A

:: Tắt Quick Edit mode (tránh treo khi click vào terminal)
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f >nul 2>&1

echo.
echo  Fx4Fun v4.1 STABLE
echo  Web: http://localhost:5173
echo  H4 trend + M15 entry | Quet 5 phut
echo  SL 1.5%% / TP 3.0%% / RR 2:1
echo.
cd /d D:\Kiro\Fx4Fun
set PORT=5173
python web_dashboard.py
pause
