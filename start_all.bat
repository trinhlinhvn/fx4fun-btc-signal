@echo off
title Fx4Fun v4.1 — Full
color 0E

:: Tắt Quick Edit mode
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f >nul 2>&1

echo.
echo  Fx4Fun v4.1 — FULL (Web + Telegram)
echo  Web: http://localhost:5173
echo  Loop 5 phut | SL 1.5%% / TP 3.0%%
echo.
cd /d D:\Kiro\Fx4Fun
set PORT=5173
python main.py --train --all
pause
