@echo off
title Fx4Fun v4.1 — Telegram
color 0B

:: Tắt Quick Edit mode
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f >nul 2>&1

echo.
echo  Fx4Fun v4.1 — Telegram Alert
echo  Loop 5 phut | SL 1.5%% / TP 3.0%%
echo.
cd /d D:\Kiro\Fx4Fun
python main.py --loop --telegram
pause
