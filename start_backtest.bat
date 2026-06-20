@echo off
title Fx4Fun v4.1 — Backtest
color 0D
echo.
echo  Fx4Fun v4.1 — Backtest
echo  60 days H4 data
echo  SL 1.5%% / TP 3.0%%
echo  Cho 30-60 giay...
echo.
cd /d D:\Kiro\Fx4Fun
python -c "from backtester import Backtester; bt=Backtester(); r=bt.run_backtest(days=60, sl_pct=1.5, tp_pct=3.0, leverage=1); print(); print('=== BACKTEST ==='); print(f'Trades: {r.get(\"total_trades\",0)}'); print(f'Win Rate: {r.get(\"win_rate\",0)}%%'); print(f'Profit Factor: {r.get(\"profit_factor\",0)}'); print(f'PnL: {r.get(\"total_pnl_pct\",0)}%%'); print(f'Max DD: {r.get(\"max_drawdown_pct\",0)}%%'); print(f'Sharpe: {r.get(\"sharpe_ratio\",0)}')"
echo.
pause
