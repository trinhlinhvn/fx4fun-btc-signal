"""
Backtesting Engine
==================
Test strategy trên historical data để biết win rate thực.
Giống như hedge fund chạy simulation trước khi trade real money.

Features:
- Test trên 90-365 ngày data
- Simulate entry/SL/TP
- Tính win rate, avg PnL, max drawdown, Sharpe ratio
- Test từng component riêng (TA only, SMC only, combined)
"""
import numpy as np
import pandas as pd
from datetime import datetime
from data_fetcher import BTCDataFetcher
from technical_analysis import TechnicalAnalyzer
from smc_ict_analysis import SMCICTAnalyzer
from money_flow import MoneyFlowAnalyzer


class Backtester:
    """Run historical backtests to validate strategy."""

    def __init__(self):
        self.fetcher = BTCDataFetcher()
        self.ta = TechnicalAnalyzer()
        self.smc = SMCICTAnalyzer()
        self.mf = MoneyFlowAnalyzer()

    def run_backtest(self, days: int = 90, sl_pct: float = 2.5, tp_pct: float = 3.5, leverage: int = 10) -> dict:
        """
        Backtest strategy on historical H4 data.
        Includes TA + SMC + Money Flow + Volume Profile/Order Flow.
        """
        print(f"[BACKTEST] Fetching {days} days of H4 data...")
        df = self.fetcher.get_klines(interval="4h", limit=min(days * 6, 1500))

        if df.empty or len(df) < 100:
            return {"error": "Insufficient data for backtest"}

        # Add indicators
        df = self.ta.calculate_indicators(df)

        # Volume Profile + Order Flow
        from volume_profile import VolumeOrderFlowEngine
        vof = VolumeOrderFlowEngine()

        trades = []
        entry_threshold = 0.15  # Lowered for more signals
        min_lookforward = 20

        print(f"[BACKTEST] Running simulation on {len(df)} candles...")

        for i in range(50, len(df) - min_lookforward):
            window = df.iloc[:i+1]
            ta_result = self.ta.analyze(window)
            ta_score = ta_result.get("score", 0)

            # SMC on recent 50 candles
            smc_window = window.iloc[-50:]
            smc_result = self.smc.generate_smc_analysis(smc_window)
            smc_score = smc_result.get("score", 0)

            # Money Flow
            mf_result = self.mf.analyze(window.iloc[-30:])
            mf_score = mf_result.get("score", 0)

            # Volume Profile + Order Flow (on last 40 candles)
            vof_result = vof.analyze(window.iloc[-40:])
            vof_score = vof_result.get("score", 0)

            # Combined score with TA priority
            combined = ta_score * 0.30 + smc_score * 0.25 + mf_score * 0.20 + vof_score * 0.25

            if abs(combined) < entry_threshold:
                continue

            is_long = combined > 0
            entry_price = float(df.iloc[i]["close"])
            sl_price = entry_price * (1 - sl_pct/100) if is_long else entry_price * (1 + sl_pct/100)
            tp_price = entry_price * (1 + tp_pct/100) if is_long else entry_price * (1 - tp_pct/100)

            result = self._simulate_trade(df, i, is_long, entry_price, sl_price, tp_price, min_lookforward)

            trades.append({
                "entry_idx": i,
                "entry_time": str(df.index[i]),
                "direction": "LONG" if is_long else "SHORT",
                "entry_price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "score": round(combined, 4),
                "ta_score": round(ta_score, 4),
                "smc_score": round(smc_score, 4),
                "mf_score": round(mf_score, 4),
                "vof_score": round(vof_score, 4),
                **result,
            })

        return self._calculate_stats(trades, leverage)

    def _simulate_trade(self, df, entry_idx: int, is_long: bool, entry: float, sl: float, tp: float, max_candles: int) -> dict:
        """Simulate a single trade, check TP/SL hit."""
        for j in range(1, min(max_candles, len(df) - entry_idx)):
            candle = df.iloc[entry_idx + j]
            high = float(candle["high"])
            low = float(candle["low"])

            if is_long:
                if low <= sl:
                    return {"outcome": "LOSS", "exit_price": sl, "candles_held": j, "hit": "SL"}
                if high >= tp:
                    return {"outcome": "WIN", "exit_price": tp, "candles_held": j, "hit": "TP"}
            else:
                if high >= sl:
                    return {"outcome": "LOSS", "exit_price": sl, "candles_held": j, "hit": "SL"}
                if low <= tp:
                    return {"outcome": "WIN", "exit_price": tp, "candles_held": j, "hit": "TP"}

        # Timeout — close at market
        exit_price = float(df.iloc[min(entry_idx + max_candles, len(df)-1)]["close"])
        pnl = (exit_price - entry) if is_long else (entry - exit_price)
        return {"outcome": "WIN" if pnl > 0 else "LOSS", "exit_price": exit_price, "candles_held": max_candles, "hit": "TIMEOUT"}

    def _calculate_stats(self, trades: list, leverage: int) -> dict:
        """Calculate backtest performance metrics."""
        if not trades:
            return {"error": "No trades generated", "trades": 0}

        wins = [t for t in trades if t["outcome"] == "WIN"]
        losses = [t for t in trades if t["outcome"] == "LOSS"]
        total = len(trades)
        win_rate = len(wins) / total * 100 if total > 0 else 0

        # PnL calculation
        pnls = []
        for t in trades:
            if t["direction"] == "LONG":
                pnl_pct = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100
            else:
                pnl_pct = (t["entry_price"] - t["exit_price"]) / t["entry_price"] * 100
            pnls.append(pnl_pct * leverage)

        avg_pnl = np.mean(pnls) if pnls else 0
        total_pnl = sum(pnls)
        max_drawdown = min(np.minimum.accumulate(np.cumsum(pnls))) if pnls else 0

        # Sharpe-like ratio
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252/6)  # Annualized for H4
        else:
            sharpe = 0

        # Avg candles held
        avg_hold = np.mean([t["candles_held"] for t in trades])

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
            "total_pnl_pct": round(total_pnl, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_hold_candles": round(avg_hold, 1),
            "leverage": leverage,
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0,
            "profit_factor": round(sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)), 2) if any(p < 0 for p in pnls) else 999,
            "trades_sample": trades[:5],  # First 5 for review
        }
