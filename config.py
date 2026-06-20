"""
Fx4Fun — BTC Signal Bot v4.1 STABLE
Backtest-optimized config: WR 55.6%, PF 2.03, RR 2:1
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (optional)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === CORE ===
SCAN_INTERVAL_SECONDS = 300  # 5 phút

# === BACKTEST-OPTIMIZED PARAMETERS ===
# Best config: SL 1.5% / TP 3.0% / RR 2:1 / PF 2.03
TRADE_CONFIG = {
    "sl_pct": 1.5,       # Stop Loss 1.5%
    "tp1_pct": 3.0,      # Take Profit 1: 3.0% (R:R 2:1)
    "tp2_pct": 4.5,      # Take Profit 2: 4.5% (R:R 3:1)
    "tp3_pct": 7.5,      # Take Profit 3: 7.5% (R:R 5:1)
}

# Technical Analysis (H4 optimized)
TA_CONFIG = {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_short": 21,
    "ema_long": 50,
    "bb_period": 20,
    "bb_std": 2,
}

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.4,
    "buy": 0.15,
    "hold_upper": 0.15,
    "hold_lower": -0.15,
    "sell": -0.15,
    "strong_sell": -0.4,
}

# Signal Weights (reference)
SIGNAL_WEIGHTS = {
    "technical": 0.30,
    "smc": 0.25,
    "volume_profile": 0.25,
    "money_flow": 0.20,
}

# ML Settings
ML_CONFIG = {
    "training_days": 365,
    "prediction_horizon": 5,
    "retrain_interval_hours": 12,
     "min_confidence": 0.45,
    "lstm_sequence_length": 20,
}

# News Settings
NEWS_CONFIG = {
    "keywords": ["bitcoin", "BTC", "crypto", "cryptocurrency"],
    "max_articles": 20,
    "lookback_hours": 24,
}
