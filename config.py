"""
Configuration for BTC Trading Signal Bot v3.0
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (all optional — app works without any keys)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Technical Analysis Settings (optimized for H4 swing trading)
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

# Signal Weights (used as reference; actual weights are dynamic in signal_engine)
SIGNAL_WEIGHTS = {
    "technical": 0.25,
    "sentiment": 0.15,
    "ml_prediction": 0.25,
    "smc": 0.35,
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

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.6,
    "buy": 0.3,
    "hold_upper": 0.3,
    "hold_lower": -0.3,
    "sell": -0.3,
    "strong_sell": -0.6,
}

# Futures Settings
FUTURES_CONFIG = {
    "leverage": 10,
    "default_margin_usd": 1000,
    "max_sl_pct": 0.03,  # 3% max SL (= 30% ROE with x10)
}
