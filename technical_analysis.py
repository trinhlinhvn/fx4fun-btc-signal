"""
Technical Analysis Module
Calculates technical indicators and generates TA-based signals.
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from config import TA_CONFIG


class TechnicalAnalyzer:
    """Performs technical analysis on BTC price data."""

    def __init__(self):
        self.config = TA_CONFIG

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators on the dataframe."""
        if df.empty or len(df) < 26:
            return df

        # RSI
        rsi = RSIIndicator(close=df["close"], window=self.config["rsi_period"])
        df["rsi"] = rsi.rsi()

        # MACD
        macd = MACD(
            close=df["close"],
            window_slow=self.config["macd_slow"],
            window_fast=self.config["macd_fast"],
            window_sign=self.config["macd_signal"],
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # EMA
        ema_short = EMAIndicator(close=df["close"], window=self.config["ema_short"])
        ema_long = EMAIndicator(close=df["close"], window=self.config["ema_long"])
        df["ema_short"] = ema_short.ema_indicator()
        df["ema_long"] = ema_long.ema_indicator()

        # Bollinger Bands
        bb = BollingerBands(
            close=df["close"],
            window=self.config["bb_period"],
            window_dev=self.config["bb_std"],
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()

        # Volume analysis (if volume column exists)
        if "volume" in df.columns:
            df["vol_sma20"] = df["volume"].rolling(20).mean()
            df["vol_ratio"] = df["volume"] / df["vol_sma20"]  # >1.5 = high volume

        return df

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze technical indicators and return a signal score.
        Returns a dict with individual signals and overall score (-1 to 1).
        """
        df = self.calculate_indicators(df)

        if df.empty or df["rsi"].isna().all():
            return {"score": 0, "signals": {}, "error": "Insufficient data"}

        latest = df.iloc[-1]
        signals = {}

        # --- RSI Signal ---
        rsi_value = latest["rsi"]
        if rsi_value <= self.config["rsi_oversold"]:
            signals["rsi"] = {"value": rsi_value, "signal": "BUY", "score": 1.0}
        elif rsi_value >= self.config["rsi_overbought"]:
            signals["rsi"] = {"value": rsi_value, "signal": "SELL", "score": -1.0}
        elif rsi_value < 45:
            signals["rsi"] = {"value": rsi_value, "signal": "LEAN BUY", "score": 0.3}
        elif rsi_value > 55:
            signals["rsi"] = {"value": rsi_value, "signal": "LEAN SELL", "score": -0.3}
        else:
            signals["rsi"] = {"value": rsi_value, "signal": "NEUTRAL", "score": 0.0}

        # --- MACD Signal ---
        macd_val = latest["macd"]
        macd_sig = latest["macd_signal"]
        macd_hist = latest["macd_histogram"]

        if macd_val > macd_sig and macd_hist > 0:
            score = min(1.0, macd_hist / abs(macd_val) if macd_val != 0 else 0.5)
            signals["macd"] = {"value": round(macd_val, 2), "signal": "BUY", "score": score}
        elif macd_val < macd_sig and macd_hist < 0:
            score = max(-1.0, macd_hist / abs(macd_val) if macd_val != 0 else -0.5)
            signals["macd"] = {"value": round(macd_val, 2), "signal": "SELL", "score": score}
        else:
            signals["macd"] = {"value": round(macd_val, 2), "signal": "NEUTRAL", "score": 0.0}

        # --- EMA Crossover Signal ---
        ema_short = latest["ema_short"]
        ema_long = latest["ema_long"]
        ema_diff_pct = (ema_short - ema_long) / ema_long * 100

        if ema_short > ema_long:
            score = min(1.0, ema_diff_pct / 3)  # Normalize
            signals["ema_cross"] = {
                "value": f"{ema_short:.0f}/{ema_long:.0f}",
                "signal": "BUY",
                "score": score,
            }
        else:
            score = max(-1.0, ema_diff_pct / 3)
            signals["ema_cross"] = {
                "value": f"{ema_short:.0f}/{ema_long:.0f}",
                "signal": "SELL",
                "score": score,
            }

        # --- Bollinger Bands Signal ---
        price = latest["close"]
        bb_upper = latest["bb_upper"]
        bb_lower = latest["bb_lower"]
        bb_middle = latest["bb_middle"]

        if price <= bb_lower:
            signals["bollinger"] = {"value": f"${price:.0f}", "signal": "BUY", "score": 0.8}
        elif price >= bb_upper:
            signals["bollinger"] = {"value": f"${price:.0f}", "signal": "SELL", "score": -0.8}
        elif price < bb_middle:
            score = (bb_middle - price) / (bb_middle - bb_lower) * 0.4
            signals["bollinger"] = {"value": f"${price:.0f}", "signal": "LEAN BUY", "score": score}
        else:
            score = -(price - bb_middle) / (bb_upper - bb_middle) * 0.4
            signals["bollinger"] = {"value": f"${price:.0f}", "signal": "LEAN SELL", "score": score}

        # --- Overall TA Score ---
        weights = {"rsi": 0.3, "macd": 0.25, "ema_cross": 0.25, "bollinger": 0.2}
        overall_score = sum(
            signals[key]["score"] * weights[key] for key in weights if key in signals
        )

        return {
            "score": round(overall_score, 4),
            "signals": signals,
            "latest_price": price,
            "data_points": len(df),
        }
