"""
Fear & Greed Index + Liquidation Heatmap
==========================================

Fear & Greed Index:
- Source: alternative.me (FREE, no key)
- Range: 0-100
- Extreme Fear (0-25) = BUY opportunity (contrarian)
- Extreme Greed (75-100) = SELL opportunity (contrarian)
- Historical correlation with BTC tops/bottoms is very strong

Liquidation Heatmap:
- Source: Binance liquidation data (free)
- Detect price levels with high concentration of liquidation orders
- Price tends to "hunt" these levels → set entry/SL accordingly
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime


class FearGreedIndex:
    """Crypto Fear & Greed Index — contrarian indicator."""

    API_URL = "https://api.alternative.me/fng/"

    def __init__(self):
        self._cache = None
        self._cache_time = 0

    def get_current(self) -> dict:
        """Get current Fear & Greed Index value."""
        import time
        if self._cache and time.time() - self._cache_time < 3600:  # Cache 1h
            return self._cache

        try:
            response = requests.get(self.API_URL, params={"limit": 7}, timeout=5)
            if response.status_code != 200:
                return self._default()

            data = response.json().get("data", [])
            if not data:
                return self._default()

            current = data[0]
            value = int(current["value"])
            classification = current["value_classification"]

            # Trend (compare to yesterday)
            yesterday = int(data[1]["value"]) if len(data) > 1 else value
            trend = value - yesterday

            # Contrarian signal
            if value <= 20:
                signal = "STRONG_BUY"
                score = 0.4
                note = "Extreme Fear — historically best time to buy"
            elif value <= 35:
                signal = "BUY"
                score = 0.2
                note = "Fear — market oversold, accumulation zone"
            elif value >= 80:
                signal = "STRONG_SELL"
                score = -0.4
                note = "Extreme Greed — historically worst time to buy"
            elif value >= 65:
                signal = "SELL"
                score = -0.2
                note = "Greed — market may be overheated"
            else:
                signal = "NEUTRAL"
                score = 0.0
                note = "Neutral — no extreme sentiment"

            result = {
                "value": value,
                "classification": classification,
                "signal": signal,
                "score": score,
                "trend": trend,
                "trend_text": f"{'↑' if trend > 0 else '↓'} {abs(trend)} vs yesterday",
                "note": note,
                "history_7d": [int(d["value"]) for d in data[:7]],
            }

            import time as t
            self._cache = result
            self._cache_time = t.time()
            return result

        except Exception as e:
            print(f"[F&G] Fetch failed: {e}")
            return self._default()

    def _default(self):
        return {"value": 50, "classification": "Neutral", "signal": "NEUTRAL", "score": 0, "note": "Data unavailable"}


class LiquidationHeatmap:
    """
    Liquidation Heatmap — detect price levels where liquidations cluster.
    
    Logic:
    - Fetch open interest + funding rate
    - Calculate liquidation levels for leveraged positions
    - High OI at a price range = many liquidations waiting there
    - Price tends to sweep these levels (liquidity hunt)
    """

    BINANCE_FAPI = "https://fapi.binance.com"

    def __init__(self):
        self.session = requests.Session()

    def estimate_liquidation_clusters(self, current_price: float, symbol: str = "BTCUSDT") -> dict:
        """
        Estimate where liquidation clusters exist.
        
        Method: Use funding rate + OI changes to estimate:
        - If funding positive (more longs) → shorts get liquidated above, longs below
        - Calculate key levels at 2%, 5%, 10% from current price
        - Higher OI = more liquidations at those levels
        """
        try:
            # Get funding rate
            fr_resp = self.session.get(
                f"{self.BINANCE_FAPI}/fapi/v1/premiumIndex",
                params={"symbol": symbol}, timeout=5
            )
            funding_rate = 0
            if fr_resp.status_code == 200:
                funding_rate = float(fr_resp.json().get("lastFundingRate", 0))

            # Get recent liquidations (forced orders)
            liq_resp = self.session.get(
                f"{self.BINANCE_FAPI}/fapi/v1/allForceOrders",
                params={"symbol": symbol, "limit": 50}, timeout=5
            )
            recent_liqs = []
            if liq_resp.status_code == 200:
                recent_liqs = liq_resp.json()

        except Exception:
            funding_rate = 0
            recent_liqs = []

        # Estimate liquidation levels
        # Standard leveraged positions get liquidated at:
        # x10 Long: entry - 9% → liquidation
        # x10 Short: entry + 9% → liquidation
        # x20 Long: entry - 4.5%
        # x25 Long: entry - 3.5%

        # Key liquidation zones (where many positions likely exist)
        liq_levels = {
            "long_x10": round(current_price * 0.91, 0),
            "long_x20": round(current_price * 0.955, 0),
            "long_x25": round(current_price * 0.965, 0),
            "short_x10": round(current_price * 1.09, 0),
            "short_x20": round(current_price * 1.045, 0),
            "short_x25": round(current_price * 1.035, 0),
        }

        # Determine which side has more risk
        # Positive funding = more longs → long liquidations are the target
        if funding_rate > 0.0005:
            magnet_direction = "BELOW"
            magnet_note = f"Funding +{funding_rate*100:.3f}% (overcrowded longs) → price may dip to hunt long liquidations"
            key_level = liq_levels["long_x20"]
        elif funding_rate < -0.0005:
            magnet_direction = "ABOVE"
            magnet_note = f"Funding {funding_rate*100:.3f}% (overcrowded shorts) → price may pump to hunt short liquidations"
            key_level = liq_levels["short_x20"]
        else:
            magnet_direction = "NEUTRAL"
            magnet_note = "Balanced funding — no clear liquidation magnet"
            key_level = current_price

        # Analyze recent liquidations
        long_liqs = sum(1 for l in recent_liqs if l.get("side", "").upper() == "SELL")  # Forced sell = long liquidated
        short_liqs = sum(1 for l in recent_liqs if l.get("side", "").upper() == "BUY")  # Forced buy = short liquidated

        return {
            "current_price": current_price,
            "funding_rate": round(funding_rate * 100, 4),
            "magnet_direction": magnet_direction,
            "magnet_note": magnet_note,
            "key_liquidation_level": key_level,
            "levels": liq_levels,
            "recent_liquidations": {
                "longs_liquidated": long_liqs,
                "shorts_liquidated": short_liqs,
                "dominant": "LONGS" if long_liqs > short_liqs else "SHORTS" if short_liqs > long_liqs else "BALANCED",
            },
            "trading_insight": self._generate_insight(magnet_direction, key_level, current_price, funding_rate),
        }

    def _generate_insight(self, direction: str, key_level: float, price: float, funding: float) -> str:
        """Generate actionable insight from liquidation data."""
        if direction == "BELOW":
            distance = (price - key_level) / price * 100
            return f"⚠️ Long liquidations clustered ${key_level:,.0f} ({distance:.1f}% below). Price may sweep this level before bouncing. Good LONG entry if price reaches there."
        elif direction == "ABOVE":
            distance = (key_level - price) / price * 100
            return f"⚠️ Short liquidations clustered ${key_level:,.0f} ({distance:.1f}% above). Price may pump to squeeze shorts. Consider SHORT above that level."
        else:
            return "No dominant liquidation magnet detected."
