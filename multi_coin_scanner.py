"""
Multi-Coin Scanner
==================
Scan top 10 cryptocurrencies, rank by signal strength.
Returns top 3 with STRONGEST signals (BUY or SELL).

Coins scanned (Binance Futures):
1. BTCUSDT  - Bitcoin
2. ETHUSDT  - Ethereum
3. SOLUSDT  - Solana
4. BNBUSDT  - BNB
5. XRPUSDT  - Ripple
6. DOGEUSDT - Dogecoin
7. AVAXUSDT - Avalanche
8. ADAUSDT  - Cardano
9. LINKUSDT - Chainlink
10. MATICUSDT - Polygon
"""
import time
from datetime import datetime
from data_fetcher import BTCDataFetcher
from technical_analysis import TechnicalAnalyzer
from expert_analysis import ExpertAnalyzer
from smc_ict_analysis import SMCICTAnalyzer


class MultiCoinScanner:
    """Scan top crypto pairs and rank by signal strength."""

    TOP_COINS = [
        {"symbol": "BTCUSDT",  "name": "Bitcoin"},
        {"symbol": "ETHUSDT",  "name": "Ethereum"},
        {"symbol": "SOLUSDT",  "name": "Solana"},
        {"symbol": "BNBUSDT",  "name": "BNB"},
        {"symbol": "XRPUSDT",  "name": "Ripple"},
        {"symbol": "DOGEUSDT", "name": "Dogecoin"},
        {"symbol": "AVAXUSDT", "name": "Avalanche"},
        {"symbol": "ADAUSDT",  "name": "Cardano"},
        {"symbol": "LINKUSDT", "name": "Chainlink"},
        {"symbol": "MATICUSDT","name": "Polygon"},
    ]

    def __init__(self):
        self.ta = TechnicalAnalyzer()
        self.expert = ExpertAnalyzer()
        self.smc = SMCICTAnalyzer()

    def _fetch_klines_for_symbol(self, symbol: str, interval: str = "4h", limit: int = 200):
        """Fetch klines for any symbol from Binance Futures."""
        import requests
        import pandas as pd

        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            response = requests.get(url, params=params, timeout=8)
            if response.status_code != 200:
                return pd.DataFrame()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df[["open", "high", "low", "close", "volume"]].sort_index()
            return df
        except Exception as e:
            print(f"[SCANNER] {symbol} fetch failed: {e}")
            import pandas as pd
            return pd.DataFrame()

    def _get_current_price(self, symbol: str) -> dict:
        """Get current price for a symbol."""
        import requests
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            response = requests.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code != 200:
                return {}
            data = response.json()
            return {
                "price": float(data["lastPrice"]),
                "change_24h": float(data["priceChangePercent"]),
                "volume_24h": float(data["quoteVolume"]),
            }
        except Exception:
            return {}

    def analyze_coin(self, coin: dict) -> dict:
        """Run full analysis on a single coin."""
        symbol = coin["symbol"]

        # Fetch data
        df = self._fetch_klines_for_symbol(symbol, interval="4h", limit=200)
        if df.empty or len(df) < 50:
            return {"symbol": symbol, "error": "Insufficient data"}

        price_info = self._get_current_price(symbol)

        # TA
        df_with_ta = self.ta.calculate_indicators(df)
        ta_result = self.ta.analyze(df)
        ta_score = ta_result.get("score", 0)

        # SMC
        smc_result = self.smc.generate_smc_analysis(df_with_ta)
        smc_score = smc_result.get("score", 0)

        # Expert (basic)
        market_structure = self.expert.detect_market_structure(df_with_ta)
        trend_phase = self.expert.calculate_trend_phase(df_with_ta)
        divergence = self.expert.detect_divergence(df_with_ta)

        # Combine scores (TA 35% + SMC 50% + Phase bonus 15%)
        combined_score = ta_score * 0.35 + smc_score * 0.50

        # Phase bonus
        phase_bonus = 0
        if trend_phase.get("bias") == "BULLISH":
            phase_bonus = 0.1
        elif trend_phase.get("bias") == "BEARISH":
            phase_bonus = -0.1
        combined_score += phase_bonus * 0.15

        # Divergence boost
        if divergence.get("detected"):
            if divergence["type"] == "BULLISH":
                combined_score += 0.1
            else:
                combined_score -= 0.1

        # Clamp
        combined_score = max(-1.0, min(1.0, combined_score))

        # Determine signal
        if combined_score >= 0.6:
            signal = "STRONG BUY"
        elif combined_score >= 0.3:
            signal = "BUY"
        elif combined_score <= -0.6:
            signal = "STRONG SELL"
        elif combined_score <= -0.3:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Calculate basic R/R levels
        risk_reward = self.expert.calculate_risk_reward(df_with_ta, signal)

        return {
            "symbol": symbol,
            "name": coin["name"],
            "price": price_info.get("price", 0),
            "change_24h": price_info.get("change_24h", 0),
            "volume_24h": price_info.get("volume_24h", 0),
            "ta_score": round(ta_score, 4),
            "smc_score": round(smc_score, 4),
            "combined_score": round(combined_score, 4),
            "abs_score": abs(combined_score),
            "signal": signal,
            "structure": market_structure.get("structure", "UNKNOWN"),
            "phase": trend_phase.get("phase", "UNKNOWN"),
            "divergence": divergence.get("type") if divergence.get("detected") else None,
            "risk_reward": risk_reward,
        }

    def scan_top_coins(self) -> dict:
        """
        Scan all top coins and return ranked results.
        Returns top 3 with strongest signals.
        """
        print(f"[SCANNER] Scanning {len(self.TOP_COINS)} coins...")
        results = []

        for coin in self.TOP_COINS:
            try:
                result = self.analyze_coin(coin)
                if "error" not in result:
                    results.append(result)
            except Exception as e:
                print(f"[SCANNER] {coin['symbol']} error: {e}")
                continue

        # Sort by absolute score (strongest signals first)
        results.sort(key=lambda x: x["abs_score"], reverse=True)

        # Top 3 with actionable signals (not HOLD)
        actionable = [r for r in results if r["signal"] != "HOLD"]
        top_3_strongest = actionable[:3] if actionable else results[:3]

        # Stats
        bullish_count = sum(1 for r in results if r["combined_score"] > 0.1)
        bearish_count = sum(1 for r in results if r["combined_score"] < -0.1)

        return {
            "timestamp": datetime.now().isoformat(),
            "scanned_count": len(results),
            "all_results": results,
            "top_3_strongest": top_3_strongest,
            "market_overview": {
                "bullish_coins": bullish_count,
                "bearish_coins": bearish_count,
                "neutral_coins": len(results) - bullish_count - bearish_count,
                "market_bias": "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL",
            },
        }
