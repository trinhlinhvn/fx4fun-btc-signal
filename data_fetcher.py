"""
Data Fetcher Module — Binance Futures API
==========================================
Lấy dữ liệu BTC/USDT từ Binance Futures (FREE, không cần API key).
Hỗ trợ multi-timeframe: 1m, 5m, 15m, 1H, 4H, 1D
Có volume thật, rate limit cao (1200 req/min).
"""
import time
import requests
import pandas as pd
from datetime import datetime, timedelta


class BTCDataFetcher:
    """
    Fetches BTC/USDT data from Binance Futures API.
    Free, no API key required for market data.
    """

    # Binance Futures (USDT-M) — dùng cho Futures trading
    BASE_URL = "https://fapi.binance.com"

    # Supported intervals
    INTERVALS = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
    }

    # Smart cache
    _cache = {}
    _cache_ttl = {
        "current_price": 30,       # 30 seconds
        "klines": 120,             # 2 minutes
        "market_data": 300,        # 5 minutes
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })
        self.symbol = "BTCUSDT"

    def _get_cached(self, key: str):
        """Get data from cache if still valid."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            ttl_key = key.split(":")[0]
            ttl = self._cache_ttl.get(ttl_key, 60)
            if time.time() - timestamp < ttl:
                return data
        return None

    def _set_cache(self, key: str, data):
        """Store data in cache."""
        self._cache[key] = (data, time.time())

    def get_current_price(self) -> dict:
        """Get current BTC/USDT price from Binance Futures ticker."""
        cached = self._get_cached("current_price")
        if cached:
            return cached

        url = f"{self.BASE_URL}/fapi/v1/ticker/24hr"
        params = {"symbol": self.symbol}

        try:
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            result = {
                "price": float(data["lastPrice"]),
                "change_24h": float(data["priceChangePercent"]),
                "volume_24h": float(data["quoteVolume"]),
                "high_24h": float(data["highPrice"]),
                "low_24h": float(data["lowPrice"]),
                "trades_24h": int(data["count"]),
                "timestamp": datetime.now().isoformat(),
            }
            self._set_cache("current_price", result)
            return result
        except Exception as e:
            print(f"[ERROR] Binance price fetch failed: {e}")
            if "current_price" in self._cache:
                return self._cache["current_price"][0]
            return {}

    def get_klines(self, interval: str = "4h", limit: int = 200) -> pd.DataFrame:
        """
        Get OHLCV klines (candlestick) data from Binance Futures.
        
        Args:
            interval: 1m, 5m, 15m, 1h, 4h, 1d, 1w
            limit: number of candles (max 1500)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        cache_key = f"klines:{interval}:{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None and not cached.empty:
            return cached

        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(limit, 1500),
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = df[col].astype(float)
            df["trades"] = df["trades"].astype(int)

            # Keep only OHLCV + useful columns
            df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades"]]
            df = df.sort_index()

            self._set_cache(cache_key, df)
            return df

        except Exception as e:
            print(f"[ERROR] Binance klines fetch failed ({interval}): {e}")
            if cache_key in self._cache:
                return self._cache[cache_key][0]
            return pd.DataFrame()

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Backward-compatible method. Maps days to appropriate interval.
        Used by existing code that calls get_historical_data(days=30).
        """
        if days <= 1:
            return self.get_klines(interval="15m", limit=96)  # 1 day of 15m
        elif days <= 7:
            return self.get_klines(interval="1h", limit=168)  # 7 days of 1h
        elif days <= 30:
            return self.get_klines(interval="4h", limit=180)  # 30 days of 4h
        else:
            return self.get_klines(interval="1d", limit=min(days, 365))

    def get_h1_data(self, limit: int = 100) -> pd.DataFrame:
        """Get H1 (1-hour) klines."""
        return self.get_klines(interval="1h", limit=limit)

    def get_h4_data(self, limit: int = 200) -> pd.DataFrame:
        """Get H4 (4-hour) klines."""
        return self.get_klines(interval="4h", limit=limit)

    def get_market_data(self) -> dict:
        """Get additional market data from Binance including funding rate."""
        cached = self._get_cached("market_data")
        if cached:
            return cached

        try:
            price_data = self.get_current_price()
            if not price_data:
                return {}

            # Get funding rate (critical for futures trading)
            funding_rate = 0.0
            try:
                fr_url = f"{self.BASE_URL}/fapi/v1/fundingRate"
                fr_resp = self.session.get(fr_url, params={"symbol": self.symbol, "limit": 1}, timeout=5)
                if fr_resp.status_code == 200:
                    fr_data = fr_resp.json()
                    if fr_data:
                        funding_rate = float(fr_data[0]["fundingRate"]) * 100  # Convert to %
            except Exception:
                pass

            # Get open interest (market positioning)
            open_interest = 0.0
            try:
                oi_url = f"{self.BASE_URL}/fapi/v1/openInterest"
                oi_resp = self.session.get(oi_url, params={"symbol": self.symbol}, timeout=5)
                if oi_resp.status_code == 200:
                    oi_data = oi_resp.json()
                    open_interest = float(oi_data.get("openInterest", 0))
            except Exception:
                pass

            # Calculate 7d and 30d changes from klines
            daily = self.get_klines(interval="1d", limit=30)
            price_change_7d = 0
            price_change_30d = 0

            if not daily.empty and len(daily) >= 7:
                current = float(daily.iloc[-1]["close"])
                price_7d_ago = float(daily.iloc[-7]["close"])
                price_change_7d = (current - price_7d_ago) / price_7d_ago * 100

                if len(daily) >= 30:
                    price_30d_ago = float(daily.iloc[-30]["close"])
                    price_change_30d = (current - price_30d_ago) / price_30d_ago * 100
                elif len(daily) >= 2:
                    price_30d_ago = float(daily.iloc[0]["close"])
                    price_change_30d = (current - price_30d_ago) / price_30d_ago * 100

            result = {
                "high_24h": price_data.get("high_24h", 0),
                "low_24h": price_data.get("low_24h", 0),
                "volume_24h": price_data.get("volume_24h", 0),
                "trades_24h": price_data.get("trades_24h", 0),
                "price_change_7d": round(price_change_7d, 2),
                "price_change_30d": round(price_change_30d, 2),
                "funding_rate": round(funding_rate, 4),
                "open_interest_btc": round(open_interest, 2),
            }
            self._set_cache("market_data", result)
            return result

        except Exception as e:
            print(f"[ERROR] Market data fetch failed: {e}")
            if "market_data" in self._cache:
                return self._cache["market_data"][0]
            return {}
