"""
On-chain & Exchange Flow Monitor
=================================
Track institutional movements & whale activity via FREE sources:

1. Whale Alert RSS — large transactions (>$1M usually) realtime
2. Binance exchange flow — open interest changes, funding rate spikes
3. Long/Short ratio — retail vs whale positioning
4. Liquidation events — forced moves often signal reversal

These signals often precede price moves by 5-30 minutes.
"""
import re
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional


class OnChainMonitor:
    """Monitor on-chain movements & exchange flows."""

    WHALE_ALERT_RSS = "https://whale-alert.io/feed.rss"
    BINANCE_FAPI = "https://fapi.binance.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._cache = {}
        self._cache_ttl = 120  # 2 minutes

    def _cached(self, key: str, ttl: int = None):
        ttl = ttl or self._cache_ttl
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < ttl:
                return data
        return None

    def _set_cache(self, key: str, data):
        self._cache[key] = (data, time.time())

    def fetch_whale_alerts(self, max_items: int = 20) -> list:
        """
        Fetch recent whale alerts (large crypto transactions).
        Whale Alert provides RSS feed for free.
        """
        cached = self._cached("whale_alerts")
        if cached is not None:
            return cached

        try:
            response = self.session.get(self.WHALE_ALERT_RSS, timeout=8)
            if response.status_code != 200:
                return []

            root = ET.fromstring(response.content)
            alerts = []

            for item in root.findall(".//item")[:max_items]:
                title = item.findtext("title", "")
                desc = item.findtext("description", "")
                pub_date = item.findtext("pubDate", "")
                link = item.findtext("link", "")

                # Parse: "1,234 BTC ($75,000,000) transferred from unknown wallet to Binance"
                amount_match = re.search(r"([\d,]+(?:\.\d+)?)\s*(\w+)", title)
                usd_match = re.search(r"\$([0-9,]+(?:\.\d+)?)", title)

                amount = 0
                symbol = ""
                usd_value = 0

                if amount_match:
                    try:
                        amount = float(amount_match.group(1).replace(",", ""))
                        symbol = amount_match.group(2).upper()
                    except ValueError:
                        pass

                if usd_match:
                    try:
                        usd_value = float(usd_match.group(1).replace(",", ""))
                    except ValueError:
                        pass

                # Detect direction (sentiment)
                title_lower = title.lower()
                direction = "NEUTRAL"
                signal_impact = 0.0

                if "to binance" in title_lower or "to coinbase" in title_lower or "to kraken" in title_lower or "to bybit" in title_lower:
                    # Move TO exchange = preparing to sell = bearish
                    direction = "BEARISH"
                    signal_impact = -0.3 if usd_value > 50_000_000 else -0.15
                elif "from binance" in title_lower or "from coinbase" in title_lower or "from kraken" in title_lower:
                    # Move FROM exchange = accumulating = bullish
                    direction = "BULLISH"
                    signal_impact = 0.3 if usd_value > 50_000_000 else 0.15
                elif "burned" in title_lower or "burn" in title_lower:
                    direction = "BULLISH"
                    signal_impact = 0.2
                elif "minted" in title_lower:
                    direction = "BEARISH"
                    signal_impact = -0.1

                alerts.append({
                    "title": title,
                    "amount": amount,
                    "symbol": symbol,
                    "usd_value": usd_value,
                    "direction": direction,
                    "impact": signal_impact,
                    "time": pub_date,
                    "url": link,
                })

            self._set_cache("whale_alerts", alerts)
            return alerts

        except Exception as e:
            print(f"[WHALE] Fetch failed: {e}")
            return []

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> dict:
        """Get current funding rate from Binance Futures."""
        try:
            url = f"{self.BINANCE_FAPI}/fapi/v1/premiumIndex"
            response = self.session.get(url, params={"symbol": symbol}, timeout=5)
            if response.status_code != 200:
                return {}
            data = response.json()
            funding_rate = float(data.get("lastFundingRate", 0)) * 100
            return {
                "symbol": symbol,
                "funding_rate": round(funding_rate, 4),
                "mark_price": float(data.get("markPrice", 0)),
                "index_price": float(data.get("indexPrice", 0)),
            }
        except Exception:
            return {}

    def get_long_short_ratio(self, symbol: str = "BTCUSDT") -> dict:
        """
        Get long/short ratio from Binance Futures.
        Top traders position ratio = smart money positioning.
        """
        try:
            # Top traders by position
            url = f"{self.BINANCE_FAPI}/futures/data/topLongShortPositionRatio"
            params = {"symbol": symbol, "period": "1h", "limit": 1}
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return {}
            data = response.json()
            if not data:
                return {}

            latest = data[0]
            long_account = float(latest["longAccount"])
            short_account = float(latest["shortAccount"])
            ratio = float(latest["longShortRatio"])

            # Interpret
            if ratio > 1.5:
                bias = "VERY_BULLISH"
                signal_impact = 0.2
            elif ratio > 1.2:
                bias = "BULLISH"
                signal_impact = 0.1
            elif ratio < 0.7:
                bias = "VERY_BEARISH"
                signal_impact = -0.2
            elif ratio < 0.85:
                bias = "BEARISH"
                signal_impact = -0.1
            else:
                bias = "NEUTRAL"
                signal_impact = 0.0

            # Contrarian check: if too one-sided, expect reversal
            contrarian_warning = ""
            if ratio > 2.5:
                contrarian_warning = "⚠️ Quá nhiều long → có thể bị flush"
                signal_impact -= 0.15
            elif ratio < 0.4:
                contrarian_warning = "⚠️ Quá nhiều short → có thể bị squeeze"
                signal_impact += 0.15

            return {
                "symbol": symbol,
                "long_pct": round(long_account * 100, 1),
                "short_pct": round(short_account * 100, 1),
                "ratio": round(ratio, 2),
                "bias": bias,
                "impact": round(signal_impact, 3),
                "warning": contrarian_warning,
            }
        except Exception:
            return {}

    def get_open_interest_change(self, symbol: str = "BTCUSDT") -> dict:
        """
        Get OI change to detect new positions opening/closing.
        Rising OI + rising price = strong trend
        Rising OI + falling price = aggressive shorts
        """
        try:
            url = f"{self.BINANCE_FAPI}/futures/data/openInterestHist"
            params = {"symbol": symbol, "period": "1h", "limit": 24}
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return {}
            data = response.json()
            if len(data) < 2:
                return {}

            current_oi = float(data[-1]["sumOpenInterest"])
            prev_oi = float(data[0]["sumOpenInterest"])
            change_pct = (current_oi - prev_oi) / prev_oi * 100 if prev_oi > 0 else 0

            return {
                "symbol": symbol,
                "current_oi": current_oi,
                "change_24h_pct": round(change_pct, 2),
                "interpretation": (
                    "📈 Mở thêm position mạnh" if change_pct > 5 else
                    "📉 Đóng position lớn" if change_pct < -5 else
                    "➖ OI ổn định"
                ),
            }
        except Exception:
            return {}

    def scan_all(self, symbol: str = "BTCUSDT") -> dict:
        """
        Comprehensive on-chain scan combining:
        - Whale alerts (last 1h)
        - Funding rate
        - Long/Short ratio
        - Open Interest change
        """
        whale_alerts = self.fetch_whale_alerts(max_items=20)
        funding = self.get_funding_rate(symbol)
        ls_ratio = self.get_long_short_ratio(symbol)
        oi_change = self.get_open_interest_change(symbol)

        # Filter recent BTC-related whales (last hour)
        symbol_base = symbol.replace("USDT", "").replace("USD", "")
        recent_whales = []
        whale_score = 0.0
        whale_count_to_exchange = 0
        whale_count_from_exchange = 0

        for alert in whale_alerts[:10]:
            if symbol_base.upper() in alert.get("symbol", "").upper() or alert.get("usd_value", 0) > 10_000_000:
                recent_whales.append(alert)
                whale_score += alert["impact"]
                if alert["direction"] == "BEARISH":
                    whale_count_to_exchange += 1
                elif alert["direction"] == "BULLISH":
                    whale_count_from_exchange += 1

        # Aggregate score
        total_score = 0.0
        components = 0

        if whale_score != 0:
            total_score += max(-1.0, min(1.0, whale_score))
            components += 1

        if ls_ratio.get("impact"):
            total_score += ls_ratio["impact"]
            components += 1

        # Funding rate signal
        funding_impact = 0.0
        if funding.get("funding_rate") is not None:
            fr = funding["funding_rate"]
            if fr > 0.05:  # Very high positive — too many longs
                funding_impact = -0.15
            elif fr < -0.05:  # Very negative — too many shorts (bullish contrarian)
                funding_impact = 0.15
            elif fr > 0.02:
                funding_impact = -0.05
            elif fr < -0.02:
                funding_impact = 0.05
            total_score += funding_impact
            components += 1

        avg_score = total_score / components if components > 0 else 0
        avg_score = max(-1.0, min(1.0, avg_score))

        # Build narratives
        narratives = []
        if whale_count_from_exchange > whale_count_to_exchange:
            narratives.append(f"🟢 {whale_count_from_exchange} whale move OFF exchange (accumulating)")
        elif whale_count_to_exchange > whale_count_from_exchange:
            narratives.append(f"🔴 {whale_count_to_exchange} whale move TO exchange (preparing to sell)")

        if ls_ratio.get("ratio"):
            narratives.append(f"📊 L/S ratio: {ls_ratio['ratio']} ({ls_ratio['bias']})")

        if funding.get("funding_rate") is not None:
            fr_text = f"Funding {funding['funding_rate']:+.4f}%"
            if abs(funding_impact) >= 0.1:
                fr_text += " ⚠️ EXTREME"
            narratives.append(f"💰 {fr_text}")

        if oi_change.get("change_24h_pct") is not None:
            narratives.append(f"📈 OI 24h: {oi_change['change_24h_pct']:+.1f}%")

        return {
            "score": round(avg_score, 4),
            "symbol": symbol,
            "whale_alerts": recent_whales[:5],
            "whale_count_to_exchange": whale_count_to_exchange,
            "whale_count_from_exchange": whale_count_from_exchange,
            "funding_rate": funding,
            "long_short": ls_ratio,
            "open_interest": oi_change,
            "narratives": narratives,
        }
