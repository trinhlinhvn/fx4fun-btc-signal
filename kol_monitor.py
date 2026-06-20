"""
KOL & Politician Twitter Monitor
================================
Monitor tweets from high-impact accounts via FREE sources:
- Nitter RSS (Twitter mirror, no API key needed)
- LunarCrush free tier (social sentiment)
- Whale Alert (on-chain big movements)

Tier 1 (HIGHEST impact - emergency alert):
- Donald Trump (@realDonaldTrump)
- Elon Musk (@elonmusk)
- Michael Saylor (@saylor)

Tier 2 (HIGH impact):
- CZ Binance (@cz_binance)
- Brian Armstrong (@brian_armstrong, Coinbase CEO)
- Vitalik Buterin (@VitalikButerin)

Tier 3 (MEDIUM impact - aggregators):
- @WhaleAlert
- @CoinDesk breaking news
"""
import re
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional


class KOLMonitor:
    """Monitor high-impact crypto influencers via free RSS sources."""

    # Multiple Nitter instances for redundancy (some go down)
    NITTER_INSTANCES = [
        "https://nitter.net",
        "https://nitter.poast.org",
        "https://nitter.privacydev.net",
        "https://nitter.kavin.rocks",
    ]

    # KOL accounts with impact tier
    KOL_ACCOUNTS = {
        # Tier 1 - Massive market mover
        "realDonaldTrump": {"tier": 1, "name": "Trump", "weight": 4.0},
        "elonmusk": {"tier": 1, "name": "Musk", "weight": 4.0},
        "saylor": {"tier": 1, "name": "Saylor", "weight": 3.5},

        # Tier 2 - Industry leaders
        "cz_binance": {"tier": 2, "name": "CZ", "weight": 3.0},
        "brian_armstrong": {"tier": 2, "name": "Coinbase CEO", "weight": 2.5},
        "VitalikButerin": {"tier": 2, "name": "Vitalik", "weight": 2.0},

        # Tier 3 - Aggregators
        "WhaleAlert": {"tier": 3, "name": "Whale Alert", "weight": 2.0},
    }

    # Crypto-related keywords (must match to consider relevant)
    CRYPTO_KEYWORDS = [
        "bitcoin", "btc", "crypto", "cryptocurrency", "ethereum", "eth",
        "blockchain", "satoshi", "sats", "hodl", "moon", "dump",
        "binance", "coinbase", "etf", "halving", "mining",
        "decentralized", "stablecoin", "defi", "nft",
        "fed", "sec", "regulation", "ban", "legal tender",
    ]

    # High-impact keywords (multiply sentiment)
    BULLISH_TRIGGERS = [
        "buy bitcoin", "btc reserve", "strategic reserve", "etf approved",
        "adopting btc", "treasury", "halving", "supply shock",
        "institutional", "legal tender", "approved", "all-time high",
    ]
    BEARISH_TRIGGERS = [
        "sec lawsuit", "ban crypto", "shut down", "fraud",
        "ponzi", "investigation", "delisted", "hacked",
        "insolvent", "bankruptcy", "dump",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (BTC-Signal-Bot/3.0)"
        })
        self._nitter_index = 0
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def _get_nitter_url(self) -> str:
        """Get current Nitter instance, with rotation on failure."""
        return self.NITTER_INSTANCES[self._nitter_index % len(self.NITTER_INSTANCES)]

    def _rotate_nitter(self):
        """Switch to next Nitter instance."""
        self._nitter_index += 1

    def fetch_tweets(self, username: str, max_tweets: int = 5) -> list:
        """
        Fetch recent tweets from a user via Nitter RSS.
        Returns list of dicts: {text, time, url}
        """
        cache_key = f"tweets:{username}"
        if cache_key in self._cache:
            data, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return data

        # Try multiple Nitter instances
        for attempt in range(len(self.NITTER_INSTANCES)):
            base = self._get_nitter_url()
            url = f"{base}/{username}/rss"

            try:
                response = self.session.get(url, timeout=5)
                if response.status_code != 200:
                    self._rotate_nitter()
                    continue

                root = ET.fromstring(response.content)
                tweets = []
                for item in root.findall(".//item")[:max_tweets]:
                    title = item.findtext("title", "")
                    desc = item.findtext("description", "")
                    pub_date = item.findtext("pubDate", "")
                    link = item.findtext("link", "")

                    # Strip HTML tags from description
                    text = re.sub(r"<[^>]*>", "", desc).strip() or title

                    tweets.append({
                        "text": text,
                        "title": title,
                        "time": pub_date,
                        "url": link,
                        "username": username,
                    })

                self._cache[cache_key] = (tweets, time.time())
                return tweets

            except Exception as e:
                self._rotate_nitter()
                continue

        return []

    def is_recent(self, time_str: str, hours: int = 6) -> bool:
        """Check if tweet was published in last N hours."""
        try:
            # Parse RSS pubDate format: "Wed, 14 Jun 2026 12:00:00 GMT"
            from email.utils import parsedate_to_datetime
            tweet_time = parsedate_to_datetime(time_str)
            now = datetime.now(timezone.utc)
            return (now - tweet_time) < timedelta(hours=hours)
        except Exception:
            return False

    def is_crypto_related(self, text: str) -> bool:
        """Check if tweet is crypto-related."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.CRYPTO_KEYWORDS)

    def calculate_tweet_sentiment(self, text: str, kol_weight: float) -> dict:
        """
        Calculate sentiment for a single tweet.
        Apply KOL weight multiplier (Tier 1 KOLs = stronger signal).
        """
        text_lower = text.lower()

        bullish_score = sum(0.3 for trigger in self.BULLISH_TRIGGERS if trigger in text_lower)
        bearish_score = sum(0.3 for trigger in self.BEARISH_TRIGGERS if trigger in text_lower)

        # Use TextBlob for general sentiment as backup
        try:
            from textblob import TextBlob
            polarity = TextBlob(text).sentiment.polarity
        except Exception:
            polarity = 0.0

        # Combine: keyword triggers are dominant, polarity is secondary
        raw_sentiment = (bullish_score - bearish_score) + polarity * 0.3
        # Clamp to [-1, 1]
        raw_sentiment = max(-1.0, min(1.0, raw_sentiment))

        # Apply KOL weight
        weighted_impact = raw_sentiment * kol_weight

        return {
            "sentiment": round(raw_sentiment, 3),
            "weighted_impact": round(weighted_impact, 3),
            "bullish_triggers": [t for t in self.BULLISH_TRIGGERS if t in text_lower],
            "bearish_triggers": [t for t in self.BEARISH_TRIGGERS if t in text_lower],
        }

    def scan_all_kols(self, hours: int = 6) -> dict:
        """
        Scan all KOL accounts for recent crypto-related tweets.
        Returns aggregated sentiment + emergency alerts.
        """
        all_tweets = []
        emergency_alerts = []
        total_weighted_sentiment = 0.0
        total_weight = 0.0
        tier1_active = False

        for username, info in self.KOL_ACCOUNTS.items():
            tweets = self.fetch_tweets(username, max_tweets=5)

            for tweet in tweets:
                if not self.is_recent(tweet["time"], hours=hours):
                    continue
                if not self.is_crypto_related(tweet["text"]):
                    continue

                sentiment_data = self.calculate_tweet_sentiment(tweet["text"], info["weight"])

                tweet_record = {
                    "username": username,
                    "kol_name": info["name"],
                    "tier": info["tier"],
                    "weight": info["weight"],
                    "text": tweet["text"][:200],
                    "time": tweet["time"],
                    "url": tweet["url"],
                    "sentiment": sentiment_data["sentiment"],
                    "weighted_impact": sentiment_data["weighted_impact"],
                    "triggers": sentiment_data["bullish_triggers"] + sentiment_data["bearish_triggers"],
                }
                all_tweets.append(tweet_record)

                total_weighted_sentiment += sentiment_data["weighted_impact"]
                total_weight += info["weight"]

                # Emergency alert: Tier 1 KOL with strong sentiment
                if info["tier"] == 1 and abs(sentiment_data["sentiment"]) > 0.3:
                    direction = "BULLISH 🟢" if sentiment_data["sentiment"] > 0 else "BEARISH 🔴"
                    emergency_alerts.append({
                        "kol": info["name"],
                        "direction": direction,
                        "sentiment": sentiment_data["sentiment"],
                        "text": tweet["text"][:150],
                        "time": tweet["time"],
                        "url": tweet["url"],
                    })
                    tier1_active = True

        # Aggregate score (-1 to 1)
        if total_weight > 0:
            aggregate_score = total_weighted_sentiment / total_weight
            aggregate_score = max(-1.0, min(1.0, aggregate_score))
        else:
            aggregate_score = 0.0

        # Sort tweets by impact
        all_tweets.sort(key=lambda x: abs(x["weighted_impact"]), reverse=True)

        return {
            "score": round(aggregate_score, 4),
            "total_tweets": len(all_tweets),
            "tier1_active": tier1_active,
            "emergency_alerts": emergency_alerts,
            "top_tweets": all_tweets[:10],
            "scanned_accounts": len(self.KOL_ACCOUNTS),
            "lookback_hours": hours,
        }

    def get_summary_text(self, scan_result: dict) -> str:
        """Generate human-readable summary for display."""
        if scan_result["total_tweets"] == 0:
            return "Không có tweet crypto nào trong 6h qua."

        lines = []
        lines.append(f"📡 Scan {scan_result['scanned_accounts']} KOLs trong {scan_result['lookback_hours']}h:")
        lines.append(f"• {scan_result['total_tweets']} tweets crypto-related")
        lines.append(f"• Aggregate sentiment: {scan_result['score']:+.3f}")

        if scan_result["tier1_active"]:
            lines.append(f"• ⚡ TIER-1 KOL ACTIVE!")

        if scan_result["emergency_alerts"]:
            lines.append("\n🚨 Emergency alerts:")
            for a in scan_result["emergency_alerts"][:3]:
                lines.append(f"• [{a['kol']}] {a['direction']}: {a['text'][:80]}")

        return "\n".join(lines)
