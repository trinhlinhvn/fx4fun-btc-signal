"""
News Sentiment Analysis Module
Fetches crypto news and analyzes sentiment to generate trading signals.
"""
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from config import NEWS_API_KEY, NEWS_CONFIG


class NewsSentimentAnalyzer:
    """Fetches and analyzes crypto news sentiment from multiple free sources."""

    NEWSAPI_URL = "https://newsapi.org/v2/everything"

    # Free sources - no API key required
    FREE_SOURCES = {
        "free_crypto_news": "https://api.free-crypto-news.com/api/v1/news",
        "coindesk_rss": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph_rss": "https://cointelegraph.com/rss",
        "bitcoin_magazine_rss": "https://bitcoinmagazine.com/.rss/full/",
    }

    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.config = NEWS_CONFIG

    def fetch_news_newsapi(self) -> list:
        """Fetch news from NewsAPI.org."""
        if not self.api_key:
            return []

        from_date = (datetime.now() - timedelta(hours=self.config["lookback_hours"])).isoformat()

        params = {
            "q": " OR ".join(self.config["keywords"]),
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": self.config["max_articles"],
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(self.NEWSAPI_URL, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "published_at": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                }
                for a in articles
                if a.get("title")
            ]
        except requests.RequestException as e:
            print(f"[WARNING] NewsAPI fetch failed: {e}")
            return []

    def fetch_news_cryptopanic(self) -> list:
        """Fetch news from RSS feeds of major crypto outlets (always free, fast)."""
        return self._fetch_rss_news()

    def _fetch_rss_news(self) -> list:
        """Fetch news from RSS feeds of major crypto outlets (always free)."""
        import xml.etree.ElementTree as ET

        articles = []
        rss_urls = [
            self.FREE_SOURCES["coindesk_rss"],
            self.FREE_SOURCES["cointelegraph_rss"],
        ]

        for rss_url in rss_urls:
            try:
                response = requests.get(rss_url, timeout=8, headers={
                    "User-Agent": "BTC-Signal-Bot/2.0"
                })
                if response.status_code != 200:
                    continue

                root = ET.fromstring(response.content)
                # Standard RSS format
                for item in root.findall(".//item")[:10]:
                    title = item.findtext("title", "")
                    desc = item.findtext("description", "")
                    pub_date = item.findtext("pubDate", "")
                    link = item.findtext("link", "")

                    # Only keep BTC-relevant articles
                    text_lower = (title + desc).lower()
                    if any(kw in text_lower for kw in ["bitcoin", "btc", "crypto", "market"]):
                        articles.append({
                            "title": title,
                            "description": desc[:200],
                            "source": rss_url.split("/")[2],
                            "published_at": pub_date,
                            "url": link,
                        })
            except Exception as e:
                print(f"[WARNING] RSS fetch failed ({rss_url}): {e}")
                continue

        return articles[:self.config["max_articles"]]

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text using TextBlob.
        Returns polarity score: -1 (very negative) to 1 (very positive).
        """
        if not text:
            return 0.0
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def get_keyword_boost(self, text: str) -> float:
        """
        Advanced keyword-based sentiment for crypto domain.
        Weighted by impact strength (major catalyst > minor news).
        """
        text_lower = text.lower()
        boost = 0.0

        # High impact bullish (major catalysts)
        high_bullish = [
            "etf approved", "etf approval", "spot etf", "institutional adoption",
            "legal tender", "strategic reserve", "halving", "supply shock",
        ]
        # Medium impact bullish
        mid_bullish = [
            "surge", "rally", "bullish", "breakout", "all-time high", "ath",
            "adoption", "institutional", "whale buying", "accumulation",
            "inflow", "buying pressure", "short squeeze",
        ]
        # Low impact bullish
        low_bullish = [
            "upgrade", "partnership", "integration", "milestone", "recovery",
        ]

        # High impact bearish
        high_bearish = [
            "ban", "banned", "sec lawsuit", "exchange hack", "insolvency",
            "bankruptcy", "rug pull", "ponzi", "collapse",
        ]
        # Medium impact bearish
        mid_bearish = [
            "crash", "dump", "bearish", "regulation", "crackdown",
            "sell-off", "liquidation", "fear", "outflow", "exploit",
            "vulnerability", "investigation",
        ]
        # Low impact bearish
        low_bearish = [
            "concern", "warning", "delay", "postpone", "uncertainty",
        ]

        for kw in high_bullish:
            if kw in text_lower:
                boost += 0.3
        for kw in mid_bullish:
            if kw in text_lower:
                boost += 0.15
        for kw in low_bullish:
            if kw in text_lower:
                boost += 0.08

        for kw in high_bearish:
            if kw in text_lower:
                boost -= 0.3
        for kw in mid_bearish:
            if kw in text_lower:
                boost -= 0.15
        for kw in low_bearish:
            if kw in text_lower:
                boost -= 0.08

        return max(-0.7, min(0.7, boost))

    def analyze(self) -> dict:
        """
        Fetch news and perform sentiment analysis.
        Sort by absolute impact (highest impact first).
        Returns top 10 most impactful articles.
        """
        # Try NewsAPI first, fallback to RSS
        articles = self.fetch_news_newsapi()
        source_used = "NewsAPI"

        if not articles:
            articles = self.fetch_news_cryptopanic()
            source_used = "CryptoPanic"

        if not articles:
            return {
                "score": 0.0,
                "articles_analyzed": 0,
                "source": "none",
                "details": [],
                "note": "No news data available.",
            }

        details = []
        total_sentiment = 0.0

        for article in articles:
            text = f"{article['title']}. {article['description']}"
            base_sentiment = self.analyze_sentiment(text)
            keyword_boost = self.get_keyword_boost(text)
            final_sentiment = max(-1.0, min(1.0, base_sentiment + keyword_boost))

            details.append({
                "title": article["title"][:100],
                "description": article.get("description", "")[:300],
                "source": article["source"],
                "sentiment": round(final_sentiment, 3),
                "impact": round(abs(final_sentiment), 3),
                "published_at": article["published_at"],
                "url": article.get("url", ""),
            })
            total_sentiment += final_sentiment

        # Sort by impact (absolute sentiment value) — most impactful first
        details.sort(key=lambda x: x["impact"], reverse=True)

        avg_sentiment = total_sentiment / len(articles) if articles else 0.0

        return {
            "score": round(avg_sentiment, 4),
            "articles_analyzed": len(articles),
            "source": source_used,
            "positive_count": sum(1 for d in details if d["sentiment"] > 0.1),
            "negative_count": sum(1 for d in details if d["sentiment"] < -0.1),
            "neutral_count": sum(1 for d in details if -0.1 <= d["sentiment"] <= 0.1),
            "details": details[:10],  # Top 10 most impactful
        }
