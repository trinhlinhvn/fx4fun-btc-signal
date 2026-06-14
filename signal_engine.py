"""
Signal Engine
Combines Technical Analysis + News Sentiment + ML Prediction + Expert Analysis
to generate final trading signals like a professional trader.
"""
from datetime import datetime
from config import SIGNAL_WEIGHTS, SIGNAL_THRESHOLDS, ML_CONFIG
from data_fetcher import BTCDataFetcher
from technical_analysis import TechnicalAnalyzer
from news_sentiment import NewsSentimentAnalyzer
from ml_predictor import EnsemblePredictor
from expert_analysis import ExpertAnalyzer
from smc_ict_analysis import SMCICTAnalyzer


class SignalEngine:
    """
    Core engine that combines all analysis sources into a unified signal.

    Signal scoring:
    - Technical Analysis: RSI, MACD, EMA, Bollinger Bands
    - News Sentiment: NLP analysis of crypto news
    - ML Prediction: XGBoost + LSTM ensemble
    - Expert Analysis: Market structure, S/R, divergence, risk management

    Final score range: -1.0 (Strong Sell) to +1.0 (Strong Buy)
    """

    def __init__(self):
        self.data_fetcher = BTCDataFetcher()
        self.ta_analyzer = TechnicalAnalyzer()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.ml_predictor = EnsemblePredictor()
        self.expert = ExpertAnalyzer()
        self.smc = SMCICTAnalyzer()
        self.weights = SIGNAL_WEIGHTS
        self.thresholds = SIGNAL_THRESHOLDS
        self.is_ml_trained = False
        self._last_signal = None
        self._signal_hold_count = 0

    def train_ml_models(self) -> dict:
        """Train ML models on historical data."""
        print("[ENGINE] Fetching training data...")
        df = self.data_fetcher.get_historical_data(days=ML_CONFIG["training_days"])

        if df.empty or len(df) < 50:
            return {"error": "Not enough historical data for training"}

        # Add TA indicators to training data
        df = self.ta_analyzer.calculate_indicators(df)

        print("[ENGINE] Training ML ensemble...")
        result = self.ml_predictor.train(df)
        self.is_ml_trained = True

        return result

    def generate_signal(self) -> dict:
        """
        Generate comprehensive trading signal with expert-level analysis.
        Returns detailed breakdown of all analysis components.
        """
        timestamp = datetime.now().isoformat()

        # 1. Fetch price data (multi-timeframe from Binance Futures)
        print("[ENGINE] Fetching price data...")
        current_price = self.data_fetcher.get_current_price()
        # H4 data: real 4-hour candles from Binance
        historical_df = self.data_fetcher.get_h4_data(limit=200)
        # H1 data: real 1-hour candles from Binance
        historical_h1 = self.data_fetcher.get_h1_data(limit=100)

        if historical_df.empty:
            return {"error": "Cannot fetch price data", "timestamp": timestamp}

        # 2. Technical Analysis
        print("[ENGINE] Running technical analysis...")
        ta_result = self.ta_analyzer.analyze(historical_df)
        ta_score = ta_result.get("score", 0)

        # 3. News Sentiment
        print("[ENGINE] Analyzing news sentiment...")
        news_result = self.news_analyzer.analyze()
        news_score = news_result.get("score", 0)

        # 4. ML Prediction
        print("[ENGINE] Running ML prediction...")
        historical_with_ta = self.ta_analyzer.calculate_indicators(historical_df.copy())
        ml_result = self.ml_predictor.predict(historical_with_ta)
        ml_score = ml_result.get("score", 0)

        # 5. Expert Analysis
        print("[ENGINE] Generating expert analysis...")
        expert_commentary = self.expert.generate_expert_commentary(
            historical_with_ta, ta_result, ml_result, news_result
        )

        # 5b. Multi-timeframe analysis (H1 + H4)
        print("[ENGINE] Multi-timeframe analysis (H1 + H4)...")
        h1_with_ta = self.ta_analyzer.calculate_indicators(historical_h1.copy()) if not historical_h1.empty else historical_h1
        mtf_result = self.expert.analyze_multi_timeframe(h1_with_ta, historical_with_ta)

        # 5c. SMC / ICT Analysis
        print("[ENGINE] SMC/ICT analysis...")
        smc_result = self.smc.generate_smc_analysis(historical_with_ta)
        smc_score = smc_result.get("score", 0)

        # 6. Dynamic weight allocation
        # Nếu source nào không có data → redistribute weight cho source khác
        has_ml = not ("error" in ml_result or not self.is_ml_trained)
        has_news = news_result.get("articles_analyzed", 0) > 0

        if has_ml and has_news:
            effective_weights = {"technical": 0.25, "sentiment": 0.15, "ml_prediction": 0.25, "smc": 0.35}
        elif has_ml and not has_news:
            effective_weights = {"technical": 0.30, "sentiment": 0.0, "ml_prediction": 0.30, "smc": 0.40}
        elif not has_ml and has_news:
            effective_weights = {"technical": 0.35, "sentiment": 0.20, "ml_prediction": 0.0, "smc": 0.45}
        else:
            effective_weights = {"technical": 0.40, "sentiment": 0.0, "ml_prediction": 0.0, "smc": 0.60}

        ml_note = None if has_ml else "ML not available. Weights redistributed."

        final_score = (
            ta_score * effective_weights["technical"] +
            news_score * effective_weights["sentiment"] +
            ml_score * effective_weights["ml_prediction"] +
            smc_score * effective_weights["smc"]
        )

        # 7. Expert bias adjustment
        # Nếu expert analysis phát hiện divergence hoặc BOS, điều chỉnh score
        final_score = self._apply_expert_adjustment(final_score, expert_commentary)

        # 8. Determine signal (with anti-whipsaw logic)
        signal = self._score_to_signal(final_score)

        # Anti-whipsaw: nếu signal vừa flip ngược, cần score mạnh hơn để confirm
        if self._last_signal and self._last_signal != signal:
            if "BUY" in self._last_signal and "SELL" in signal:
                # Flip from BUY to SELL — cần score < -0.4 (thay vì -0.3) để confirm
                if final_score > -0.4:
                    signal = "HOLD 🟡"
            elif "SELL" in self._last_signal and "BUY" in signal:
                if final_score < 0.4:
                    signal = "HOLD 🟡"

        self._last_signal = signal

        # 9. Risk/Reward calculation
        risk_reward = self.expert.calculate_risk_reward(historical_with_ta, signal)

        # 10. Calculate confidence (includes SMC)
        agreement = self._calculate_agreement(ta_score, news_score, ml_score, smc_score)

        return {
            "timestamp": timestamp,
            "signal": signal,
            "final_score": round(final_score, 4),
            "confidence": agreement,
            "current_price": current_price,
            "components": {
                "technical_analysis": {
                    "score": ta_score,
                    "weight": effective_weights["technical"],
                    "weighted_score": round(ta_score * effective_weights["technical"], 4),
                    "details": ta_result.get("signals", {}),
                },
                "news_sentiment": {
                    "score": news_score,
                    "weight": effective_weights["sentiment"],
                    "weighted_score": round(news_score * effective_weights["sentiment"], 4),
                    "articles_analyzed": news_result.get("articles_analyzed", 0),
                    "source": news_result.get("source", "none"),
                    "top_articles": news_result.get("details", [])[:5],
                },
                "ml_prediction": {
                    "score": ml_score,
                    "weight": effective_weights["ml_prediction"],
                    "weighted_score": round(ml_score * effective_weights["ml_prediction"], 4),
                    "prediction": ml_result.get("prediction", "N/A"),
                    "ml_confidence": ml_result.get("confidence", 0),
                    "models_used": ml_result.get("models_used", []),
                    "note": ml_note,
                },
            },
            "expert_analysis": expert_commentary,
            "multi_timeframe": mtf_result,
            "smc_ict": smc_result,
            "risk_reward": risk_reward,
            "market_context": self._safe_get_market_data(),
        }

    def _safe_get_market_data(self) -> dict:
        """Get market data without blocking if rate limited."""
        try:
            return self.data_fetcher.get_market_data()
        except Exception:
            return {}

    def _apply_expert_adjustment(self, score: float, expert: dict) -> float:
        """
        Điều chỉnh score dựa trên expert analysis.
        Divergence và Break of Structure là tín hiệu mạnh.
        """
        adjustment = 0.0

        # Divergence boost
        divergence = expert.get("divergence", {})
        if divergence.get("detected"):
            if divergence["type"] == "BULLISH" and divergence.get("strength") == "STRONG":
                adjustment += 0.15
            elif divergence["type"] == "BEARISH" and divergence.get("strength") == "STRONG":
                adjustment -= 0.15
            elif divergence["type"] == "BULLISH":
                adjustment += 0.08
            elif divergence["type"] == "BEARISH":
                adjustment -= 0.08

        # Break of Structure boost
        bos = expert.get("market_structure", {}).get("break_of_structure")
        if bos:
            if "BULLISH" in bos:
                adjustment += 0.1
            elif "BEARISH" in bos:
                adjustment -= 0.1

        # Clamp to valid range
        return max(-1.0, min(1.0, score + adjustment))

    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score to trading signal."""
        if score >= self.thresholds["strong_buy"]:
            return "STRONG BUY 🟢🟢"
        elif score >= self.thresholds["buy"]:
            return "BUY 🟢"
        elif score <= self.thresholds["strong_sell"]:
            return "STRONG SELL 🔴🔴"
        elif score <= self.thresholds["sell"]:
            return "SELL 🔴"
        else:
            return "HOLD 🟡"

    def _calculate_agreement(self, ta: float, news: float, ml: float, smc: float = 0) -> str:
        """Calculate confidence based on all source agreement."""
        scores = [s for s in [ta, news, ml, smc] if s != 0]  # Exclude zero (no data)
        if not scores:
            return "LOW (no data)"
        
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        total = len(scores)

        # Numeric confidence
        if positive == total or negative == total:
            return f"HIGH ({positive}/{total} sources agree)"
        elif positive >= total * 0.6 or negative >= total * 0.6:
            return f"MEDIUM ({max(positive,negative)}/{total} sources agree)"
        else:
            return f"LOW ({max(positive,negative)}/{total} — sources diverge)"
