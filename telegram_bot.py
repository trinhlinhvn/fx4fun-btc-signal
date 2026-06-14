"""
Telegram Bot Module
Gửi tín hiệu trading BTC về Telegram với format chuyên nghiệp.
Hỗ trợ:
- Gửi alert tự động khi có signal mới
- Command /signal để lấy signal ngay
- Command /status để xem trạng thái bot
- Command /train để retrain ML models
"""
import asyncio
import logging
from datetime import datetime
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from signal_engine import SignalEngine

logger = logging.getLogger(__name__)


class TelegramSignalBot:
    """Telegram bot gửi tín hiệu trading BTC."""

    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.engine = SignalEngine()
        self.bot = Bot(token=self.token) if self.token else None
        self.last_signal = None

    def format_signal_message(self, result: dict) -> str:
        """Format signal result thành message Telegram đẹp."""
        if "error" in result:
            return f"❌ Error: {result['error']}"

        signal = result["signal"]
        score = result["final_score"]
        confidence = result["confidence"]
        price_info = result.get("current_price", {})
        price = price_info.get("price", 0)
        change_24h = price_info.get("change_24h", 0)

        # Signal emoji
        if "STRONG BUY" in signal:
            signal_emoji = "🟢🟢🚀"
        elif "BUY" in signal:
            signal_emoji = "🟢📈"
        elif "STRONG SELL" in signal:
            signal_emoji = "🔴🔴💥"
        elif "SELL" in signal:
            signal_emoji = "🔴📉"
        else:
            signal_emoji = "🟡⏸️"

        # Components
        components = result["components"]
        ta = components["technical_analysis"]
        news = components["news_sentiment"]
        ml = components["ml_prediction"]

        # Expert analysis
        expert = result.get("expert_analysis", {})
        risk_reward = result.get("risk_reward", {})
        narratives = expert.get("narratives", [])
        risk = expert.get("risk_level", {})
        phase = expert.get("trend_phase", {})
        bias = expert.get("overall_bias", {})

        # Build message
        msg = f"""
{signal_emoji} *BTC SIGNAL: {signal}*
{'━' * 30}

💰 *Price:* ${price:,.2f} ({change_24h:+.2f}% 24h)
📊 *Score:* {score:+.4f} (-1.0 to +1.0)
🎯 *Confidence:* {confidence}

{'─' * 25}
*📈 PHÂN TÍCH KỸ THUẬT*
• RSI: {ta['details'].get('rsi', {}).get('signal', 'N/A')} ({ta['details'].get('rsi', {}).get('value', 0):.1f})
• MACD: {ta['details'].get('macd', {}).get('signal', 'N/A')}
• EMA Cross: {ta['details'].get('ema_cross', {}).get('signal', 'N/A')}
• Bollinger: {ta['details'].get('bollinger', {}).get('signal', 'N/A')}
• TA Score: {ta['score']:+.4f}

{'─' * 25}
*📰 TIN TỨC & SENTIMENT*
• Score: {news['score']:+.4f}
• Articles: {news['articles_analyzed']} ({news['source']})

{'─' * 25}
*🤖 ML/AI PREDICTION*
• Prediction: {ml['prediction']}
• Confidence: {ml['ml_confidence']:.0%}
• Score: {ml['score']:+.4f}
"""

        # Expert section
        if phase:
            msg += f"""
{'─' * 25}
*🧠 EXPERT ANALYSIS*
• Phase: {phase.get('phase', 'N/A')}
• Bias: {bias.get('direction', 'N/A')} ({bias.get('strength', 0)}%)
• Risk: {risk.get('level', 'N/A')}
"""

        # Risk/Reward
        if risk_reward and risk_reward.get("position_type") != "NO TRADE":
            futures = risk_reward.get("futures", {})
            msg += f"""
{'─' * 25}
*💎 TRADE SETUP ({risk_reward['position_type']})*
• Entry: ${risk_reward['entry']:,.2f}
• Stop Loss: ${risk_reward['stop_loss']:,.2f}
• TP1: ${risk_reward['take_profit_1']:,.2f}
• TP2: ${risk_reward['take_profit_2']:,.2f}
• TP3: ${risk_reward['take_profit_3']:,.2f}
• R:R Ratio: {risk_reward['risk_reward_ratio']}:1
• Risk: {risk_reward['risk_percent']:.2f}%
"""
            if futures:
                msg += f"""
{'─' * 25}
*⚡ FUTURES x{futures.get('leverage', 10)}*
• Margin: ${futures.get('margin_usd', 0):,.0f}
• Position Size: ${futures.get('position_size_usd', 0):,.0f}
• Qty: {futures.get('quantity_btc', 0):.6f} BTC
• Liquidation: ${futures.get('liquidation_price', 0):,.2f}
• SL Loss: ${futures.get('sl_pnl_usd', 0):,.2f} (ROE {futures.get('sl_roe_pct', 0):+.1f}%)
"""
                for tp in futures.get("take_profits", []):
                    msg += f"• {tp['target']}: +${tp['pnl_usd']:,.2f} (ROE {tp['roe_pct']:+.1f}%)\n"

        # Multi-timeframe
        mtf = result.get("multi_timeframe", {})
        if mtf and mtf.get("recommendation"):
            msg += f"""
{'─' * 25}
*📐 MULTI-TIMEFRAME*
• H4: {mtf.get('h4', {}).get('trend', 'N/A')} ({mtf.get('h4', {}).get('phase', '')})
• H1: {mtf.get('h1', {}).get('trend', 'N/A')} ({mtf.get('h1', {}).get('phase', '')})
• {mtf.get('recommendation', '')}
"""

        if not risk_reward or risk_reward.get("position_type") == "NO TRADE":
            if risk_reward:
                msg += f"""
{'─' * 25}
*💎 TRADE SETUP*
• {risk_reward.get('reason', 'No clear setup')}
• {risk_reward.get('suggestion', '')}
"""

        # Key narratives (top 3)
        if narratives:
            msg += f"\n{'─' * 25}\n*📝 KEY INSIGHTS*\n"
            for n in narratives[:4]:
                msg += f"• {n}\n"

        # Risk factors
        if risk.get("factors"):
            msg += f"\n⚠️ *Risk Factors:* {', '.join(risk['factors'][:3])}"

        msg += f"""

{'━' * 30}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⚠️ _Not financial advice. DYOR._
"""
        return msg

    async def send_signal(self, result: dict = None):
        """Gửi signal về Telegram."""
        if not self.bot or not self.chat_id:
            logger.warning("Telegram not configured (missing token or chat_id)")
            return False

        if result is None:
            result = self.engine.generate_signal()

        message = self.format_signal_message(result)
        self.last_signal = result

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown",
            )
            logger.info("Signal sent to Telegram successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            # Try without markdown if parsing fails
            try:
                plain_message = message.replace("*", "").replace("_", "")
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=plain_message,
                )
                return True
            except Exception as e2:
                logger.error(f"Failed to send plain message: {e2}")
                return False

    async def send_alert(self, alert_type: str, message: str):
        """Gửi alert nhanh (không phải full signal)."""
        if not self.bot or not self.chat_id:
            return False

        alert_msg = f"🚨 *BTC ALERT — {alert_type}*\n\n{message}\n\n⏰ {datetime.now().strftime('%H:%M:%S')}"

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=alert_msg,
                parse_mode="Markdown",
            )
            return True
        except Exception as e:
            logger.error(f"Alert send failed: {e}")
            return False


def create_telegram_app() -> Application:
    """
    Tạo Telegram Application với các commands.
    Dùng khi muốn chạy bot interactive (nhận lệnh từ user).
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

    engine = SignalEngine()

    async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command."""
        await update.message.reply_text("🔄 Analyzing BTC... Please wait.")
        result = engine.generate_signal()
        bot = TelegramSignalBot()
        message = bot.format_signal_message(result)
        await update.message.reply_text(message, parse_mode="Markdown")

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        status = f"""
🤖 *BTC Signal Bot Status*
• Status: Running ✅
• ML Trained: {'Yes' if engine.is_ml_trained else 'No'}
• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Version: 2.0 (Expert Edition)
"""
        await update.message.reply_text(status, parse_mode="Markdown")

    async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /train command."""
        await update.message.reply_text("🧠 Training ML models... This may take a few minutes.")
        result = engine.train_ml_models()
        if "error" in result:
            await update.message.reply_text(f"❌ Training failed: {result['error']}")
        else:
            msg = "✅ *Training Complete!*\n"
            if "xgboost" in result:
                msg += f"• XGBoost accuracy: {result['xgboost'].get('avg_accuracy', 'N/A')}\n"
            if "lstm" in result:
                msg += f"• LSTM val accuracy: {result['lstm'].get('val_accuracy', 'N/A')}\n"
            await update.message.reply_text(msg, parse_mode="Markdown")

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """
🪙 *BTC Trading Signal Bot*

Commands:
/signal — Lấy tín hiệu trading ngay
/status — Xem trạng thái bot
/train — Train lại ML models
/help — Hiện menu này

Bot tự động gửi signal mỗi 5 phút (khi chạy ở chế độ auto).
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("train", cmd_train))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))

    return app
