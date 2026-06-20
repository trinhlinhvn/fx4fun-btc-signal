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
        """
        Telegram message ngắn gọn, focus:
        - Direction (LONG/SHORT)
        - Entry Zone (vùng giá)
        - SL, TP1/TP2/TP3
        - 3 lý do ngắn gọn
        """
        if "error" in result:
            return f"❌ Error: {result['error']}"

        signal = result["signal"]
        score = result["final_score"]
        price_info = result.get("current_price", {})
        price = price_info.get("price", 0)
        change_24h = price_info.get("change_24h", 0)
        risk_reward = result.get("risk_reward", {})
        trade_reasons = result.get("trade_reasons", [])
        fund_mgmt = result.get("fund_management", {})
        position_sizing = fund_mgmt.get("position_sizing", {})

        is_actionable = "BUY" in signal or "SELL" in signal

        if "STRONG BUY" in signal:
            header = "🟢🟢 STRONG LONG 🚀"
        elif "BUY" in signal:
            header = "🟢 LONG"
        elif "STRONG SELL" in signal:
            header = "🔴🔴 STRONG SHORT 💥"
        elif "SELL" in signal:
            header = "🔴 SHORT"
        else:
            header = "🟡 HOLD — No Trade"

        # === SHORT & FOCUSED MESSAGE ===
        if is_actionable and risk_reward and "LONG" in str(risk_reward.get("position_type","")) or "SHORT" in str(risk_reward.get("position_type","")):
            pos = risk_reward.get("position_type", "")
            entry = risk_reward.get("entry", 0)
            sl = risk_reward.get("stop_loss", 0)
            tp1 = risk_reward.get("take_profit_1", 0)
            tp2 = risk_reward.get("take_profit_2", 0)
            tp3 = risk_reward.get("take_profit_3", 0)
            rr = risk_reward.get("risk_reward_ratio", 0)

            entry_zone = risk_reward.get("entry_zone", {})
            entry_low = entry_zone.get("low", entry * 0.998)
            entry_high = entry_zone.get("high", entry * 1.002)

            msg = f"""
{header}
━━━━━━━━━━━━━━━━━━━

BTC/USDT | ${price:,.0f} ({change_24h:+.1f}%)

Entry Zone:
${entry_low:,.0f} - ${entry_high:,.0f}

SL: ${sl:,.0f} ({risk_reward.get('risk_percent',0):.1f}%)
TP1: ${tp1:,.0f}
TP2: ${tp2:,.0f}
TP3: ${tp3:,.0f}

R:R {rr}:1 | Score {score:+.3f}
"""
            # 3 Reasons
            if trade_reasons:
                msg += "\nLy do:\n"
                for i, reason in enumerate(trade_reasons[:3], 1):
                    short = reason[:70] + "..." if len(reason) > 70 else reason
                    # Remove emojis for clean text
                    msg += f"{i}. {short}\n"

            msg += f"\n{datetime.now().strftime('%H:%M %d/%m')} | H4/M15 | Refresh 5 min"
            msg += "\nDYOR. Not financial advice."

        else:
            # HOLD message (very short)
            msg = f"""
*{header}*

BTC ${price:,.0f} ({change_24h:+.1f}%)
Score: {score:+.3f}

Không có setup rõ ràng.
Chờ confluence mạnh hơn.

⏰ {datetime.now().strftime('%H:%M %d/%m')}
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
                parse_mode=None,
            )
            logger.info("Signal sent to Telegram successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            try:
                plain_message = message.replace("*", "").replace("_", "").replace("`", "")
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
