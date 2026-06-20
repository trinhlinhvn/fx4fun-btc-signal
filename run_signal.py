"""
GitHub Actions runner — lightweight, chỉ gửi Telegram khi có signal.
Chạy mỗi 5 phút bởi GitHub Actions cron.
"""
import asyncio
import os

# Ensure env vars available
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "")
os.environ.setdefault("TELEGRAM_CHAT_ID", "")

from signal_engine import SignalEngine
from telegram_bot import TelegramSignalBot


async def main():
    engine = SignalEngine()
    result = engine.generate_signal()

    signal = result.get("signal", "")
    score = result.get("final_score", 0)
    rr = result.get("risk_reward", {})
    rr_ratio = rr.get("risk_reward_ratio", 0) if rr else 0

    print(f"Signal: {signal} | Score: {score:+.4f} | R:R: {rr_ratio}")

    # Gửi Telegram nếu có signal actionable (BUY/SELL) và score >= 0.2
    is_actionable = "BUY" in signal or "SELL" in signal
    is_strong = abs(score) >= 0.2 and rr_ratio >= 1.5

    if is_actionable and is_strong:
        print("✅ Signal detected — sending Telegram...")
        bot = TelegramSignalBot()
        ok = await bot.send_signal(result)
        print(f"Telegram: {'sent' if ok else 'failed'}")
    else:
        print(f"⏸️ No alert: {'not actionable' if not is_actionable else f'score {abs(score):.2f}<0.2 or R:R {rr_ratio}<1.5'}")


if __name__ == "__main__":
    asyncio.run(main())
