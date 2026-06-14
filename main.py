"""
BTC Trading Signal Bot — Expert Edition
==========================================
Combines Technical Analysis + News Sentiment + ML/AI Prediction + Expert Analysis
to generate professional-grade trading signals for Bitcoin.

Usage:
    python main.py                  # Run once in terminal
    python main.py --loop           # Run continuously every 5 minutes
    python main.py --train          # Train ML models first, then run
    python main.py --web            # Start web dashboard (http://localhost:5000)
    python main.py --telegram       # Run with Telegram alerts
    python main.py --telegram-bot   # Start interactive Telegram bot
    python main.py --all            # Web + Telegram + Loop (production mode)
"""
import sys
import time
import asyncio
import threading
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from signal_engine import SignalEngine
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

console = Console()


def display_signal(result: dict):
    """Display trading signal in a beautiful terminal UI with expert analysis."""
    if "error" in result:
        console.print(f"[red]❌ Error: {result['error']}[/red]")
        return

    console.print()
    signal_text = result["signal"]
    score = result["final_score"]
    confidence = result["confidence"]

    # Color based on signal
    if "BUY" in signal_text:
        color = "green"
    elif "SELL" in signal_text:
        color = "red"
    else:
        color = "yellow"

    # Main Signal Panel
    price_info = result.get("current_price", {})
    price = price_info.get("price", 0)
    change_24h = price_info.get("change_24h", 0)
    change_color = "green" if change_24h >= 0 else "red"

    header = f"""
[bold {color}]{'═' * 50}
         SIGNAL: {signal_text}
{'═' * 50}[/bold {color}]

[white]BTC Price:[/white] [bold]${price:,.2f}[/bold]  [{change_color}]({change_24h:+.2f}% 24h)[/{change_color}]
[white]Score:[/white] [bold {color}]{score:+.4f}[/bold {color}] (range: -1.0 to +1.0)
[white]Confidence:[/white] {confidence}
[white]Time:[/white] {result['timestamp'][:19]}
"""
    console.print(Panel(header, title="🪙 BTC TRADING SIGNAL — Expert Edition", border_style=color))

    # Components Table
    components = result["components"]

    table = Table(title="📊 Signal Components", box=box.ROUNDED, show_header=True)
    table.add_column("Source", style="cyan", width=20)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Weight", justify="center", width=10)
    table.add_column("Contribution", justify="center", width=14)
    table.add_column("Details", width=35)

    # Technical Analysis row
    ta = components["technical_analysis"]
    ta_details = ", ".join(
        f"{k}: {v['signal']}" for k, v in ta["details"].items()
    ) if ta["details"] else "No data"
    ta_color = "green" if ta["score"] > 0 else "red" if ta["score"] < 0 else "white"
    table.add_row(
        "📈 Technical",
        f"[{ta_color}]{ta['score']:+.4f}[/{ta_color}]",
        f"{ta['weight']:.0%}",
        f"[{ta_color}]{ta['weighted_score']:+.4f}[/{ta_color}]",
        ta_details[:35],
    )

    # News Sentiment row
    news = components["news_sentiment"]
    news_detail = f"{news['articles_analyzed']} articles ({news['source']})"
    news_color = "green" if news["score"] > 0 else "red" if news["score"] < 0 else "white"
    table.add_row(
        "📰 News Sentiment",
        f"[{news_color}]{news['score']:+.4f}[/{news_color}]",
        f"{news['weight']:.0%}",
        f"[{news_color}]{news['weighted_score']:+.4f}[/{news_color}]",
        news_detail,
    )

    # ML Prediction row
    ml = components["ml_prediction"]
    ml_detail = f"{ml['prediction']} (conf: {ml['ml_confidence']:.0%})"
    if ml.get("note"):
        ml_detail = ml["note"][:35]
    ml_color = "green" if ml["score"] > 0 else "red" if ml["score"] < 0 else "white"
    table.add_row(
        "🤖 ML/AI Predict",
        f"[{ml_color}]{ml['score']:+.4f}[/{ml_color}]",
        f"{ml['weight']:.0%}",
        f"[{ml_color}]{ml['weighted_score']:+.4f}[/{ml_color}]",
        ml_detail[:35],
    )

    console.print(table)

    # Expert Analysis Panel
    expert = result.get("expert_analysis", {})
    if expert:
        narratives = expert.get("narratives", [])
        if narratives:
            expert_panel = "\n".join(f"  {n}" for n in narratives)
            bias = expert.get("overall_bias", {})
            bias_text = f"\n\n  [bold]Overall Bias: {bias.get('direction', 'N/A')} ({bias.get('strength', 0)}%)[/bold]"
            bias_text += f"\n  Reasons: {', '.join(bias.get('reasons', []))}"

            risk = expert.get("risk_level", {})
            risk_text = f"\n  Risk Level: {risk.get('level', 'N/A')}"
            if risk.get("factors"):
                risk_text += f" — {', '.join(risk['factors'][:3])}"
            risk_text += f"\n  💡 {risk.get('recommendation', '')}"

            console.print(Panel(
                expert_panel + bias_text + risk_text,
                title="🧠 Expert Analysis",
                border_style="cyan",
            ))

    # Trade Setup
    risk_reward = result.get("risk_reward", {})
    if risk_reward:
        if risk_reward.get("position_type") == "NO TRADE":
            console.print(Panel(
                f"  {risk_reward.get('reason', '')}\n  💡 {risk_reward.get('suggestion', '')}",
                title="💎 Trade Setup",
                border_style="yellow",
            ))
        else:
            rr_table = Table(title=f"💎 Trade Setup — {risk_reward['position_type']}", box=box.SIMPLE)
            rr_table.add_column("", style="dim", width=15)
            rr_table.add_column("Price", justify="right", width=15)
            rr_table.add_column("Note", width=30)

            rr_table.add_row("Entry", f"[blue]${risk_reward['entry']:,.2f}[/blue]", "")
            rr_table.add_row("Stop Loss", f"[red]${risk_reward['stop_loss']:,.2f}[/red]", f"Risk: {risk_reward['risk_percent']:.2f}%")
            rr_table.add_row("TP1", f"[green]${risk_reward['take_profit_1']:,.2f}[/green]", "Conservative")
            rr_table.add_row("TP2", f"[green]${risk_reward['take_profit_2']:,.2f}[/green]", "3:1 R:R")
            rr_table.add_row("TP3", f"[green]${risk_reward['take_profit_3']:,.2f}[/green]", "5:1 R:R")
            rr_table.add_row("R:R Ratio", f"[yellow]{risk_reward['risk_reward_ratio']}:1[/yellow]", "")
            rr_table.add_row("Position Size", "", risk_reward.get("position_size_suggestion", ""))

            console.print(rr_table)

    # Market Context
    market = result.get("market_context", {})
    if market:
        context_table = Table(title="🌍 Market Context", box=box.SIMPLE)
        context_table.add_column("Metric", style="dim")
        context_table.add_column("Value", justify="right")

        if market.get("high_24h"):
            context_table.add_row("24h High", f"${market['high_24h']:,.0f}")
        if market.get("low_24h"):
            context_table.add_row("24h Low", f"${market['low_24h']:,.0f}")
        if market.get("price_change_7d"):
            c = "green" if market["price_change_7d"] > 0 else "red"
            context_table.add_row("7d Change", f"[{c}]{market['price_change_7d']:+.1f}%[/{c}]")
        if market.get("price_change_30d"):
            c = "green" if market["price_change_30d"] > 0 else "red"
            context_table.add_row("30d Change", f"[{c}]{market['price_change_30d']:+.1f}%[/{c}]")
        if market.get("sentiment_up"):
            context_table.add_row(
                "Community Sentiment",
                f"↑{market['sentiment_up']:.0f}% / ↓{market['sentiment_down']:.0f}%"
            )
        console.print(context_table)

    console.print(
        "\n[dim]⚠️  Disclaimer: This is for educational purposes only. "
        "Not financial advice. Always DYOR.[/dim]\n"
    )


def run_with_telegram(engine, result):
    """Send signal to Telegram."""
    from telegram_bot import TelegramSignalBot

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        console.print("[yellow]⚠️  Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env[/yellow]")
        return

    bot = TelegramSignalBot()
    asyncio.run(bot.send_signal(result))
    console.print("[green]✅ Signal sent to Telegram[/green]")


def main():
    """Main entry point."""
    console.print("[bold cyan]🪙 BTC Trading Signal Bot — Expert Edition[/bold cyan]")
    console.print("[dim]Technical Analysis + News Sentiment + ML/AI + Expert Analysis[/dim]\n")

    engine = SignalEngine()

    # Train ML models if requested
    if "--train" in sys.argv:
        console.print("[bold yellow]🧠 Training ML models...[/bold yellow]")
        console.print("[dim]This may take a few minutes on first run.[/dim]\n")
        with console.status("[bold green]Training in progress..."):
            train_result = engine.train_ml_models()

        if "error" in train_result:
            console.print(f"[red]Training error: {train_result['error']}[/red]")
        else:
            console.print("[green]✅ ML models trained successfully![/green]")
            if "xgboost" in train_result:
                xgb = train_result["xgboost"]
                console.print(f"   XGBoost accuracy: {xgb.get('avg_accuracy', 'N/A')}")
            if "lstm" in train_result:
                lstm = train_result["lstm"]
                console.print(f"   LSTM val accuracy: {lstm.get('val_accuracy', 'N/A')}")
            console.print()

    # Web Dashboard mode
    if "--web" in sys.argv or "--all" in sys.argv:
        from web_dashboard import run_dashboard
        console.print("[bold cyan]🌐 Starting Web Dashboard at http://localhost:5000[/bold cyan]")
        if "--all" in sys.argv:
            # Run web in a thread, telegram in another
            web_thread = threading.Thread(target=run_dashboard, kwargs={"port": 5000}, daemon=True)
            web_thread.start()
            console.print("[green]✅ Web Dashboard started[/green]")
        else:
            run_dashboard(port=5000, debug=True)
            return

    # Telegram Bot interactive mode
    if "--telegram-bot" in sys.argv:
        from telegram_bot import create_telegram_app
        console.print("[bold cyan]🤖 Starting Telegram Bot (interactive mode)...[/bold cyan]")
        console.print("[dim]Send /signal to the bot to get trading signals.[/dim]")
        app = create_telegram_app()
        app.run_polling()
        return

    # Loop mode with optional Telegram
    if "--loop" in sys.argv or "--all" in sys.argv:
        interval = 300  # 5 minutes
        use_telegram = "--telegram" in sys.argv or "--all" in sys.argv
        console.print(f"[cyan]Running in loop mode (every {interval // 60} min). Press Ctrl+C to stop.[/cyan]")
        if use_telegram:
            console.print("[cyan]Telegram alerts: ON[/cyan]")
        console.print()

        try:
            while True:
                with console.status("[bold green]Analyzing BTC..."):
                    result = engine.generate_signal()
                display_signal(result)

                if use_telegram:
                    run_with_telegram(engine, result)

                console.print(f"[dim]Next update in {interval // 60} minutes...[/dim]\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user.[/yellow]")
    else:
        # Single run
        with console.status("[bold green]Analyzing BTC..."):
            result = engine.generate_signal()
        display_signal(result)

        if "--telegram" in sys.argv:
            run_with_telegram(engine, result)


if __name__ == "__main__":
    main()
