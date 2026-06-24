# Swing Trading Signal Dashboard

An autonomous, data-driven Streamlit dashboard designed to scan, backtest, and track swing trading signals on the Nifty 500 universe using the **Donchian Channel breakout strategy**.

---

## ⚠️ Disclaimer

> [!WARNING]
> **Educational Purpose Only**: This application is built for educational and research purposes only. The strategy, indicators, and historical performance statistics presented here do not constitute financial advice, buy/sell recommendations, or investment advice of any kind. Trading in financial markets involves substantial risk of loss, and you are solely responsible for any investment decisions, financial losses, or profits resulting from the use of this software. We are not responsible or liable for any actions taken or decisions made based on this application.

---

## Features

1. **📊 Daily Scanner**:
   - Downloads live price data in parallel for Nifty 50, Nifty Next 50, Nifty 200, and Nifty 500 stocks directly from the official NSE archives (cached for 1 week).
   - Generates daily **Buy Signals** (touch/breakout of the lower channel band) and **Sell Signals** (touch/breakout of the upper channel band).
   - Shows connection fetch status logs and labels signals as `🟢 New Entry` or `⏳ Already Open` to prevent pyramiding.
   - Allows exporting scanned signals to CSV.

2. **📈 Strategy Backtester**:
   - Runs historical backtests on one, multiple, or all Nifty 500 tickers for various durations (1 to 10 years).
   - Supports configurable position size and flat transaction commissions.
   - Caches backtest runs in session state.
   - Displays KPIs: Total Trades, Win Rate %, Average Return per Trade %, realized profit, cumulative return plot, and detailed trade logs.
   - Includes an on-demand **Ticker-Level Stats** breakdown table.
   - Easily loads only fresh scanned Buy signals with a single click.

3. **💼 Autonomous Position Tracker**:
   - Automatically tracks open positions historically without manual portfolio data input.
   - Highlights exit alerts (`🚨 EXIT SIGNAL (Sell on Next Open)`) when held stocks generate sell signals today.

---

## Installation & Setup

1. **Activate the Virtual Environment**:
   ```bash
   source env/bin/activate
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501/` in your browser.
