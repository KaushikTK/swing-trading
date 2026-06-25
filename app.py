import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import os
import urllib.request
from multi_strategy import run_multi_strategy_backtest, run_pretrade_screening

# Set page configuration
st.set_page_config(
    page_title="Swing Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    .reportview-container {
        background: #0f1116;
    }
    .metric-card {
        background-color: #1a1e27;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00e676;
    }
    .metric-label {
        font-size: 14px;
        color: #8a909d;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(WORKSPACE_DIR, "ind_nifty500list.csv")

# 1. Load tickers from NSE website with 1 week cache
@st.cache_data(ttl=604800)
def load_tickers():
    files = {
        'Nifty 50': 'ind_nifty50list',
        'Nifty Next 50': 'ind_niftynext50list',
        'Nifty 200': 'ind_nifty200list',
        'Nifty 500': 'ind_nifty500list'
    }
    
    dfs = {}
    fetch_status = {}
    for index_name, filename in files.items():
        try:
            url = f"https://nsearchives.nseindia.com/content/indices/{filename}.csv"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
            )
            with urllib.request.urlopen(req) as response:
                df = pd.read_csv(response)
                df.columns = df.columns.str.strip()
                df['Symbol'] = df['Symbol'].str.strip()
                # Ensure .NS suffix
                df['Symbol'] = df['Symbol'].apply(lambda s: s if s.endswith('.NS') else f"{s}.NS")
                dfs[index_name] = df
                fetch_status[index_name] = "🟢 Successful"
        except Exception as e:
            fetch_status[index_name] = f"🔴 Failed ({e})"
            
    # Combine them, using Nifty 500 as base
    base_df = None
    source = ""
    if 'Nifty 500' in dfs:
        base_df = dfs['Nifty 500'].copy()
        source = "official NSE website (https://nsearchives.nseindia.com/)"
    else:
        # Fallback to local file if available
        if os.path.exists(CSV_PATH):
            base_df = pd.read_csv(CSV_PATH)
            base_df.columns = base_df.columns.str.strip()
            base_df['Symbol'] = base_df['Symbol'].str.strip()
            base_df['Symbol'] = base_df['Symbol'].apply(lambda s: s if s.endswith('.NS') else f"{s}.NS")
            source = f"local file ({CSV_PATH})"
            
    if base_df is None or base_df.empty:
        # Emergency return of empty dataframe
        return pd.DataFrame(columns=['Company Name', 'Industry', 'Symbol', 'Indices']), "Failed to load", fetch_status
        
    # Map symbols to all matching indices
    def get_indices(symbol):
        matched = []
        for index_name, df in dfs.items():
            if symbol in df['Symbol'].values:
                matched.append(index_name)
        if not matched:
            matched.append('Nifty 500')
        return ", ".join(matched)
        
    base_df['Indices'] = base_df['Symbol'].apply(get_indices)
    return base_df, source, fetch_status

tickers_df, data_source, fetch_status = load_tickers()
all_tickers = tickers_df['Symbol'].tolist() if not tickers_df.empty else []

commission_flat = 20.0  # ₹20 per trade transaction

# Session State Initialization
if 'today_buy_signals' not in st.session_state:
    st.session_state['today_buy_signals'] = []
if 'backtest_tickers_selection' not in st.session_state:
    st.session_state['backtest_tickers_selection'] = ['EICHERMOT.NS', 'HEROMOTOCO.NS'] if 'EICHERMOT.NS' in all_tickers else all_tickers[:2]
if 'backtest_tickers_selection_t4' not in st.session_state:
    st.session_state['backtest_tickers_selection_t4'] = ['EICHERMOT.NS', 'HEROMOTOCO.NS'] if 'EICHERMOT.NS' in all_tickers else all_tickers[:2]

def load_buy_signals_to_backtest():
    if st.session_state.get('today_buy_signals'):
        st.session_state['backtest_tickers_selection'] = st.session_state['today_buy_signals']
        st.session_state['load_signals_warning'] = False
    else:
        st.session_state['load_signals_warning'] = True

def load_buy_signals_to_backtest_t4():
    if st.session_state.get('today_buy_signals'):
        st.session_state['backtest_tickers_selection_t4'] = st.session_state['today_buy_signals']
        st.session_state['load_signals_warning_t4'] = False
        st.session_state['show_pretrade_screening'] = True
    else:
        st.session_state['load_signals_warning_t4'] = True
        st.session_state['show_pretrade_screening'] = False

# Caching yfinance downloads for 1 hour
@st.cache_data(ttl=3600)
def download_stock_data(tickers_tuple, period):
    tickers = list(tickers_tuple)
    if not tickers:
        return None
    try:
        # Download in parallel using threads=True
        data = yf.download(tickers, period=period, group_by='ticker', threads=True, progress=False)
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Process data and generate signals
def process_ticker_data(ticker_df, low, high):
    if ticker_df is None or ticker_df.empty or len(ticker_df) < max(low, high):
        return None
    
    df = ticker_df.copy()
    # Reset index to make Date a column
    if 'Date' not in df.columns:
        df = df.reset_index()
    
    # Calculate Donchian Channels in pure pandas to avoid numba/pandas_ta compatibility issues on Python 3.14+
    try:
        df['low'] = df['Low'].rolling(window=low).min()
        df['high'] = df['High'].rolling(window=high).max()
        df['mid'] = (df['low'] + df['high']) / 2
    except Exception as e:
        return None
    
    # Drop rows without channel calculation
    df = df.dropna(subset=['low', 'high']).copy()
    if df.empty:
        return None
        
    # Long and Short signals
    df['long'] = ((df['Close'] == df['low']) | (df['Low'] == df['low'])).astype('int')
    df['short'] = ((df['Close'] == df['high']) | (df['High'] == df['high'])).astype('int')
    
    # Shift to get next day's open
    df['next_day_open_price'] = df['Open'].shift(-1)
    df['next_date'] = df['Date'].shift(-1).astype(str)
    
    return df

# Simulation logic based on processed data
def run_simulation(df_processed, pos_size, commission):
    if df_processed is None or df_processed.empty:
        return []
    
    trades = []
    trade_open = False
    current_trade = None
    
    for idx, row in df_processed.iterrows():
        is_last_row = pd.isna(row['next_day_open_price'])
        
        if trade_open:
            if row['short'] == 1:
                if not is_last_row:
                    exit_price = row['next_day_open_price']
                    exit_date = row['next_date']
                    qty = current_trade['quantity']
                    
                    buy_cost = current_trade['buy_price'] * qty
                    sell_value = exit_price * qty
                    
                    net_buy_cost = buy_cost + commission
                    net_sell_value = sell_value - commission
                    pnl = net_sell_value - net_buy_cost
                    returns_pct = (pnl / net_buy_cost) * 100 if net_buy_cost > 0 else 0.0
                    
                    current_trade.update({
                        'sell_date': exit_date,
                        'sell_price': round(exit_price, 2),
                        'sell_value': round(net_sell_value, 2),
                        'pnl': round(pnl, 2),
                        'returns': round(returns_pct, 2),
                        'holding_period': (pd.to_datetime(exit_date) - pd.to_datetime(current_trade['buy_date'])).days,
                        'status': 'Closed'
                    })
                    trades.append(current_trade)
                    current_trade = None
                    trade_open = False
                else:
                    current_trade['status'] = '🚨 EXIT SIGNAL (Sell on Next Open)'
            else:
                if is_last_row:
                    current_trade['status'] = 'Open'
        else:
            if row['long'] == 1:
                if not is_last_row:
                    buy_price = row['next_day_open_price']
                    buy_date = row['next_date']
                    qty = int(pos_size // buy_price)
                    if qty > 0:
                        current_trade = {
                            'buy_date': buy_date,
                            'buy_price': round(buy_price, 2),
                            'quantity': qty,
                            'buy_value': round(buy_price * qty + commission, 2),
                            'sell_date': None,
                            'sell_price': None,
                            'sell_value': None,
                            'pnl': None,
                            'returns': None,
                            'holding_period': None,
                            'status': 'Open'
                        }
                        trade_open = True
                        
    if trade_open and current_trade is not None:
        trades.append(current_trade)
        
    return trades

# Helper to normalize multi-index dataframe from yfinance
def get_ticker_dataframe(full_data, ticker):
    if full_data is None:
        return None
    try:
        if isinstance(full_data.columns, pd.MultiIndex):
            if ticker in full_data.columns.levels[0]:
                df = full_data[ticker].copy()
                return df.dropna(how='all')
        else:
            # Single ticker download returns flat columns
            return full_data.copy()
    except Exception:
        pass
    return None

# Title and Source Info
st.title("📈 Swing Trading Signal Dashboard")
st.caption(f"**Ticker list source:** {data_source}")

# Show connection details
with st.expander("🌐 Live Index Fetch Status details", expanded=False):
    for idx_name, status in fetch_status.items():
        st.write(f"**{idx_name}**: {status}")

# App Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Daily Signals", "📈 Simple Backtest", "💼 Current Open Trades", "🧪 Advanced Backtest", "ℹ️ About"])

# ----------------- TAB 1: DAILY SIGNALS -----------------
with tab1:
    # Local Settings for Daily Signals
    with st.expander("⚙️ Scan Parameters", expanded=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            low_len = st.number_input("Lower Channel Length", min_value=5, max_value=100, value=20, key="low_len_t1")
        with col_s2:
            high_len = st.number_input("Upper Channel Length", min_value=5, max_value=100, value=20, key="high_len_t1")

    if st.button("🚀 Scan Nifty 500 Tickers", key="scan_btn"):
        with st.spinner("Downloading and scanning latest Nifty 500 prices..."):
            # We need ~60 trading days of data to compute a 20-day Donchian channel correctly
            raw_data = download_stock_data(tuple(all_tickers), "3mo")
            
            if raw_data is not None:
                buy_signals_list = []
                sell_signals_list = []
                scanned_count = 0
                
                for ticker in all_tickers:
                    ticker_raw = get_ticker_dataframe(raw_data, ticker)
                    processed = process_ticker_data(ticker_raw, low_len, high_len)
                    
                    if processed is not None and not processed.empty:
                        scanned_count += 1
                        latest_row = processed.iloc[-1]
                        
                        # Info about ticker
                        meta = tickers_df[tickers_df['Symbol'] == ticker].iloc[0] if ticker in tickers_df['Symbol'].values else None
                        industry = meta['Industry'] if meta is not None else 'Unknown'
                        company = meta['Company Name'] if meta is not None else ticker
                        
                        # Check buy signal today
                        if latest_row['long'] == 1:
                            # Run simulation to see if position is already open
                            sim_trades = run_simulation(processed, 5000, 20.0)
                            is_already_open = False
                            if sim_trades:
                                last_t = sim_trades[-1]
                                if last_t['status'] in ['Open', '🚨 EXIT SIGNAL (Sell on Next Open)']:
                                    is_already_open = True
                                    
                            status_label = "⏳ Already Open" if is_already_open else "🟢 New Entry"
                            
                            buy_signals_list.append({
                                "Symbol": ticker,
                                "Company Name": company,
                                "Industry": industry,
                                "Indices": meta['Indices'] if meta is not None else 'Nifty 500',
                                "Close Price": round(latest_row['Close'], 2),
                                "Donchian Low": round(latest_row['low'], 2),
                                "Date": latest_row['Date'].strftime('%Y-%m-%d'),
                                "Position Status": status_label
                            })
                            
                        # Check sell signal today
                        if latest_row['short'] == 1:
                            sell_signals_list.append({
                                "Symbol": ticker,
                                "Company Name": company,
                                "Industry": industry,
                                "Indices": meta['Indices'] if meta is not None else 'Nifty 500',
                                "Close Price": round(latest_row['Close'], 2),
                                "Donchian High": round(latest_row['high'], 2),
                                "Date": latest_row['Date'].strftime('%Y-%m-%d')
                            })
                
                # Save to session state (only New Entries)
                st.session_state['today_buy_signals'] = [sig['Symbol'] for sig in buy_signals_list if sig['Position Status'] == "🟢 New Entry"]
                
                # Show Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tickers Scanned", scanned_count)
                with col2:
                    st.metric("Buy Signals", len(buy_signals_list))
                with col3:
                    st.metric("Sell Signals", len(sell_signals_list))
                
                # Buy Signals Table
                st.subheader("🟢 Active Buy Signals (Buy on Next Open)")
                if buy_signals_list:
                    buy_df = pd.DataFrame(buy_signals_list)
                    st.dataframe(buy_df, width="stretch")
                    st.download_button("Export Buy Signals", buy_df.to_csv(index=False), "buy_signals.csv", "text/csv")
                else:
                    st.info("No active Buy signals detected today.")
                    
                # Sell Signals Table
                st.subheader("🔴 Active Sell Signals (Sell on Next Open)")
                if sell_signals_list:
                    sell_df = pd.DataFrame(sell_signals_list)
                    st.dataframe(sell_df, width="stretch")
                    st.download_button("Export Sell Signals", sell_df.to_csv(index=False), "sell_signals.csv", "text/csv")
                else:
                    st.info("No active Sell signals detected today.")
            else:
                st.error("Failed to load historical data for Nifty 500.")

# ----------------- TAB 2: BACKTESTING -----------------
with tab2:
    # Local Settings for Backtesting
    with st.expander("⚙️ Strategy & Backtest Parameters", expanded=True):
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            position_size = st.number_input("Position Size (₹)", min_value=100, value=5000, step=500, key="pos_size_t2")
        with col_b2:
            commission_flat = st.number_input("Commission per Transaction (₹)", min_value=0.0, value=20.0, step=1.0, key="commission_t2")
        with col_b3:
            low_len = st.number_input("Lower Channel Length", min_value=5, max_value=100, value=20, key="low_len_t2")
        with col_b4:
            high_len = st.number_input("Upper Channel Length", min_value=5, max_value=100, value=20, key="high_len_t2")

    # Backtest inputs
    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
    with b_col1:
        backtest_selection = st.multiselect("Select Tickers", all_tickers, key="backtest_tickers_selection")
    with b_col2:
        period_selection = st.selectbox("Backtest Duration", ["1y", "2y", "3y", "5y", "10y"], index=1)
    with b_col3:
        run_all_backtest = st.checkbox("Backtest all Nifty 500 tickers")
        
    # Display warning if loaded signals are empty
    if st.session_state.get('load_signals_warning', False):
        st.warning("No Buy signals have been scanned yet today. Please run the scanner in the 'Daily Signals' tab first.")
        st.session_state['load_signals_warning'] = False

    # Button to load today's buy signals
    st.button("📋 Load Today's Buy Signals", on_click=load_buy_signals_to_backtest)
        
    if st.button("🚀 Run Backtest"):
        tickers_to_test = all_tickers if run_all_backtest else backtest_selection
        
        if not tickers_to_test:
            st.warning("Please select at least one ticker.")
        else:
            with st.spinner(f"Running backtest for {len(tickers_to_test)} tickers..."):
                raw_data = download_stock_data(tuple(tickers_to_test), period_selection)
                
                all_sim_trades = []
                for ticker in tickers_to_test:
                    ticker_raw = get_ticker_dataframe(raw_data, ticker)
                    processed = process_ticker_data(ticker_raw, low_len, high_len)
                    if processed is not None:
                        sim_trades = run_simulation(processed, position_size, commission_flat)
                        for t in sim_trades:
                            t['Stock'] = ticker
                        all_sim_trades.extend(sim_trades)
                
                if all_sim_trades:
                    trades_df = pd.DataFrame(all_sim_trades)
                    closed_trades = trades_df[trades_df['status'] == 'Closed'].copy()
                    
                    # Compute statistics
                    total_trades = len(closed_trades)
                    winning_trades = len(closed_trades[closed_trades['returns'] > 0]) if total_trades > 0 else 0
                    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
                    avg_return = closed_trades['returns'].mean() if total_trades > 0 else 0.0
                    avg_hold = closed_trades['holding_period'].mean() if total_trades > 0 else 0.0
                    total_pnl_val = closed_trades['pnl'].sum() if total_trades > 0 else 0.0
                    
                    # Cache in session state
                    st.session_state['backtest_results'] = {
                        'trades_df': trades_df,
                        'closed_trades': closed_trades,
                        'total_trades': total_trades,
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'total_pnl_val': total_pnl_val
                    }
                else:
                    if 'backtest_results' in st.session_state:
                        del st.session_state['backtest_results']
                    st.info("No trades were executed in the given period with the current parameters.")

    # Render backtest results from session state if available
    if 'backtest_results' in st.session_state:
        res = st.session_state['backtest_results']
        trades_df = res['trades_df']
        closed_trades = res['closed_trades']
        
        # Display metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Realized Trades", res['total_trades'])
        m_col2.metric("Win Rate %", f"{res['win_rate']:.2f}%")
        m_col3.metric("Average Return / Trade", f"{res['avg_return']:.2f}%")
        m_col4.metric("Total P&L (₹)", f"₹{res['total_pnl_val']:,.2f}")
        
        # Checkbox for Ticker-Level breakdown
        show_ticker_breakdown = st.checkbox("📊 Show Ticker-Level Stats", value=not run_all_backtest)
        if show_ticker_breakdown:
            st.subheader("Ticker-Level Backtest Breakdown")
            if not closed_trades.empty:
                # Group by stock
                ticker_stats = []
                for ticker, group in closed_trades.groupby('Stock'):
                    tot = len(group)
                    wins = len(group[group['returns'] > 0])
                    wr = (wins / tot) * 100 if tot > 0 else 0.0
                    avg_ret = group['returns'].mean()
                    pnl_val = group['pnl'].sum()
                    avg_hold_val = group['holding_period'].mean()
                    ticker_stats.append({
                        "Ticker": ticker,
                        "Total Trades": tot,
                        "Win Rate %": round(wr, 2),
                        "Avg Return %": round(avg_ret, 2),
                        "Realized P&L (₹)": round(pnl_val, 2),
                        "Avg Hold (Days)": round(avg_hold_val, 1)
                    })
                st.dataframe(pd.DataFrame(ticker_stats), width="stretch")
            else:
                st.info("No closed trades to calculate ticker stats.")
                
        st.subheader("Detailed Trade Logs")
        st.dataframe(trades_df.sort_values(by='buy_date', ascending=False), width="stretch")
        
        # Performance plot
        if not closed_trades.empty:
            closed_trades = closed_trades.sort_values(by='sell_date')
            closed_trades['Cumulative P&L'] = closed_trades['pnl'].cumsum()
            fig = px.line(closed_trades, x='sell_date', y='Cumulative P&L', title="Cumulative Realized Profit/Loss (₹)", markers=True)
            st.plotly_chart(fig, width="stretch")

# ----------------- TAB 3: CURRENT OPEN TRADES -----------------
with tab3:
    # Local Settings for Open Trades
    with st.expander("⚙️ Portfolio Parameters", expanded=True):
        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        with col_o1:
            position_size = st.number_input("Position Size (₹)", min_value=100, value=5000, step=500, key="pos_size_t3")
        with col_o2:
            commission_flat = st.number_input("Commission per Transaction (₹)", min_value=0.0, value=20.0, step=1.0, key="commission_t3")
        with col_o3:
            low_len = st.number_input("Lower Channel Length", min_value=5, max_value=100, value=20, key="low_len_t3")
        with col_o4:
            high_len = st.number_input("Upper Channel Length", min_value=5, max_value=100, value=20, key="high_len_t3")

    if st.button("🔄 Sync Open Trades", key="sync_open_btn"):
        with st.spinner("Downloading and processing portfolio status..."):
            # Fetch last 3 months data for Nifty 500 to evaluate current positions
            raw_data = download_stock_data(tuple(all_tickers), "3mo")
            
            if raw_data is not None:
                open_trades_list = []
                
                for ticker in all_tickers:
                    ticker_raw = get_ticker_dataframe(raw_data, ticker)
                    processed = process_ticker_data(ticker_raw, low_len, high_len)
                    
                    if processed is not None and not processed.empty:
                        sim_trades = run_simulation(processed, position_size, commission_flat)
                        
                        # Find if the last trade is currently Open or has an Exit Signal
                        if sim_trades:
                            last_trade = sim_trades[-1]
                            if last_trade['status'] in ['Open', '🚨 EXIT SIGNAL (Sell on Next Open)']:
                                latest_price = round(processed.iloc[-1]['Close'], 2)
                                buy_val = last_trade['buy_value']
                                current_val = latest_price * last_trade['quantity'] - commission_flat
                                pnl = current_val - buy_val
                                returns_pct = (pnl / buy_val) * 100
                                
                                meta = tickers_df[tickers_df['Symbol'] == ticker].iloc[0] if ticker in tickers_df['Symbol'].values else None
                                industry = meta['Industry'] if meta is not None else 'Unknown'
                                
                                open_trades_list.append({
                                    "Symbol": ticker,
                                    "Industry": industry,
                                    "Indices": meta['Indices'] if meta is not None else 'Nifty 500',
                                    "Buy Date": last_trade['buy_date'],
                                    "Buy Price (₹)": last_trade['buy_price'],
                                    "Quantity": last_trade['quantity'],
                                    "Current Price (₹)": latest_price,
                                    "Current Return %": round(returns_pct, 2),
                                    "Current P&L (₹)": round(pnl, 2),
                                    "Status": last_trade['status']
                                })
                
                if open_trades_list:
                    open_trades_df = pd.DataFrame(open_trades_list)
                    
                    # Highlight exit signals
                    def highlight_status(val):
                        if '🚨' in str(val):
                            return 'background-color: #ffd2d2; color: #d32f2f; font-weight: bold;'
                        return ''
                    
                    if hasattr(open_trades_df.style, 'map'):
                        styled_df = open_trades_df.style.map(highlight_status, subset=['Status'])
                    else:
                        styled_df = open_trades_df.style.applymap(highlight_status, subset=['Status'])
                    st.dataframe(styled_df, width="stretch")
                    
                    # Filter for exit signals
                    exits_only = [t for t in open_trades_list if '🚨' in t['Status']]
                    if exits_only:
                        st.warning(f"⚠️ Exit signal generated for {len(exits_only)} position(s). These should be closed at tomorrow's open.")
                        st.table(pd.DataFrame(exits_only)[["Symbol", "Buy Date", "Current Price (₹)", "Current Return %"]])
                else:
                    st.info("No active open positions detected across monitored tickers.")
            else:
                st.error("Failed to load historical data.")

# ----------------- TAB 4: MULTI-STRATEGY BACKTESTING -----------------
with tab4:
    st.header("🧪 Incremental Walk-Forward Backtesting")
    st.markdown("""
    This tab evaluates the strategy using an **Incremental Walk-Forward (walk-forward optimization)** approach:
    1. The first $N$ trades (Warm-up phase) are bypassed for each stock to establish baseline metrics.
    2. Subsequent trades (Walk-Forward phase) are entered only if the historical win rate and returns from **all previously generated signals** meet the criteria.
    3. Running metrics are continuously updated trade-by-trade to adapt to changing market conditions.
    """)

    # Local Settings for Multi-Strategy Backtesting
    with st.expander("⚙️ Strategy & Backtest Parameters", expanded=True):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            position_size_m = st.number_input("Position Size (₹)", min_value=100, value=5000, step=500, key="pos_size_t4")
        with col_m2:
            commission_flat_m = st.number_input("Commission per Transaction (₹)", min_value=0.0, value=20.0, step=1.0, key="commission_t4")
        with col_m3:
            low_len_m = st.number_input("Lower Channel Length", min_value=5, max_value=100, value=20, key="low_len_t4")
        with col_m4:
            high_len_m = st.number_input("Upper Channel Length", min_value=5, max_value=100, value=20, key="high_len_t4")

    # Hypothesis Parameters
    with st.expander("🔬 Walk-Forward Filtering Criteria", expanded=True):
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            metric_type = st.selectbox("Hypothesis Metric", ["Mean Return", "Median Return"], index=0, key="metric_type_t4")
        with col_h2:
            win_rate_threshold = st.slider("Min Win Rate (%)", min_value=0, max_value=100, value=75, step=5, key="wr_threshold_t4")
        with col_h3:
            min_return_threshold = st.number_input("Min Return / Trade (%)", value=0.0, step=0.5, key="ret_threshold_t4")
        with col_h4:
            warmup_trades = st.number_input("Warm-up Trades (Skip first N)", min_value=1, max_value=50, value=5, step=1, key="warmup_trades_t4")

    # Backtest inputs
    m_col1, m_col2, m_col3 = st.columns([2, 1, 1])
    with m_col1:
        backtest_selection_m = st.multiselect("Select Tickers", all_tickers, default=['EICHERMOT.NS', 'HEROMOTOCO.NS'] if 'EICHERMOT.NS' in all_tickers else all_tickers[:2], key="backtest_tickers_selection_t4")
    with m_col2:
        period_selection_m = st.selectbox("Backtest Duration", ["1y", "2y", "3y", "5y", "10y"], index=2, key="period_selection_t4")
    with m_col3:
        run_all_backtest_m = st.checkbox("Backtest all Nifty 500 tickers", key="run_all_t4")

    # Display warning if loaded signals are empty
    if st.session_state.get('load_signals_warning_t4', False):
        st.warning("No Buy signals have been scanned yet today. Please run the scanner in the 'Daily Signals' tab first.")
        st.session_state['load_signals_warning_t4'] = False

    # Button to load today's buy signals
    st.button("📋 Load Today's Buy Signals", key="load_signals_btn_t4", on_click=load_buy_signals_to_backtest_t4)

    # Pre-trade screening panel for tomorrow's entries
    if st.session_state.get('show_pretrade_screening', False) and st.session_state.get('backtest_tickers_selection_t4'):
        st.subheader("🔍 Tomorrow's Entry Screening (Walk-Forward Rules)")
        with st.spinner("Screening loaded signals against latest historical metrics..."):
            screen_df = run_pretrade_screening(
                tickers_to_screen=st.session_state['backtest_tickers_selection_t4'],
                period_selection=period_selection_m,
                position_size=position_size_m,
                commission_flat=commission_flat_m,
                low_len=low_len_m,
                high_len=high_len_m,
                metric_type=metric_type,
                win_rate_threshold=win_rate_threshold,
                min_return_threshold=min_return_threshold,
                download_stock_data_fn=download_stock_data,
                get_ticker_dataframe_fn=get_ticker_dataframe,
                process_ticker_data_fn=process_ticker_data,
                run_simulation_fn=run_simulation
            )
            if not screen_df.empty:
                def highlight_action(row):
                    bg_color = 'background-color: #d4edda; color: #155724; font-weight: bold;' if "Enter" in str(row['Action for Tomorrow']) else 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                    return [bg_color] * len(row)
                
                if hasattr(screen_df.style, 'apply'):
                    styled_screen = screen_df.style.apply(highlight_action, axis=1)
                else:
                    styled_screen = screen_df
                st.dataframe(styled_screen, width="stretch")
            else:
                st.info("No screenable data could be retrieved.")

    if st.button("🚀 Run Multi-Strategy Backtest", key="run_btn_t4"):
        st.session_state['show_pretrade_screening'] = False
        tickers_to_test = all_tickers if run_all_backtest_m else backtest_selection_m
        
        if not tickers_to_test:
            st.warning("Please select at least one ticker.")
        else:
            with st.spinner(f"Running walk-forward backtest for {len(tickers_to_test)} tickers..."):
                stats_df, all_trades, all_eval_unfiltered, all_eval_filtered = run_multi_strategy_backtest(
                    tickers_to_test=tickers_to_test,
                    period_selection=period_selection_m,
                    position_size=position_size_m,
                    commission_flat=commission_flat_m,
                    low_len=low_len_m,
                    high_len=high_len_m,
                    metric_type=metric_type,
                    win_rate_threshold=win_rate_threshold,
                    min_return_threshold=min_return_threshold,
                    warmup_trades=warmup_trades,
                    download_stock_data_fn=download_stock_data,
                    get_ticker_dataframe_fn=get_ticker_dataframe,
                    process_ticker_data_fn=process_ticker_data,
                    run_simulation_fn=run_simulation
                )
                
                if not stats_df.empty:
                    st.session_state['multi_backtest_results'] = {
                        'stats_df': stats_df,
                        'all_trades': all_trades,
                        'all_eval_unfiltered': all_eval_unfiltered,
                        'all_eval_filtered': all_eval_filtered
                    }
                else:
                    st.warning("No backtest data could be generated.")

    # Display results if available
    if 'multi_backtest_results' in st.session_state:
        res = st.session_state['multi_backtest_results']
        stats_df = res['stats_df']
        eval_unfiltered = res['all_eval_unfiltered']
        eval_filtered = res['all_eval_filtered']
        
        # Summary calculations for Evaluation Period
        unfiltered_total = len(eval_unfiltered)
        unfiltered_wins = len([t for t in eval_unfiltered if t['returns'] > 0])
        unfiltered_wr = (unfiltered_wins / unfiltered_total * 100) if unfiltered_total > 0 else 0.0
        unfiltered_pnl = sum([t['pnl'] for t in eval_unfiltered])
        unfiltered_avg_ret = np.mean([t['returns'] for t in eval_unfiltered]) if unfiltered_total > 0 else 0.0
        
        filtered_total = len(eval_filtered)
        filtered_wins = len([t for t in eval_filtered if t['returns'] > 0])
        filtered_wr = (filtered_wins / filtered_total * 100) if filtered_total > 0 else 0.0
        filtered_pnl = sum([t['pnl'] for t in eval_filtered])
        filtered_avg_ret = np.mean([t['returns'] for t in eval_filtered]) if filtered_total > 0 else 0.0
        
        # Display side-by-side comparison metrics
        st.subheader("📊 Walk-Forward Evaluation Summary (Post-Warmup)")
        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            st.markdown("### ⚠️ Standard Strategy (Unfiltered)")
            st.metric("Total Trades", unfiltered_total)
            st.metric("Win Rate %", f"{unfiltered_wr:.2f}%")
            st.metric("Avg Return / Trade", f"{unfiltered_avg_ret:.2f}%")
            st.metric("Total P&L (₹)", f"₹{unfiltered_pnl:,.2f}")
            
        with col_sum2:
            st.markdown("### 🛡️ Filtered Strategy (Hypothesis)")
            st.metric("Total Trades", filtered_total)
            st.metric("Win Rate %", f"{filtered_wr:.2f}%")
            st.metric("Avg Return / Trade", f"{filtered_avg_ret:.2f}%")
            st.metric("Total P&L (₹)", f"₹{filtered_pnl:,.2f}")
            
        st.subheader("📋 Ticker-Level Walk-Forward Statistics")
        st.dataframe(stats_df, width="stretch")
        st.download_button("Export Ticker Stats", stats_df.to_csv(index=False), "ticker_walk_forward_stats.csv", "text/csv")
        
        st.subheader("📈 Cumulative P&L Comparison")
        # Build cumulative line chart for both strategies
        if eval_unfiltered:
            df_unf = pd.DataFrame(eval_unfiltered).sort_values('sell_date')
            df_unf['Cumulative P&L (Unfiltered)'] = df_unf['pnl'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_unf['sell_date'], y=df_unf['Cumulative P&L (Unfiltered)'], name='Standard Strategy (Unfiltered)', mode='lines+markers'))
            
            if eval_filtered:
                df_fil = pd.DataFrame(eval_filtered).sort_values('sell_date')
                df_fil['Cumulative P&L (Filtered)'] = df_fil['pnl'].cumsum()
                fig.add_trace(go.Scatter(x=df_fil['sell_date'], y=df_fil['Cumulative P&L (Filtered)'], name='Filtered Strategy (Hypothesis)', mode='lines+markers'))
                
            fig.update_layout(title="Walk-Forward Period Cumulative Profit/Loss (₹)", xaxis_title="Date", yaxis_title="Profit/Loss (₹)")
            st.plotly_chart(fig, width="stretch")
            
        st.subheader("📝 Detailed Split-Phase Trade Logs")
        if 'all_trades' in res and res['all_trades']:
            all_trades_df = pd.DataFrame(res['all_trades'])
            
            # Filter by ticker
            available_tickers = sorted(all_trades_df['Stock'].unique())
            selected_ticker = st.selectbox("Select Ticker", available_tickers, key="selected_log_ticker_t4")
            filtered_trades_df = all_trades_df[all_trades_df['Stock'] == selected_ticker].copy()
            
            # Reorder columns to put Phase & Execution Status in front
            cols_order = ['Stock', 'Phase', 'Execution Status', 'buy_date', 'buy_price', 'quantity', 'sell_date', 'sell_price', 'pnl', 'returns', 'holding_period', 'status']
            existing_cols = [c for c in cols_order if c in filtered_trades_df.columns]
            other_cols = [c for c in filtered_trades_df.columns if c not in existing_cols]
            filtered_trades_df = filtered_trades_df[existing_cols + other_cols].sort_values(by='buy_date', ascending=False)
            
            # Style the execution status
            def style_status(row):
                bg_color = ''
                if row['Phase'] == 'Warm-up':
                    bg_color = 'background-color: #f1f3f5; color: #6c757d; font-style: italic;'
                elif row['Execution Status'] == 'Filtered Out':
                    bg_color = 'background-color: #fff3cd; color: #856404;'
                elif row['Execution Status'] == 'Taken':
                    ret_val = row.get('returns')
                    if pd.notna(ret_val):
                        if ret_val > 0:
                            bg_color = 'background-color: #d4edda; color: #155724; font-weight: bold;'
                        else:
                            bg_color = 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                    else:
                        bg_color = 'background-color: #e2e3e5; color: #383d41;' # Open/pending trades
                return [bg_color] * len(row)
                
            if hasattr(filtered_trades_df.style, 'apply'):
                styled_trades = filtered_trades_df.style.apply(style_status, axis=1)
            else:
                styled_trades = filtered_trades_df
                
            st.dataframe(styled_trades, width="stretch")
            st.download_button(f"Export Trade Logs ({selected_ticker})", filtered_trades_df.to_csv(index=False), f"trade_log_{selected_ticker}.csv", "text/csv")
        else:
            st.info("No trades were generated in the selected period.")

# ----------------- TAB 5: ABOUT -----------------
with tab5:
    st.header("ℹ️ About the Strategy & Developer")
    
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
        <img src="https://avatars.githubusercontent.com/u/67477110?v=4" style="width: 100px; height: 100px; border-radius: 8px;" />
        <div>
            <h3 style="margin: 0; font-weight: bold; font-size: 24px;">Kaushik Jegannathan</h3>
            <div style="margin-top: 8px; font-size: 16px;">
                <span style="margin-right: 25px;">🔗 <b>LinkedIn</b>: <a href="https://in.linkedin.com/in/kaushik-jegannathan-02757626" target="_blank">kaushik-jegannathan-02757626</a></span>
                <span>🐙 <b>GitHub</b>: <a href="https://github.com/kaushiktk" target="_blank">kaushiktk</a></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📈 Strategy Theory & Execution Logic")
    
    st.markdown("""
    This application automates and backtests swing-trading strategies using the **Donchian Channel**, structured across three phases:
    
    #### 1. Donchian Channel Indicator
    The Donchian Channel is a trend-following band indicator developed by Richard Donchian. It plots three lines:
    * **Upper Band**: The maximum high price over a chosen channel period ($H$). Represents local resistance.
    * **Lower Band**: The minimum low price over a chosen channel period ($L$). Represents local support.
    * **Midline**: The average of the Upper and Lower bands, acting as the local fair value.
    
    #### 2. Entry and Exit Logic
    * **Entry Trigger (Buy next Open)**: Generated on day $T$ when the price touches or breaches the **Lower Band** (oversold support). The trade executes at day $T+1$'s Open price.
    * **Exit Trigger (Sell next Open)**: Generated on day $T$ when the price touches or breaches the **Upper Band** (overbought resistance). The trade closes at day $T+1$'s Open price.
    
    #### 3. Incremental Walk-Forward Optimization
    To prevent data leakage and adapt to market shifts, the system implements walk-forward evaluation:
    * **Warm-up**: The first $N$ trades of any ticker are skipped. They are analyzed purely to generate a baseline.
    * **Dynamic Filtering**: Any subsequent trade is taken **only if** the running win rate of all previously generated signals (taken or bypassed) exceeds the user-defined threshold (e.g. 75%) and the average/median return is positive.
    """)
    
    st.markdown("---")
    
    st.subheader("📦 Under-the-Hood Stack & Packages")
    st.markdown("""
    This dashboard is built on a modern Python quant-tech stack:
    * **Frontend Framework**: [Streamlit](https://streamlit.io/) — for creating interactive dashboards.
    * **Data Fetching**: [yfinance](https://github.com/ranarousbih/yfinance) — for parallelized downloading of historical NSE/BSE stock data.
    * **Numerical Processing**: [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) — for high-performance rolling calculations and time-series manipulation.
    * **Charting Engines**: [Plotly Express](https://plotly.com/python/) and [Plotly Graph Objects](https://plotly.com/python/graph-objects/) — for interactive canvas renderings of cumulative profits and metrics.
    """)

# Footer Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This application is for educational purposes only. The strategy, indicators, and data presented here do not constitute financial advice, buy/sell recommendations, or investment advice. Trading in financial markets involves high risk, and you are solely responsible for any financial losses or profits resulting from the use of this information. We are not responsible for any decisions made based on this application.")

