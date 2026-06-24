import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
import os
import urllib.request

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

def load_buy_signals_to_backtest():
    if st.session_state.get('today_buy_signals'):
        st.session_state['backtest_tickers_selection'] = st.session_state['today_buy_signals']
        st.session_state['load_signals_warning'] = False
    else:
        st.session_state['load_signals_warning'] = True

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
    
    # Calculate Donchian Channels using pandas_ta
    try:
        donchian = df.ta.donchian(lower_length=low, upper_length=high)
        if donchian is not None and len(donchian.columns) >= 3:
            df[['low', 'mid', 'high']] = donchian
        else:
            return None
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
tab1, tab2, tab3 = st.tabs(["📊 Daily Signals", "📈 Backtesting", "💼 Current Open Trades"])

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
                    st.dataframe(buy_df, use_container_width=True)
                    st.download_button("Export Buy Signals", buy_df.to_csv(index=False), "buy_signals.csv", "text/csv")
                else:
                    st.info("No active Buy signals detected today.")
                    
                # Sell Signals Table
                st.subheader("🔴 Active Sell Signals (Sell on Next Open)")
                if sell_signals_list:
                    sell_df = pd.DataFrame(sell_signals_list)
                    st.dataframe(sell_df, use_container_width=True)
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
        show_ticker_breakdown = st.checkbox("📊 Show Ticker-Level Stats")
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
                st.dataframe(pd.DataFrame(ticker_stats), use_container_width=True)
            else:
                st.info("No closed trades to calculate ticker stats.")
                
        st.subheader("Detailed Trade Logs")
        st.dataframe(trades_df.sort_values(by='buy_date', ascending=False), use_container_width=True)
        
        # Performance plot
        if not closed_trades.empty:
            closed_trades = closed_trades.sort_values(by='sell_date')
            closed_trades['Cumulative P&L'] = closed_trades['pnl'].cumsum()
            fig = px.line(closed_trades, x='sell_date', y='Cumulative P&L', title="Cumulative Realized Profit/Loss (₹)", markers=True)
            st.plotly_chart(fig, use_container_width=True)

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
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Filter for exit signals
                    exits_only = [t for t in open_trades_list if '🚨' in t['Status']]
                    if exits_only:
                        st.warning(f"⚠️ Exit signal generated for {len(exits_only)} position(s). These should be closed at tomorrow's open.")
                        st.table(pd.DataFrame(exits_only)[["Symbol", "Buy Date", "Current Price (₹)", "Current Return %"]])
                else:
                    st.info("No active open positions detected across monitored tickers.")
            else:
                st.error("Failed to load historical data.")

# Footer Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This application is for educational purposes only. The strategy, indicators, and data presented here do not constitute financial advice, buy/sell recommendations, or investment advice. Trading in financial markets involves high risk, and you are solely responsible for any financial losses or profits resulting from the use of this information. We are not responsible for any decisions made based on this application.")

