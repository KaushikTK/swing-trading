import pandas as pd
import numpy as np

def run_multi_strategy_backtest(
    tickers_to_test,
    period_selection,
    position_size,
    commission_flat,
    low_len,
    high_len,
    metric_type,
    win_rate_threshold,
    min_return_threshold,
    warmup_trades,
    download_stock_data_fn,
    get_ticker_dataframe_fn,
    process_ticker_data_fn,
    run_simulation_fn
):
    """
    Executes Incremental Walk-Forward backtesting on a list of tickers.
    """
    raw_data = download_stock_data_fn(tuple(tickers_to_test), period_selection)
    
    ticker_stats = []
    all_trades = []
    all_eval_unfiltered = []
    all_eval_filtered = []
    
    for ticker in tickers_to_test:
        ticker_raw = get_ticker_dataframe_fn(raw_data, ticker)
        processed = process_ticker_data_fn(ticker_raw, low_len, high_len)
        
        if processed is not None and not processed.empty:
            sim_trades = run_simulation_fn(processed, position_size, commission_flat)
            for t in sim_trades:
                t['Stock'] = ticker

            # Sort chronologically
            sim_trades = sorted(sim_trades, key=lambda x: x['buy_date'])
            
            for i, t in enumerate(sim_trades):
                if i < warmup_trades:
                    t['Phase'] = 'Warm-up'
                    t['Execution Status'] = 'Filtered Out'
                else:
                    # Evaluate using all prior closed trades
                    prev_closed = [prev for prev in sim_trades[:i] if prev['status'] == 'Closed']
                    prev_count = len(prev_closed)
                    prev_wins = len([prev for prev in prev_closed if prev['returns'] > 0])
                    prev_wr = (prev_wins / prev_count * 100) if prev_count > 0 else 0.0
                    prev_mean = np.mean([prev['returns'] for prev in prev_closed]) if prev_count > 0 else 0.0
                    prev_median = np.median([prev['returns'] for prev in prev_closed]) if prev_count > 0 else 0.0
                    
                    metric_val = prev_mean if metric_type == "Mean Return" else prev_median
                    met_criteria = (prev_wr > win_rate_threshold) and (metric_val > min_return_threshold)
                    
                    t['Phase'] = 'Walk-Forward'
                    t['Execution Status'] = 'Taken' if met_criteria else 'Filtered Out'
            
            # Warmup Stats (for reporting)
            warmup_closed = [t for t in sim_trades[:warmup_trades] if t['status'] == 'Closed']
            warmup_count = len(warmup_closed)
            warmup_wins = len([t for t in warmup_closed if t['returns'] > 0])
            warmup_wr = (warmup_wins / warmup_count * 100) if warmup_count > 0 else 0.0
            warmup_mean = np.mean([t['returns'] for t in warmup_closed]) if warmup_count > 0 else 0.0
            warmup_median = np.median([t['returns'] for t in warmup_closed]) if warmup_count > 0 else 0.0
            
            # Eval Trades (any trade after warmup trades index)
            eval_closed_unfiltered = [t for t in sim_trades[warmup_trades:] if t['status'] == 'Closed']
            eval_closed_filtered = [t for t in eval_closed_unfiltered if t['Execution Status'] == 'Taken']
            
            eval_unfiltered_count = len(eval_closed_unfiltered)
            eval_unfiltered_pnl = sum([t['pnl'] for t in eval_closed_unfiltered])
            eval_filtered_count = len(eval_closed_filtered)
            eval_filtered_pnl = sum([t['pnl'] for t in eval_closed_filtered])
            
            all_trades.extend(sim_trades)
            all_eval_unfiltered.extend(eval_closed_unfiltered)
            all_eval_filtered.extend(eval_closed_filtered)
            
            ticker_stats.append({
                "Ticker": ticker,
                "Warm-up Trades": warmup_count,
                "Warm-up Win Rate %": round(warmup_wr, 2),
                "Warm-up Mean Return %": round(warmup_mean, 2),
                "Warm-up Median Return %": round(warmup_median, 2),
                "Evaluation Mode": "🔄 Walk-Forward",
                "Eval Trades (Unfiltered)": eval_unfiltered_count,
                "Eval P&L Unfiltered (₹)": round(eval_unfiltered_pnl, 2),
                "Eval Trades (Filtered)": eval_filtered_count,
                "Eval P&L Filtered (₹)": round(eval_filtered_pnl, 2)
            })
            
    stats_df = pd.DataFrame(ticker_stats) if ticker_stats else pd.DataFrame()
    return stats_df, all_trades, all_eval_unfiltered, all_eval_filtered


def run_pretrade_screening(
    tickers_to_screen,
    period_selection,
    position_size,
    commission_flat,
    low_len,
    high_len,
    metric_type,
    win_rate_threshold,
    min_return_threshold,
    download_stock_data_fn,
    get_ticker_dataframe_fn,
    process_ticker_data_fn,
    run_simulation_fn
):
    """
    Evaluates whether today's buy signals for tickers should be taken tomorrow based on historical metrics.
    """
    raw_data = download_stock_data_fn(tuple(tickers_to_screen), period_selection)
    results = []
    
    for ticker in tickers_to_screen:
        ticker_raw = get_ticker_dataframe_fn(raw_data, ticker)
        processed = process_ticker_data_fn(ticker_raw, low_len, high_len)
        
        if processed is not None and not processed.empty:
            sim_trades = run_simulation_fn(processed, position_size, commission_flat)
            # Find closed trades up to now
            closed_trades = [t for t in sim_trades if t['status'] == 'Closed']
            count = len(closed_trades)
            wins = len([t for t in closed_trades if t['returns'] > 0])
            wr = (wins / count * 100) if count > 0 else 0.0
            mean_ret = np.mean([t['returns'] for t in closed_trades]) if count > 0 else 0.0
            median_ret = np.median([t['returns'] for t in closed_trades]) if count > 0 else 0.0
            
            metric_val = mean_ret if metric_type == "Mean Return" else median_ret
            met_criteria = (wr > win_rate_threshold) and (metric_val > min_return_threshold)
            
            results.append({
                "Ticker": ticker,
                "Historical Closed Trades": count,
                "Win Rate %": round(wr, 2),
                "Mean Return %": round(mean_ret, 2),
                "Median Return %": round(median_ret, 2),
                "Action for Tomorrow": "🟢 Enter Trade" if met_criteria else "❌ Filter Out"
            })
            
    return pd.DataFrame(results) if results else pd.DataFrame()
