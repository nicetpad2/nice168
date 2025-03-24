# backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from risk import calculate_position_size

logger = logging.getLogger(__name__)

def simulate_trade(data, trade_signal, entry_index, entry_price, SL, TP):
    """
    จำลองการเทรด โดยพิจารณา stop loss และ take profit
    """
    exit_index, exit_price = None, None
    outcome = "neutral"
    for i in range(entry_index+1, len(data)):
        bar = data.iloc[i]
        if trade_signal == "buy":
            if bar['low'] <= SL:
                exit_index, exit_price = i, SL
                outcome = "loss"
                break
            if bar['high'] >= TP:
                exit_index, exit_price = i, TP
                outcome = "win"
                break
        elif trade_signal == "sell":
            if bar['high'] >= SL:
                exit_index, exit_price = i, SL
                outcome = "loss"
                break
            if bar['low'] <= TP:
                exit_index, exit_price = i, TP
                outcome = "win"
                break
    return exit_index, exit_price, outcome

def run_backtest(data, features, scaler, model, look_back, risk_module, strategy_func):
    """
    จำลองการเทรดและบันทึกผล (Backtesting)
    """
    initial_capital = 100.0
    capital = initial_capital
    trades = []
    TRANSACTION_COST = 0.01
    
    for i in range(look_back, len(data) - 1):
        window = data[features].values[i - look_back:i]
        try:
            trade_signal, SL, TP = strategy_func(window, scaler, model, look_back, features)
        except Exception as e:
            logger.error(f"Error in generating signal at index {i}: {e}")
            continue
        if trade_signal == "hold":
            continue
        
        entry_price = round(data.iloc[i]['close'], 3)
        # ปรับสัดส่วนตำแหน่งด้วย ATR และค่าเฉลี่ยราคาใน window
        volatility_factor = data.iloc[i]['ATR_14'] / data['close'].rolling(window=look_back).mean().iloc[-1]
        position_size = risk_module.calculate_position_size(capital, 0.03, entry_price, SL, volatility_factor)
        if position_size == 0:
            continue
        
        exit_index, exit_price, outcome = simulate_trade(data, trade_signal, i, entry_price, SL, TP)
        if exit_index is None:
            exit_price = round(data.iloc[-1]['close'], 3)
            outcome = "neutral"
        profit_per_unit = (TP - entry_price) if trade_signal=="buy" else (entry_price - TP)
        trade_profit = position_size * profit_per_unit
        capital += trade_profit - TRANSACTION_COST
        
        trades.append({
            "Entry Time": data.index[i],
            "Exit Time": data.index[exit_index] if exit_index is not None else None,
            "Trade": trade_signal,
            "Entry": entry_price,
            "Exit": exit_price,
            "SL": SL,
            "TP": TP,
            "Outcome": outcome,
            "Position Size": position_size,
            "Profit": trade_profit,
            "Capital": capital
        })
        logger.info(f"{data.index[i]} | {trade_signal.upper()} | Entry: {entry_price} | Exit: {exit_price} | Outcome: {outcome.upper()} | Profit: {trade_profit:.2f} | Capital: {capital:.2f}")
    
    # Plot Equity Curve
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.set_index("Entry Time", inplace=True)
        plt.figure(figsize=(12, 6))
        plt.plot(trades_df['Capital'], marker='o', linestyle='-')
        plt.title("Equity Curve V4.0")
        plt.xlabel("Datetime")
        plt.ylabel("Capital ($)")
        plt.grid(True)
        plt.show()
    return trades

def walk_forward_analysis(data, features, scaler, model_builder, look_back, risk_module, strategy_func, window_size=200, step_size=50):
    """
    ทดสอบกลยุทธ์แบบ Walk-Forward Analysis
    """
    results = []
    for start in range(0, len(data)-window_size, step_size):
        train_data = data.iloc[start:start+window_size]
        # สร้างโมเดลใหม่ในแต่ละช่วง (model_builder เป็นฟังก์ชันที่สร้างโมเดล)
        model = model_builder()
        trades = run_backtest(train_data, features, scaler, model, look_back, risk_module, strategy_func)
        results.append(trades)
    return results
