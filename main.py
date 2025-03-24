# main.py
import os
import time
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import MetaTrader5 as mt5

# Import โมดูลที่เราเขียน
from models import build_lstm_model, build_cnn_lstm_model, build_transformer_model
from risk import monte_carlo_risk_simulation, calculate_position_size
from backtest import run_backtest, walk_forward_analysis
from live_trading import live_trading_loop
from utils import load_and_preprocess_data

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler("trading_system_v4.log")
                    ])
logger = logging.getLogger()

# Configuration
CONFIG = {
    "mode": "backtest",          # "backtest" หรือ "live"
    "data_file": "XAUUSD_M5.csv",  # ไฟล์ข้อมูลสำหรับ backtest
    "symbol": "XAUUSD",          # สำหรับ live trading
    "timeframe": mt5.TIMEFRAME_M5,  # timeframe สำหรับ MT5
    "initial_capital": 100.0,
    "transaction_cost": 0.01,
    "look_back": 10,
    "epochs": 100,
    "batch_size": 32,
    "early_stopping_patience": 10,
    "risk_percent": 0.03,
    "strategy_func": None         # จะถูกกำหนดใน main ต่อไป
}

# ตัวอย่าง strategy function สำหรับสร้างสัญญาณการเทรด
def generate_trade_signal(data_window, scaler, model, look_back, features):
    """
    สร้างสัญญาณเทรดแบบพื้นฐาน โดยใช้ moving averages และ RSI
    """
    current_price = data_window[-1, features.index('close')]
    ma_short = data_window[-1, features.index('MA_short')]
    ma_long = data_window[-1, features.index('MA_long')]
    rsi = data_window[-1, features.index('RSI_14')]
    
    trade_signal = "hold"
    SL, TP = None, None
    if ma_short > ma_long and rsi < 35:
        trade_signal = "buy"
        SL = current_price * 0.98
        TP = current_price * 1.02
    elif ma_short < ma_long and rsi > 65:
        trade_signal = "sell"
        SL = current_price * 1.02
        TP = current_price * 0.98
    
    return trade_signal, SL, TP

# กำหนด strategy function ลงใน CONFIG
CONFIG["strategy_func"] = generate_trade_signal

# รายการ feature ที่ใช้
features = ['open', 'high', 'low', 'close', 'RSI_14', 'ATR_14', 'MACD',
            'Bollinger_Upper', 'Bollinger_Lower', 'MA_short', 'MA_long']

def main():
    if CONFIG["mode"] == "backtest":
        data = load_and_preprocess_data(CONFIG["data_file"])
        if data is None:
            logger.error("ไม่มีข้อมูลให้ทำ backtest")
            return
        logger.info("ข้อมูลหลังการ Preprocessing:")
        logger.info(data[features].tail(5))
        
        dataset = data[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = scaler.fit_transform(dataset)
        
        n_features = len(features)
        look_back = CONFIG["look_back"]
        # ตัวอย่าง: ใช้ CNN-LSTM model สำหรับ backtesting
        model = build_cnn_lstm_model(look_back, n_features)
        
        # ในกรณีที่ต้องเทรนโมเดล สามารถเพิ่ม model.fit() ได้
        run_backtest(data, features, scaler, model, look_back, 
                     risk_module = __import__("risk"), 
                     strategy_func = CONFIG["strategy_func"])
        
        # ตัวอย่าง Walk-Forward Analysis
        # walk_forward_analysis(data, features, scaler, lambda: build_cnn_lstm_model(look_back, n_features),
        #                       look_back, __import__("risk"), CONFIG["strategy_func"])
    else:
        # โหมด live trading
        data = load_and_preprocess_data(CONFIG["data_file"])
        if data is None:
            logger.error("ไม่มีข้อมูลสำหรับ live trading")
            return
        dataset = data[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dataset)
        
        n_features = len(features)
        look_back = CONFIG["look_back"]
        model = build_cnn_lstm_model(look_back, n_features)
        # สมมุติว่าโมเดลนี้ถูกเทรนแล้ว หรือ load model ที่เทรนไว้
        
        live_trading_loop(scaler, model, features, look_back, CONFIG)

if __name__ == "__main__":
    main()
