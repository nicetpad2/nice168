# live_trading.py
import time
import logging
import MetaTrader5 as mt5
import pandas as pd
from utils import preprocess_mt5_data

logger = logging.getLogger(__name__)

def connect_mt5():
    """
    เชื่อมต่อกับ MetaTrader 5
    """
    try:
        if not mt5.initialize():
            logger.error("ไม่สามารถเชื่อมต่อ MT5 ได้, error code =", mt5.last_error())
            return False
    except Exception as e:
        logger.error(f"Error connecting to MT5: {e}")
        return False
    logger.info("เชื่อมต่อ MT5 สำเร็จ")
    return True

def fetch_mt5_data(symbol, timeframe, num_bars):
    """
    ดึงข้อมูลราคาจาก MT5
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None:
            logger.error("ไม่สามารถดึงข้อมูลจาก MT5 ได้")
            return None
        df = pd.DataFrame(rates)
        return df
    except Exception as e:
        logger.error(f"Error fetching MT5 data: {e}")
        return None

def send_order_to_mt5(symbol, trade_signal, volume, SL, TP):
    """
    ส่งคำสั่งซื้อขายไปยัง MT5
    """
    try:
        tick = mt5.symbol_info_tick(symbol)
        if trade_signal == "buy":
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": SL,
            "tp": TP,
            "deviation": 20,
            "magic": 234000,
            "comment": "Trade from Advanced Strategy V4.0",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed, retcode={result.retcode}")
            return False
        logger.info(f"Order {trade_signal.upper()} successful | Price: {price:.3f} | Volume: {volume}")
        return True
    except Exception as e:
        logger.error(f"Error sending order to MT5: {e}")
        return False

def live_trading_loop(scaler, model, features, look_back, config):
    """
    วนลูปสำหรับ live trading ผ่าน MT5
    """
    if not connect_mt5():
        return
    symbol = config["symbol"]
    timeframe = config["timeframe"]
    capital = config["initial_capital"]
    while True:
        df = fetch_mt5_data(symbol, timeframe, 200)
        if df is None:
            logger.warning("ไม่สามารถดึงข้อมูล MT5 ได้, รอ 60 วินาที...")
            time.sleep(60)
            continue
        data_live = preprocess_mt5_data(df)
        if data_live is None or data_live.shape[0] < look_back:
            logger.warning("ข้อมูลไม่เพียงพอสำหรับการเทรด")
            time.sleep(60)
            continue
        window = data_live[features].values[-look_back:]
        try:
            trade_signal, SL, TP = config["strategy_func"](window, scaler, model, look_back, features)
        except Exception as e:
            logger.error(f"Error generating live signal: {e}")
            time.sleep(60)
            continue
        if trade_signal == "hold":
            logger.info("No trade signal, HOLD")
        else:
            entry_price = round(data_live.iloc[-1]['close'], 3)
            volatility_factor = data_live.iloc[-1]['ATR_14'] / data_live['close'].rolling(window=look_back).mean().iloc[-1]
            from risk import calculate_position_size
            position_size = calculate_position_size(capital, config["risk_percent"], entry_price, SL, volatility_factor)
            if position_size == 0:
                logger.warning("Position size เป็น 0, ข้ามการส่งออเดอร์")
            else:
                if send_order_to_mt5(symbol, trade_signal, position_size, SL, TP):
                    logger.info(f"Live Trade: {trade_signal.upper()} | Entry: {entry_price} | SL: {SL} | TP: {TP}")
        time.sleep(60)
