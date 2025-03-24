# utils.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_RSI(series, period=14):
    """
    คำนวณ Relative Strength Index (RSI)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, np.nan)
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI.fillna(0)

def calculate_ATR(data, period=14):
    """
    คำนวณ Average True Range (ATR)
    """
    high_low = data['high'] - data['low']
    high_prev_close = (data['high'] - data['close'].shift()).abs()
    low_prev_close = (data['low'] - data['close'].shift()).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean().fillna(0)

def calculate_MACD(series, short_window=12, long_window=26, signal_window=9):
    """
    คำนวณ Moving Average Convergence Divergence (MACD)
    """
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal

def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    คำนวณ Bollinger Bands
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def load_and_preprocess_data(file_path):
    """
    โหลดข้อมูลจาก CSV และคำนวณ indicators
    """
    try:
        data = pd.read_csv(file_path, delimiter="\t")
    except FileNotFoundError:
        logger.error(f"ไม่พบไฟล์: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    try:
        data['DATETIME'] = pd.to_datetime(data['<DATE>'] + " " + data['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        data.set_index('DATETIME', inplace=True)
        data.sort_index(inplace=True)
        rename_cols = {
            '<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close',
            '<TICKVOL>': 'tickvol', '<VOL>': 'vol', '<SPREAD>': 'spread'
        }
        data.rename(columns=rename_cols, inplace=True)
        data['RSI_14'] = calculate_RSI(data['close'], period=14)
        data['ATR_14'] = calculate_ATR(data, period=14)
        data['MACD'] = calculate_MACD(data['close'])
        data['Bollinger_Upper'], data['Bollinger_Lower'] = calculate_bollinger_bands(data['close'])
        data['MA_short'] = data['close'].rolling(window=5, min_periods=1).mean()
        data['MA_long'] = data['close'].rolling(window=20, min_periods=1).mean()
        data.dropna(inplace=True)
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None
    return data

def preprocess_mt5_data(df):
    """
    ปรับ DataFrame จาก MT5 ให้เหมาะสมกับการคำนวณ indicators
    """
    try:
        df = df.copy()
        df.index = pd.to_datetime(df['time'], unit='s')
        df.sort_index(inplace=True)
        df['RSI_14'] = calculate_RSI(df['close'], period=14)
        df['ATR_14'] = calculate_ATR(df.rename(columns={'real_volume':'vol'}), period=14)
        df['MACD'] = calculate_MACD(df['close'])
        df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['close'])
        df['MA_short'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['MA_long'] = df['close'].rolling(window=20, min_periods=1).mean()
        df.dropna(inplace=True)
    except Exception as e:
        logger.error(f"Error in MT5 preprocessing: {e}")
        return None
    return df
