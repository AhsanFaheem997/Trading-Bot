# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 18:18:23 2025

@author: HP
"""

import ccxt
import pandas as pd
import talib
import time

# === User Input ===
capital = float(input("Enter your USDT capital to use: "))
target_profit = float(input("Enter your desired profit in USDT (e.g., 1.0): "))

# === Settings ===
timeframe = '1m'
limit = 1000
rsi_period = 14
ema_fast = 10
ema_slow = 50
overbought = 70
oversold = 30
volume_spike_multiplier = 1.5
max_leverage = 20

# === Exchange Setup ===
exchange = ccxt.mexc({
    'options': {'defaultType': 'future'},
    'enableRateLimit': True,
})

def fetch_data(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[ERROR] Fetch failed for {symbol}: {e}")
        return None

def detect_trend(df):
    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
    return "uptrend" if df['close'].iloc[-1] > df['EMA_50'].iloc[-1] else "downtrend"

def detect_market_structure(df):
    recent_highs = df['high'].tail(5).reset_index(drop=True)
    recent_lows = df['low'].tail(5).reset_index(drop=True)
    higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
    lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))

    if higher_highs:
        return "bullish structure"
    elif lower_lows:
        return "bearish structure"
    else:
        return "range"

def detect_fvg(df):
    if df['high'].iloc[-3] < df['low'].iloc[-2]:
        return "Bullish FVG"
    elif df['low'].iloc[-3] > df['high'].iloc[-2]:
        return "Bearish FVG"
    return None

def detect_liquidity_sweep(df):
    prev_high = df['high'].iloc[-2]
    prev_low = df['low'].iloc[-2]
    cur_high = df['high'].iloc[-1]
    cur_low = df['low'].iloc[-1]
    cur_close = df['close'].iloc[-1]

    if cur_high > prev_high and cur_close < prev_high:
        return "Liquidity Grab (Short Wick)"
    elif cur_low < prev_low and cur_close > prev_low:
        return "Liquidity Grab (Long Wick)"
    return None

def detect_volume_spike(df):
    avg_volume = df['volume'].iloc[-20:-1].mean()
    last_volume = df['volume'].iloc[-1]
    return last_volume > avg_volume * volume_spike_multiplier

def detect_fake_wick(df):
    last_candle = df.iloc[-1]
    body_size = abs(last_candle['close'] - last_candle['open'])
    wick_size = last_candle['high'] - last_candle['low']
    if wick_size > 2 * body_size and last_candle['volume'] < df['volume'].iloc[-20:-1].mean():
        return True
    return False

def detect_candle_patterns(df):
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    patterns = {
        "HAMMER": talib.CDLHAMMER(open_, high, low, close),
        "INVERTED_HAMMER": talib.CDLINVERTEDHAMMER(open_, high, low, close),
        "ENGULFING": talib.CDLENGULFING(open_, high, low, close),
        "MORNING_STAR": talib.CDLMORNINGSTAR(open_, high, low, close),
        "EVENING_STAR": talib.CDLEVENINGSTAR(open_, high, low, close),
        "DOJI": talib.CDLDOJI(open_, high, low, close),
        "SHOOTING_STAR": talib.CDLSHOOTINGSTAR(open_, high, low, close),
    }

    detected = []
    for name, pattern_array in patterns.items():
        if pattern_array[-1] != 0:
            detected.append(name)
    
    return detected if detected else None

def calculate_tp(entry_price, capital, leverage, desired_profit):
    qty = (capital * leverage) / entry_price
    price_diff = desired_profit / qty
    tp_price = entry_price + price_diff if leverage > 0 else None
    return tp_price

def recommend_leverage(signal, trend, rsi, patterns, fvg, sweep):
    if signal == "WAIT":
        return 0
    confluences = 0
    if trend in ["uptrend", "downtrend"]:
        confluences += 1
    if fvg is not None:
        confluences += 1
    if sweep is not None:
        confluences += 1
    if patterns is not None:
        confluences += 1
    if (signal == "LONG" and rsi < 30) or (signal == "SHORT" and rsi > 70):
        confluences += 1
    if confluences >= 4:
        return 20
    elif confluences == 3:
        return 10
    elif confluences == 2:
        return 5
    else:
        return 2

def analyze(df):
    df['RSI'] = talib.RSI(df['close'], timeperiod=rsi_period)
    df['EMA_fast'] = talib.EMA(df['close'], timeperiod=ema_fast)
    df['EMA_slow'] = talib.EMA(df['close'], timeperiod=ema_slow)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Detect EMA crossover
    ema_cross_up = prev['EMA_fast'] <= prev['EMA_slow'] and latest['EMA_fast'] > latest['EMA_slow']
    ema_cross_down = prev['EMA_fast'] >= prev['EMA_slow'] and latest['EMA_fast'] < latest['EMA_slow']

    signal = "WAIT"
    action = "Hold / No Entry"
    leverage = 0
    tp = None

    # Simple RSI + EMA crossover strategy
    if ema_cross_up and latest['RSI'] < 50:
        signal = "LONG"
    elif ema_cross_down and latest['RSI'] > 50:
        signal = "SHORT"

    # You can reuse your recommend_leverage and calculate_tp functions here
    leverage = recommend_leverage(signal, "trend ignored here", latest['RSI'], None, None, None)
    position_size = capital * leverage
    entry_price = latest['close']
    tp = calculate_tp(entry_price, capital, leverage, target_profit) if signal != "WAIT" else None

    if signal == "LONG":
        action = f"🟢 Open LONG at {entry_price:.6f}, TP: {tp:.6f}"
    elif signal == "SHORT":
        action = f"🔴 Open SHORT at {entry_price:.6f}, TP: {tp:.6f}"

    return {
        'time': latest['timestamp'],
        'price': entry_price,
        'RSI': latest['RSI'],
        'signal': signal,
        'action': action,
        'leverage': leverage,
        'tp': tp,
        'position_size': position_size,
        'manipulation': ""
    }


def run_live():
    print(f"\n🚨 Precision Futures Bot | Capital: {capital} USDT | Desired Profit: {target_profit} USDT\n")
    while True:
        all_signals = []
        for sym in symbols_to_scan:
            df = fetch_data(sym)
            if df is not None:
                result = analyze(df)
                if result['signal'] in ['LONG', 'SHORT']:
                    all_signals.append((sym, result))
        if all_signals:
            for sym, result in all_signals:
                print(
                    f"[{result['time']}] {sym} | Price: {result['price']:.6f} | RSI: {result['RSI']:.2f} | "
                    f"Signal: {result['signal']} | Action: {result['action']} | Leverage: {result['leverage']}x | "
                    f"Position: ${result['position_size']:.2f} | Manipulation: {result['manipulation']}"
                )
        else:
            print("No LONG or SHORT signals detected this scan.")
        time.sleep(30)

if __name__ == '__main__':
    print("Loading MEXC Futures markets...")
    exchange.load_markets()
    futures_symbols = [s for s in exchange.symbols if s.endswith('/USDT')]

    print(f"Found {len(futures_symbols)} USDT Futures symbols on MEXC.")

    print("Fetching tickers to select top 100 by volume...")
    tickers = exchange.fetch_tickers(futures_symbols)
    sorted_symbols = sorted(tickers.items(), key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
    top_symbols = [s[0] for s in sorted_symbols[:100]]

    print(f"Selected top 100 symbols by volume.\n")

    # Assign globally for run_live()
    symbols_to_scan = top_symbols

    # Start live scan
    run_live()
