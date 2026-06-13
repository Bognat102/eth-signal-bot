import os
from datetime import datetime

import ccxt
import pandas as pd
import telebot
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

exchange = ccxt.binance()


def get_data(symbol="ETH/USDT", timeframe="15m", limit=250):
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        candles,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["avg_volume"] = df["volume"].rolling(20).mean()

    return df


def save_signal(signal, price, rsi, mode):
    file_name = "signals_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "time": now,
        "symbol": "ETHUSDT",
        "mode": mode,
        "signal": signal,
        "price": price,
        "rsi": rsi
    }

    df = pd.DataFrame([row])

    if os.path.exists(file_name):
        df.to_csv(file_name, mode="a", header=False, index=False)
    else:
        df.to_csv(file_name, index=False)


def analyze_strong_signal():
    df_15m = get_data("ETH/USDT", "15m", 250)
    df_1h = get_data("ETH/USDT", "1h", 250)

    last_15m = df_15m.iloc[-1]
    last_1h = df_1h.iloc[-1]

    price = round(last_15m["close"], 2)
    rsi = round(last_15m["rsi"], 2)
    atr = round(last_15m["atr"], 2)

    signal = "NO TRADE"
    reasons = []

    trend_long = (
        last_1h["close"] > last_1h["ema200"]
        and last_1h["ema20"] > last_1h["ema50"]
    )

    trend_short = (
        last_1h["close"] < last_1h["ema200"]
        and last_1h["ema20"] < last_1h["ema50"]
    )

    candle_green = last_15m["close"] > last_15m["open"]
    candle_red = last_15m["close"] < last_15m["open"]
    volume_ok = last_15m["volume"] > last_15m["avg_volume"]

    long_entry = (
        trend_long
        and last_15m["close"] > last_15m["ema200"]
        and last_15m["ema20"] > last_15m["ema50"]
        and 50 <= rsi <= 65
        and volume_ok
        and candle_green
    )

    short_entry = (
        trend_short
        and last_15m["close"] < last_15m["ema200"]
        and last_15m["ema20"] < last_15m["ema50"]
        and 35 <= rsi <= 50
        and volume_ok
        and candle_red
    )

    if long_entry:
        signal = "LONG"
        stop = round(price - atr * 1.5, 2)
        take = round(price + atr * 3, 2)

        reasons = [
            "1H тренд вверх",
            "15M цена выше EMA200",
            "EMA20 выше EMA50",
            f"RSI подходит: {rsi}",
            "Объём выше среднего",
            "Свеча зелёная"
        ]

    elif short_entry:
        signal = "SHORT"
        stop = round(price + atr * 1.5, 2)
        take = round(price - atr * 3, 2)

        reasons = [
            "1H тренд вниз",
            "15M цена ниже EMA200",
            "EMA20 ниже EMA50",
            f"RSI подходит: {rsi}",
            "Объём выше среднего",
            "Свеча красная"
        ]

    else:
        stop = "-"
        take = "-"
        reasons = [
            "Нет сильного сигнала",
            "Бот ждёт более качественный вход"
        ]

    save_signal(signal, price, rsi, "strong")

    text = f"""
ETHUSDT STRONG SIGNAL

Сигнал: {signal}
Цена: {price} USDT
RSI: {rsi}
ATR: {atr}

Стоп: {stop}
Тейк: {take}

Причины:
- {chr(10).join(reasons)}

Деньги НЕ используем.
Только собираем статистику.
"""
    return text


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "Привет! Бот работает.\n\nКоманды:\n/price\n/strong_signal\n/stats"
    )


@bot.message_handler(commands=["price"])
def price(message):
    ticker = exchange.fetch_ticker("ETH/USDT")
    eth_price = ticker["last"]
    bot.reply_to(message, f"Текущая цена ETH: {eth_price} USDT")


@bot.message_handler(commands=["strong_signal"])
def strong_signal(message):
    result = analyze_strong_signal()
    bot.reply_to(message, result)


@bot.message_handler(commands=["stats"])
def stats(message):
    file_name = "signals_log.csv"

    if not os.path.exists(file_name):
        bot.reply_to(message, "Статистики пока нет.")
        return

    df = pd.read_csv(file_name)

    total = len(df)
    longs = len(df[df["signal"] == "LONG"])
    shorts = len(df[df["signal"] == "SHORT"])
    no_trade = len(df[df["signal"] == "NO TRADE"])

    bot.reply_to(
        message,
        f"Всего сигналов: {total}\nLONG: {longs}\nSHORT: {shorts}\nNO TRADE: {no_trade}"
    )

@bot.message_handler(commands=["market"])
def market(message):

    df15 = get_data("ETH/USDT", "15m", 250)
    df1h = get_data("ETH/USDT", "1h", 250)

    last15 = df15.iloc[-1]
    last1h = df1h.iloc[-1]

    trend1h = "BULLISH" if last1h["ema20"] > last1h["ema50"] else "BEARISH"
    trend15 = "BULLISH" if last15["ema20"] > last15["ema50"] else "BEARISH"

    volume_status = (
        "Высокий"
        if last15["volume"] > last15["avg_volume"]
        else "Низкий"
    )

    text = f"""
РЫНОК ETH

Тренд 1H: {trend1h}
Тренд 15M: {trend15}

Цена: {round(last15['close'],2)}
RSI: {round(last15['rsi'],2)}

Объем: {volume_status}

EMA20: {round(last15['ema20'],2)}
EMA50: {round(last15['ema50'],2)}
EMA200: {round(last15['ema200'],2)}
"""

    bot.reply_to(message, text)
print("Бот запущен...")
bot.infinity_polling()