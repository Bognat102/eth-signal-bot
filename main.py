import os
import time
import threading
from datetime import datetime

import ccxt
import pandas as pd
import telebot
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))

bot = telebot.TeleBot(TOKEN)
exchange = ccxt.binance()

SYMBOL = "ETH/USDT"
SYMBOL_NAME = "ETHUSDT"

TRADES_FILE = "trades_log.csv"


def get_data(symbol=SYMBOL, timeframe="15m", limit=250):
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


def has_open_trade():
    if not os.path.exists(TRADES_FILE):
        return False

    df = pd.read_csv(TRADES_FILE)

    if df.empty:
        return False

    open_trades = df[df["status"].isin(["OPEN", "TP1_HIT"])]
    return not open_trades.empty


def save_trade(signal, price, rsi, atr, stop, tp1, tp2, reasons):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "time": now,
        "symbol": SYMBOL_NAME,
        "signal": signal,
        "entry": price,
        "rsi": rsi,
        "atr": atr,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "status": "OPEN",
        "tp1_hit": "NO",
        "result": "",
        "reasons": " | ".join(reasons)
    }

    df = pd.DataFrame([row])

    if os.path.exists(TRADES_FILE):
        df.to_csv(TRADES_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADES_FILE, index=False)


def update_trade_results():
    if not os.path.exists(TRADES_FILE):
        return

    df = pd.read_csv(TRADES_FILE)

    if df.empty:
        return

    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = float(ticker["last"])

    changed = False

    for i, row in df.iterrows():
        if row["status"] not in ["OPEN", "TP1_HIT"]:
            continue

        signal = row["signal"]
        entry = float(row["entry"])
        stop = float(row["stop"])
        tp1 = float(row["tp1"])
        tp2 = float(row["tp2"])
        status = row["status"]

        if signal == "LONG":
            if current_price >= tp2:
                df.at[i, "status"] = "TP2_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "result"] = round(tp2 - entry, 2)
                changed = True

            elif current_price >= tp1 and status == "OPEN":
                df.at[i, "status"] = "TP1_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "stop"] = entry
                df.at[i, "result"] = round(tp1 - entry, 2)
                changed = True

            elif current_price <= stop:
                df.at[i, "status"] = "STOP"
                df.at[i, "result"] = round(stop - entry, 2)
                changed = True

        elif signal == "SHORT":
            if current_price <= tp2:
                df.at[i, "status"] = "TP2_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "result"] = round(entry - tp2, 2)
                changed = True

            elif current_price <= tp1 and status == "OPEN":
                df.at[i, "status"] = "TP1_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "stop"] = entry
                df.at[i, "result"] = round(entry - tp1, 2)
                changed = True

            elif current_price >= stop:
                df.at[i, "status"] = "STOP"
                df.at[i, "result"] = round(entry - stop, 2)
                changed = True

    if changed:
        df.to_csv(TRADES_FILE, index=False)


def analyze_strong_signal():
    update_trade_results()

    df_15m = get_data(SYMBOL, "15m", 250)
    df_1h = get_data(SYMBOL, "1h", 250)

    last_15m = df_15m.iloc[-1]
    last_1h = df_1h.iloc[-1]

    price = round(last_15m["close"], 2)
    rsi = round(last_15m["rsi"], 2)
    atr = round(last_15m["atr"], 2)

    signal = "NO TRADE"
    stop = "-"
    tp1 = "-"
    tp2 = "-"

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

    distance_from_ema20 = abs(last_15m["close"] - last_15m["ema20"])
    not_late_entry = distance_from_ema20 <= last_15m["atr"] * 1.2

    last_3_move = abs(df_15m["close"].iloc[-1] - df_15m["close"].iloc[-4])
    no_fast_pump = last_3_move <= last_15m["atr"] * 1.8

    long_entry = (
        trend_long
        and last_15m["close"] > last_15m["ema200"]
        and last_15m["ema20"] > last_15m["ema50"]
        and 50 <= rsi <= 62
        and volume_ok
        and candle_green
        and not_late_entry
        and no_fast_pump
    )

    short_entry = (
        trend_short
        and last_15m["close"] < last_15m["ema200"]
        and last_15m["ema20"] < last_15m["ema50"]
        and 38 <= rsi <= 50
        and volume_ok
        and candle_red
        and not_late_entry
        and no_fast_pump
    )

    if has_open_trade():
        reasons = [
            "Уже есть открытая сделка",
            "Новый сигнал не записываем",
            "Ждём TP1, TP2 или стоп"
        ]

    elif long_entry:
        signal = "LONG"
        stop = round(price - atr * 1.2, 2)
        tp1 = round(price + atr * 1, 2)
        tp2 = round(price + atr * 2, 2)

        reasons = [
            "1H тренд вверх",
            "15M цена выше EMA200",
            "EMA20 выше EMA50",
            f"RSI подходит: {rsi}",
            "Объём выше среднего",
            "Свеча зелёная",
            "Вход НЕ запоздалый"
        ]

        save_trade(signal, price, rsi, atr, stop, tp1, tp2, reasons)

    elif short_entry:
        signal = "SHORT"
        stop = round(price + atr * 1.2, 2)
        tp1 = round(price - atr * 1, 2)
        tp2 = round(price - atr * 2, 2)

        reasons = [
            "1H тренд вниз",
            "15M цена ниже EMA200",
            "EMA20 ниже EMA50",
            f"RSI подходит: {rsi}",
            "Объём выше среднего",
            "Свеча красная",
            "Вход НЕ запоздалый"
        ]

        save_trade(signal, price, rsi, atr, stop, tp1, tp2, reasons)

    else:
        reasons = [
            "Нет сильного сигнала",
            "Бот ждёт более качественный вход",
            "Поздние входы теперь отсекаются"
        ]

    text = f"""
ETHUSDT STRONG SIGNAL

Сигнал: {signal}
Цена: {price} USDT
RSI: {rsi}
ATR: {atr}

Стоп: {stop}
Take Profit 1: {tp1}
Take Profit 2: {tp2}

Причины:
- {chr(10).join(reasons)}

Деньги НЕ используем.
Только собираем статистику.
"""
    return signal, text


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "Привет! Бот работает.\n\nКоманды:\n/price\n/strong_signal\n/stats\n/market"
    )


@bot.message_handler(commands=["price"])
def price(message):
    ticker = exchange.fetch_ticker(SYMBOL)
    eth_price = ticker["last"]
    bot.reply_to(message, f"Текущая цена ETH: {eth_price} USDT")


@bot.message_handler(commands=["strong_signal"])
def strong_signal(message):
    signal, result = analyze_strong_signal()
    bot.reply_to(message, result)


@bot.message_handler(commands=["stats"])
def stats(message):
    update_trade_results()

    if not os.path.exists(TRADES_FILE):
        bot.reply_to(message, "Статистики пока нет.")
        return

    df = pd.read_csv(TRADES_FILE)

    total = len(df)
    longs = len(df[df["signal"] == "LONG"])
    shorts = len(df[df["signal"] == "SHORT"])
    open_trades = len(df[df["status"].isin(["OPEN", "TP1_HIT"])])
    tp1 = len(df[df["tp1_hit"] == "YES"])
    tp2 = len(df[df["status"] == "TP2_HIT"])
    stops = len(df[df["status"] == "STOP"])

    bot.reply_to(
        message,
        f"""
СТАТИСТИКА ETH

Всего сделок: {total}
LONG: {longs}
SHORT: {shorts}

TP1 был достигнут: {tp1}
TP2 был достигнут: {tp2}
Стопов: {stops}
Открытых: {open_trades}
"""
    )


@bot.message_handler(commands=["market"])
def market(message):
    df15 = get_data(SYMBOL, "15m", 250)
    df1h = get_data(SYMBOL, "1h", 250)

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

Цена: {round(last15['close'], 2)}
RSI: {round(last15['rsi'], 2)}

Объем: {volume_status}

EMA20: {round(last15['ema20'], 2)}
EMA50: {round(last15['ema50'], 2)}
EMA200: {round(last15['ema200'], 2)}
"""

    bot.reply_to(message, text)


def auto_check():
    while True:
        try:
            signal, result = analyze_strong_signal()

            if signal in ["LONG", "SHORT"]:
                bot.send_message(CHAT_ID, result)

            print("Автопроверка ETH выполнена:", signal, flush=True)

        except Exception as e:
            print("Ошибка автопроверки:", e, flush=True)

        time.sleep(900)


threading.Thread(target=auto_check, daemon=True).start()

print("Бот запущен...")
bot.infinity_polling()
