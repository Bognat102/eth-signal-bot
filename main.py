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
CHAT_ID_RAW = os.getenv("CHAT_ID")
CHAT_ID = int(CHAT_ID_RAW) if CHAT_ID_RAW else None

bot = telebot.TeleBot(TOKEN) if TOKEN else None
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"
    }
})

SYMBOL = "ETH/USDT"
BTC_SYMBOL = "BTC/USDT"
SYMBOL_NAME = "ETHUSDT"

TRADES_FILE = "trades_log.csv"
DECISIONS_FILE = "decisions_log.csv"

SLEEP_SECONDS = 300
ALPHA_THRESHOLD = 70
MAX_TRADES_PER_DAY = 10
LIVE_TRADING_ENABLED = False


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def get_data(symbol=SYMBOL, timeframe="15m", limit=250):
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["avg_volume"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["avg_volume"].replace(0, pd.NA)

    # ADX
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).sum() / atr14.replace(0, pd.NA)
    minus_di = 100 * minus_dm.rolling(14).sum() / atr14.replace(0, pd.NA)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, pd.NA)) * 100
    df["adx"] = dx.rolling(14).mean()

    # VWAP for current loaded window
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].replace(0, pd.NA).cumsum()

    return df


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def has_open_trade():
    if not os.path.exists(TRADES_FILE):
        return False
    df = pd.read_csv(TRADES_FILE)
    if df.empty or "status" not in df.columns:
        return False
    return not df[df["status"].isin(["OPEN", "TP1_HIT"])].empty


def trades_today_count():
    if not os.path.exists(TRADES_FILE):
        return 0
    df = pd.read_csv(TRADES_FILE)
    if df.empty or "time" not in df.columns:
        return 0
    return len(df[df["time"].astype(str).str.startswith(today_str())])


def save_decision(action, candidate, alpha, price, rsi, atr, adx, reasons):
    row = {
        "time": now_str(),
        "symbol": SYMBOL_NAME,
        "action": action,
        "candidate": candidate,
        "alpha": alpha,
        "price": price,
        "rsi": rsi,
        "atr": atr,
        "adx": adx,
        "reasons": " | ".join(reasons),
    }
    df = pd.DataFrame([row])
    df.to_csv(DECISIONS_FILE, mode="a", header=not os.path.exists(DECISIONS_FILE), index=False)


def save_trade(signal, price, rsi, atr, adx, alpha, stop, tp1, tp2, reasons):
    row = {
        "time": now_str(),
        "symbol": SYMBOL_NAME,
        "signal": signal,
        "entry": price,
        "rsi": rsi,
        "atr": atr,
        "adx": adx,
        "alpha": alpha,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "status": "OPEN",
        "tp1_hit": "NO",
        "result": "",
        "reasons": " | ".join(reasons),
    }
    df = pd.DataFrame([row])
    df.to_csv(TRADES_FILE, mode="a", header=not os.path.exists(TRADES_FILE), index=False)


def update_trade_results():
    if not os.path.exists(TRADES_FILE):
        return
    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        return

    current_price = float(exchange.fetch_ticker(SYMBOL)["last"])
    changed = False

    for i, row in df.iterrows():
        if row.get("status") not in ["OPEN", "TP1_HIT"]:
            continue

        signal = row["signal"]
        entry = float(row["entry"])
        stop = float(row["stop"])
        tp1 = float(row["tp1"])
        tp2 = float(row["tp2"])
        status = row["status"]
        tp1_hit = str(row.get("tp1_hit", "NO"))

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
                if tp1_hit == "YES" or status == "TP1_HIT":
                    df.at[i, "status"] = "TP1_BE"
                    df.at[i, "result"] = round(tp1 - entry, 2)
                else:
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
                if tp1_hit == "YES" or status == "TP1_HIT":
                    df.at[i, "status"] = "TP1_BE"
                    df.at[i, "result"] = round(entry - tp1, 2)
                else:
                    df.at[i, "status"] = "STOP"
                    df.at[i, "result"] = round(entry - stop, 2)
                changed = True

    if changed:
        df.to_csv(TRADES_FILE, index=False)

def detect_candidate(last_15m, last_1h):
    trend_long = last_1h["close"] > last_1h["ema50"] and last_1h["ema20"] > last_1h["ema50"]
    trend_short = last_1h["close"] < last_1h["ema50"] and last_1h["ema20"] < last_1h["ema50"]

    local_long = last_15m["close"] > last_15m["ema50"] and last_15m["ema20"] > last_15m["ema50"]
    local_short = last_15m["close"] < last_15m["ema50"] and last_15m["ema20"] < last_15m["ema50"]

    rsi = safe_float(last_15m["rsi"])

    if trend_long and local_long and 45 <= rsi <= 75:
        return "LONG"
    if trend_short and local_short and 25 <= rsi <= 55:
        return "SHORT"
    return "NONE"


def btc_confirmation(candidate):
    try:
        btc = get_data(BTC_SYMBOL, "15m", 120).iloc[-2]
        if candidate == "LONG":
            return btc["close"] > btc["ema50"]
        if candidate == "SHORT":
            return btc["close"] < btc["ema50"]
    except Exception:
        return None
    return None


def calculate_alpha(candidate, df_15m, last_15m, last_1h):
    if candidate == "NONE":
        return 0, ["Нет базового направления"]

    reasons = []
    score = 0

    close = safe_float(last_15m["close"])
    open_ = safe_float(last_15m["open"])
    ema20 = safe_float(last_15m["ema20"])
    ema50 = safe_float(last_15m["ema50"])
    ema200 = safe_float(last_15m["ema200"])
    atr = safe_float(last_15m["atr"])
    rsi = safe_float(last_15m["rsi"])
    adx = safe_float(last_15m["adx"])
    volume_ratio = safe_float(last_15m["volume_ratio"], 1.0)
    vwap = safe_float(last_15m["vwap"])

    # Trend: 30 points
    if candidate == "LONG":
        if last_1h["ema20"] > last_1h["ema50"]:
            score += 12; reasons.append("1H тренд вверх")
        if close > ema50:
            score += 8; reasons.append("15M цена выше EMA50")
        if ema20 > ema50:
            score += 6; reasons.append("EMA20 выше EMA50")
        if close > ema200:
            score += 4; reasons.append("Цена выше EMA200")
    else:
        if last_1h["ema20"] < last_1h["ema50"]:
            score += 12; reasons.append("1H тренд вниз")
        if close < ema50:
            score += 8; reasons.append("15M цена ниже EMA50")
        if ema20 < ema50:
            score += 6; reasons.append("EMA20 ниже EMA50")
        if close < ema200:
            score += 4; reasons.append("Цена ниже EMA200")

    # RSI: 15 points
    if candidate == "LONG":
        if 50 <= rsi <= 68:
            score += 15; reasons.append(f"RSI хороший для LONG: {round(rsi,2)}")
        elif 45 <= rsi < 50 or 68 < rsi <= 75:
            score += 8; reasons.append(f"RSI допустимый для LONG: {round(rsi,2)}")
    else:
        if 32 <= rsi <= 50:
            score += 15; reasons.append(f"RSI хороший для SHORT: {round(rsi,2)}")
        elif 25 <= rsi < 32 or 50 < rsi <= 55:
            score += 8; reasons.append(f"RSI допустимый для SHORT: {round(rsi,2)}")

    # ADX/trend strength: 15 points
    if adx >= 25:
        score += 15; reasons.append(f"ADX сильный: {round(adx,2)}")
    elif adx >= 18:
        score += 8; reasons.append(f"ADX допустимый: {round(adx,2)}")

    # Volume: 10 points, not mandatory
    if volume_ratio >= 1.2:
        score += 10; reasons.append(f"Объём выше среднего: x{round(volume_ratio,2)}")
    elif volume_ratio >= 0.8:
        score += 5; reasons.append(f"Объём допустимый: x{round(volume_ratio,2)}")

    # Candle confirmation: 10 points, not mandatory
    if candidate == "LONG" and close > open_:
        score += 10; reasons.append("Зелёная свеча подтверждает LONG")
    elif candidate == "SHORT" and close < open_:
        score += 10; reasons.append("Красная свеча подтверждает SHORT")
    else:
        reasons.append("Свеча не подтверждает идеально, но это не полный запрет")

    # VWAP: 10 points
    if candidate == "LONG" and close >= vwap:
        score += 10; reasons.append("Цена выше VWAP")
    elif candidate == "SHORT" and close <= vwap:
        score += 10; reasons.append("Цена ниже VWAP")

    # Entry timing: 10 points, hard penalty if extreme
    distance = abs(close - ema20)
    last_3_move = abs(safe_float(df_15m["close"].iloc[-2]) - safe_float(df_15m["close"].iloc[-5]))

    if atr > 0 and distance <= atr * 1.8:
        score += 10; reasons.append("Вход не слишком поздний")
    elif atr > 0 and distance <= atr * 2.5:
        score += 4; reasons.append("Вход немного поздний, но допустимый")
    else:
        score -= 20; reasons.append("Слишком поздний вход от EMA20")

    if atr > 0 and last_3_move > atr * 3.0:
        score -= 15; reasons.append("Слишком резкое движение перед сигналом")

    btc_ok = btc_confirmation(candidate)
    if btc_ok is True:
        score += 10; reasons.append("BTC подтверждает направление")
    elif btc_ok is False:
        score -= 10; reasons.append("BTC против направления")
    else:
        reasons.append("BTC подтверждение недоступно")

    score = max(0, min(100, int(round(score))))
    return score, reasons


def make_trade_levels(signal, price, atr):
    if signal == "LONG":
        stop = round(price - atr * 1.1, 2)
        tp1 = round(price + atr * 1.0, 2)
        tp2 = round(price + atr * 1.8, 2)
    else:
        stop = round(price + atr * 1.1, 2)
        tp1 = round(price - atr * 1.0, 2)
        tp2 = round(price - atr * 1.8, 2)
    return stop, tp1, tp2


def analyze_strong_signal():
    update_trade_results()

    df_15m = get_data(SYMBOL, "15m", 250)
    df_1h = get_data(SYMBOL, "1h", 250)

    last_15m = df_15m.iloc[-2]
    last_1h = df_1h.iloc[-2]

    price = round(safe_float(last_15m["close"]), 2)
    rsi = round(safe_float(last_15m["rsi"]), 2)
    atr = round(safe_float(last_15m["atr"]), 2)
    adx = round(safe_float(last_15m["adx"]), 2)

    candidate = detect_candidate(last_15m, last_1h)
    alpha, reasons = calculate_alpha(candidate, df_15m, last_15m, last_1h)

    signal = "NO TRADE"
    stop = "-"
    tp1 = "-"
    tp2 = "-"

    if has_open_trade():
        reasons = ["Уже есть открытая сделка", "Ждём TP1, TP2 или Stop"] + reasons
        save_decision("SKIP", candidate, alpha, price, rsi, atr, adx, reasons)

    elif trades_today_count() >= MAX_TRADES_PER_DAY:
        reasons = [f"Дневной лимит сделок достигнут: {MAX_TRADES_PER_DAY}"] + reasons
        save_decision("SKIP", candidate, alpha, price, rsi, atr, adx, reasons)

    elif candidate in ["LONG", "SHORT"] and alpha >= ALPHA_THRESHOLD and atr > 0:
        signal = candidate
        stop, tp1, tp2 = make_trade_levels(signal, price, atr)
        save_trade(signal, price, rsi, atr, adx, alpha, stop, tp1, tp2, reasons)
        save_decision("OPEN", candidate, alpha, price, rsi, atr, adx, reasons)

    else:
        reasons = ["NO TRADE", f"Alpha ниже порога {ALPHA_THRESHOLD} или нет направления"] + reasons
        save_decision("SKIP", candidate, alpha, price, rsi, atr, adx, reasons)

    text = f"""
ETHUSDT STRONG SIGNAL

Сигнал: {signal}
Кандидат: {candidate}
Alpha Score: {alpha}/100
Цена: {price} USDT
RSI: {rsi}
ATR: {atr}
ADX: {adx}

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
    bot.reply_to(message, "Привет! Бот работает.\n\nКоманды:\n/price\n/strong_signal\n/stats\n/market\n/decisions")


@bot.message_handler(commands=["price"])
def price(message):
    eth_price = exchange.fetch_ticker(SYMBOL)["last"]
    bot.reply_to(message, f"Текущая цена ETH: {eth_price} USDT")


@bot.message_handler(commands=["strong_signal"])
def strong_signal(message):
    _, result = analyze_strong_signal()
    bot.reply_to(message, result)


@bot.message_handler(commands=["stats"])
def stats(message):
    update_trade_results()
    if not os.path.exists(TRADES_FILE):
        bot.reply_to(message, "Статистики пока нет.")
        return
    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        bot.reply_to(message, "Статистики пока нет.")
        return

    total = len(df)
    longs = len(df[df["signal"] == "LONG"])
    shorts = len(df[df["signal"] == "SHORT"])
    open_trades = len(df[df["status"].isin(["OPEN", "TP1_HIT"])])
    tp1 = len(df[df["tp1_hit"] == "YES"])
    tp2 = len(df[df["status"] == "TP2_HIT"])
    stops = len(df[df["status"] == "STOP"])

    bot.reply_to(message, f"""
СТАТИСТИКА ETH

Всего сделок: {total}
LONG: {longs}
SHORT: {shorts}

TP1 был достигнут: {tp1}
TP2 был достигнут: {tp2}
Стопов: {stops}
Открытых: {open_trades}
""")


@bot.message_handler(commands=["decisions"])
def decisions(message):
    if not os.path.exists(DECISIONS_FILE):
        bot.reply_to(message, "Журнал решений пока пуст.")
        return
    df = pd.read_csv(DECISIONS_FILE)
    if df.empty:
        bot.reply_to(message, "Журнал решений пока пуст.")
        return

    total = len(df)
    opened = len(df[df["action"] == "OPEN"])
    skipped = len(df[df["action"] == "SKIP"])
    avg_alpha = round(df["alpha"].mean(), 2) if "alpha" in df.columns else 0
    last = df.tail(10)

    lines = []
    for _, row in last.iterrows():
        lines.append(f"{row['time']} | {row['action']} | {row['candidate']} | Alpha {row['alpha']}")

    bot.reply_to(message, f"""
ЖУРНАЛ РЕШЕНИЙ

Всего решений: {total}
Открыто: {opened}
Пропущено: {skipped}
Средний Alpha: {avg_alpha}

Последние 10:
{chr(10).join(lines)}
""")


@bot.message_handler(commands=["market"])
def market(message):
    df15 = get_data(SYMBOL, "15m", 250)
    df1h = get_data(SYMBOL, "1h", 250)
    last15 = df15.iloc[-2]
    last1h = df1h.iloc[-2]

    trend1h = "BULLISH" if last1h["ema20"] > last1h["ema50"] else "BEARISH"
    trend15 = "BULLISH" if last15["ema20"] > last15["ema50"] else "BEARISH"
    volume_status = "Высокий" if last15["volume"] > last15["avg_volume"] else "Нормальный/низкий"

    bot.reply_to(message, f"""
РЫНОК ETH

Тренд 1H: {trend1h}
Тренд 15M: {trend15}

Цена: {round(safe_float(last15['close']), 2)}
RSI: {round(safe_float(last15['rsi']), 2)}
ADX: {round(safe_float(last15['adx']), 2)}
ATR: {round(safe_float(last15['atr']), 2)}

Объём: {volume_status}
Volume Ratio: {round(safe_float(last15['volume_ratio'], 1), 2)}

EMA20: {round(safe_float(last15['ema20']), 2)}
EMA50: {round(safe_float(last15['ema50']), 2)}
EMA200: {round(safe_float(last15['ema200']), 2)}
VWAP: {round(safe_float(last15['vwap']), 2)}
""")

@bot.message_handler(commands=["results"])
def results(message):
    update_trade_results()

    if not os.path.exists(TRADES_FILE):
        bot.reply_to(message, "Результатов пока нет.")
        return

    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        bot.reply_to(message, "Результатов пока нет.")
        return

    total = len(df)
    open_count = len(df[df["status"].isin(["OPEN", "TP1_HIT"])]) if "status" in df.columns else 0
    closed = df[~df["status"].isin(["OPEN", "TP1_HIT"])] if "status" in df.columns else df.iloc[0:0]

    tp2 = len(df[df["status"] == "TP2_HIT"]) if "status" in df.columns else 0
    stops = len(df[df["status"] == "STOP"]) if "status" in df.columns else 0
    tp1_only = len(df[df["status"] == "TP1_HIT"]) if "status" in df.columns else 0

    result_sum = 0.0
    if "result" in df.columns:
        result_sum = pd.to_numeric(df["result"], errors="coerce").fillna(0).sum()

    win_rate = 0
    if len(closed) > 0:
        wins = 0
        if "result" in closed.columns:
            wins = (pd.to_numeric(closed["result"], errors="coerce").fillna(0) > 0).sum()
        win_rate = round((wins / len(closed)) * 100, 1)

    bot.reply_to(message, f"""
РЕЗУЛЬТАТЫ ETH

Всего сделок: {total}
Закрытых: {len(closed)}
Открытых: {open_count}

TP2: {tp2}
TP1/в работе: {tp1_only}
Стопов: {stops}

Win Rate закрытых: {win_rate}%
Общий результат по цене: {round(result_sum, 2)} USDT

Деньги НЕ используем.
Это paper-статистика.
""")


@bot.message_handler(commands=["history"])
def history(message):
    update_trade_results()

    if not os.path.exists(TRADES_FILE):
        bot.reply_to(message, "Истории сделок пока нет.")
        return

    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        bot.reply_to(message, "Истории сделок пока нет.")
        return

    lines = []
    last_rows = df.tail(10)

    for _, row in last_rows.iterrows():
        time_value = row.get("time", "-")
        signal = row.get("signal", "-")
        status = row.get("status", "-")
        entry = row.get("entry", "-")
        result = row.get("result", "")
        alpha = row.get("alpha", row.get("alpha_score", "-"))

        lines.append(
            f"{time_value} | {signal} | {status} | Entry: {entry} | Result: {result} | Alpha: {alpha}"
        )

    bot.reply_to(message, "ИСТОРИЯ ПОСЛЕДНИХ СДЕЛОК\n\n" + "\n".join(lines))




@bot.message_handler(commands=["test_tp1"])
def test_tp1(message):
    # Safe logic test: does not open trades and does not change trades_log.csv
    long_entry = 100.0
    long_tp1 = 110.0
    long_result_after_be = round(long_tp1 - long_entry, 2)

    short_entry = 100.0
    short_tp1 = 90.0
    short_result_after_be = round(short_entry - short_tp1, 2)

    bot.reply_to(message, f"""
TP1_BE TEST

LONG example:
Entry: {long_entry}
TP1: {long_tp1}
After TP1 + BE result: +{long_result_after_be}
Status should be: TP1_BE

SHORT example:
Entry: {short_entry}
TP1: {short_tp1}
After TP1 + BE result: +{short_result_after_be}
Status should be: TP1_BE

OK: TP1 profit is preserved after breakeven.
This test does NOT change real trades.
""")

@bot.message_handler(commands=["skip_audit"])
def skip_audit(message):
    if not os.path.exists(DECISIONS_FILE):
        bot.reply_to(message, "Аудита пока нет. Журнал решений ещё пуст.")
        return

    df = pd.read_csv(DECISIONS_FILE)
    if df.empty:
        bot.reply_to(message, "Аудита пока нет. Журнал решений ещё пуст.")
        return

    total = len(df)
    opened = len(df[df["action"] == "OPEN"]) if "action" in df.columns else 0
    skipped = len(df[df["action"] == "SKIP"]) if "action" in df.columns else 0

    none_count = len(df[df["candidate"] == "NONE"]) if "candidate" in df.columns else 0
    long_count = len(df[df["candidate"] == "LONG"]) if "candidate" in df.columns else 0
    short_count = len(df[df["candidate"] == "SHORT"]) if "candidate" in df.columns else 0

    low_alpha = 0
    if "alpha" in df.columns and "action" in df.columns:
        low_alpha = len(df[(df["action"] == "SKIP") & (pd.to_numeric(df["alpha"], errors="coerce").fillna(0) < ALPHA_THRESHOLD)])

    reasons_text = " | ".join(df["reasons"].astype(str).tolist()) if "reasons" in df.columns else ""

    open_trade_blocks = reasons_text.count("Уже есть открытая сделка")
    daily_limit_blocks = reasons_text.count("Дневной лимит сделок")
    no_direction_blocks = reasons_text.count("Нет базового направления")
    btc_against = reasons_text.count("BTC против направления")
    late_entries = reasons_text.count("Слишком поздний вход")
    sharp_moves = reasons_text.count("Слишком резкое движение")

    bot.reply_to(message, f"""
АУДИТ ПРОПУСКОВ

Всего решений: {total}
Открыто: {opened}
Пропущено: {skipped}

Кандидаты:
LONG: {long_count}
SHORT: {short_count}
NONE: {none_count}

Что мешало сделкам:
Низкий Alpha: {low_alpha}
Уже есть сделка: {open_trade_blocks}
Дневной лимит: {daily_limit_blocks}
Нет направления: {no_direction_blocks}
BTC против: {btc_against}
Поздний вход: {late_entries}
Резкое движение: {sharp_moves}

Главное смотреть:
1) если много NONE — фильтр направления слишком жёсткий;
2) если много низкий Alpha — порог/баллы слишком жёсткие;
3) если много "уже есть сделка" — бот часто ждёт закрытия старой сделки.
""")


def auto_check():
    while True:
        try:
            signal, result = analyze_strong_signal()
            if signal in ["LONG", "SHORT"] and CHAT_ID:
                bot.send_message(CHAT_ID, result)
            print("Автопроверка ETH выполнена:", signal, flush=True)
        except Exception as e:
            print("Ошибка автопроверки:", e, flush=True)
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    if not TOKEN or CHAT_ID is None:
        raise RuntimeError("BOT_TOKEN или CHAT_ID не указаны в переменных окружения")
    threading.Thread(target=auto_check, daemon=True).start()
    print("Бот запущен... LIVE_TRADING_ENABLED =", LIVE_TRADING_ENABLED, flush=True)
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)
