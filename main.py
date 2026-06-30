import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Any

import ccxt
import pandas as pd
import telebot
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID_RAW = os.getenv("CHAT_ID")
CHAT_ID = int(CHAT_ID_RAW) if CHAT_ID_RAW else None

bot = telebot.TeleBot(TOKEN) if TOKEN else None
exchange = ccxt.binance({"enableRateLimit": True})

SYMBOL = "ETH/USDT"
SYMBOL_NAME = "ETHUSDT"
BTC_SYMBOL = "BTC/USDT"

TRADES_FILE = "trades_log.csv"
DECISIONS_FILE = "decisions_log.csv"

TIMEFRAME = "15m"
SLEEP_SECONDS = 900

# Production safety defaults
LIVE_TRADING_ENABLED = False
MIN_ALPHA_SCORE = 85
MIN_TRADE_RR = 1.6
MAX_DAILY_SIGNALS = 7
MAX_OPEN_TRADES = 1
MAX_LATE_ENTRY_ATR = 1.05
MAX_FAST_MOVE_ATR = 1.65
MIN_VOLUME_MULTIPLIER = 1.05


# -----------------------------
# Data and indicators
# -----------------------------

def get_data(symbol: str = SYMBOL, timeframe: str = "15m", limit: int = 250) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])

    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
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
    down_move = df["low"].shift() - df["low"]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_adx.replace(0, pd.NA))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_adx.replace(0, pd.NA))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, pd.NA)) * 100
    df["adx"] = dx.rolling(14).mean()

    # VWAP approximation for recent loaded window
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].replace(0, pd.NA).cumsum()

    return df


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# -----------------------------
# Journal helpers
# -----------------------------

def append_csv(file_path: str, row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


def has_open_trade() -> bool:
    if not os.path.exists(TRADES_FILE):
        return False
    df = pd.read_csv(TRADES_FILE)
    if df.empty or "status" not in df.columns:
        return False
    open_trades = df[df["status"].isin(["OPEN", "TP1_HIT", "TP2_HIT"])]
    return len(open_trades) >= MAX_OPEN_TRADES


def today_signal_count() -> int:
    if not os.path.exists(TRADES_FILE):
        return 0
    df = pd.read_csv(TRADES_FILE)
    if df.empty or "time" not in df.columns:
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    return len(df[df["time"].astype(str).str.startswith(today)])


def log_decision(signal: str, alpha_score: int, decision: str, reasons: List[str], metrics: Dict[str, Any]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": now,
        "symbol": SYMBOL_NAME,
        "signal": signal,
        "decision": decision,
        "alpha_score": alpha_score,
        "price": metrics.get("price", ""),
        "rsi": metrics.get("rsi", ""),
        "atr": metrics.get("atr", ""),
        "adx": metrics.get("adx", ""),
        "volume_ratio": metrics.get("volume_ratio", ""),
        "btc_confirmed": metrics.get("btc_confirmed", ""),
        "late_entry_atr": metrics.get("late_entry_atr", ""),
        "fast_move_atr": metrics.get("fast_move_atr", ""),
        "rr": metrics.get("rr", ""),
        "reasons": " | ".join(reasons),
    }
    append_csv(DECISIONS_FILE, row)


def save_trade(signal: str, price: float, rsi: float, atr: float, adx: float, alpha_score: int,
               stop: float, tp1: float, tp2: float, reasons: List[str], metrics: Dict[str, Any]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk = abs(price - stop)
    reward = abs(tp2 - price)
    rr = round(reward / risk, 2) if risk > 0 else 0

    row = {
        "time": now,
        "symbol": SYMBOL_NAME,
        "signal": signal,
        "entry": price,
        "rsi": rsi,
        "atr": atr,
        "adx": adx,
        "alpha_score": alpha_score,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "status": "OPEN",
        "tp1_hit": "NO",
        "tp2_hit": "NO",
        "result": "",
        "result_r": "",
        "mae": 0,
        "mfe": 0,
        "volume_ratio": metrics.get("volume_ratio", ""),
        "btc_confirmed": metrics.get("btc_confirmed", ""),
        "late_entry_atr": metrics.get("late_entry_atr", ""),
        "fast_move_atr": metrics.get("fast_move_atr", ""),
        "reasons": " | ".join(reasons),
    }
    append_csv(TRADES_FILE, row)


# -----------------------------
# Position management
# -----------------------------

def update_trade_results() -> None:
    if not os.path.exists(TRADES_FILE):
        return

    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        return

    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = float(ticker["last"])
    changed = False

    for i, row in df.iterrows():
        if row.get("status") not in ["OPEN", "TP1_HIT", "TP2_HIT"]:
            continue

        signal = row["signal"]
        entry = safe_float(row["entry"])
        stop = safe_float(row["stop"])
        tp1 = safe_float(row["tp1"])
        tp2 = safe_float(row["tp2"])
        atr = safe_float(row.get("atr", 0))
        status = row["status"]
        initial_risk = abs(entry - stop) if status == "OPEN" else max(atr * 1.2, 0.01)

        # MAE/MFE update
        if signal == "LONG":
            mae = min(safe_float(row.get("mae", 0)), round(current_price - entry, 2))
            mfe = max(safe_float(row.get("mfe", 0)), round(current_price - entry, 2))
        else:
            mae = min(safe_float(row.get("mae", 0)), round(entry - current_price, 2))
            mfe = max(safe_float(row.get("mfe", 0)), round(entry - current_price, 2))
        df.at[i, "mae"] = mae
        df.at[i, "mfe"] = mfe

        if signal == "LONG":
            # TP2 final close
            if current_price >= tp2:
                profit = tp2 - entry
                df.at[i, "status"] = "TP2_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "tp2_hit"] = "YES"
                df.at[i, "result"] = round(profit, 2)
                df.at[i, "result_r"] = round(profit / initial_risk, 2) if initial_risk else ""
                changed = True

            # TP1: move stop to breakeven, keep trade open for TP2
            elif current_price >= tp1 and status == "OPEN":
                df.at[i, "status"] = "TP1_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "stop"] = entry
                changed = True

            elif current_price <= stop:
                result = stop - entry
                df.at[i, "status"] = "STOP" if stop != entry else "BREAKEVEN"
                df.at[i, "result"] = round(result, 2)
                df.at[i, "result_r"] = round(result / initial_risk, 2) if initial_risk else ""
                changed = True

        elif signal == "SHORT":
            if current_price <= tp2:
                profit = entry - tp2
                df.at[i, "status"] = "TP2_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "tp2_hit"] = "YES"
                df.at[i, "result"] = round(profit, 2)
                df.at[i, "result_r"] = round(profit / initial_risk, 2) if initial_risk else ""
                changed = True

            elif current_price <= tp1 and status == "OPEN":
                df.at[i, "status"] = "TP1_HIT"
                df.at[i, "tp1_hit"] = "YES"
                df.at[i, "stop"] = entry
                changed = True

            elif current_price >= stop:
                result = entry - stop
                df.at[i, "status"] = "STOP" if stop != entry else "BREAKEVEN"
                df.at[i, "result"] = round(result, 2)
                df.at[i, "result_r"] = round(result / initial_risk, 2) if initial_risk else ""
                changed = True

    if changed:
        df.to_csv(TRADES_FILE, index=False)


# -----------------------------
# Alpha and No Trade Engine
# -----------------------------

def calculate_alpha(direction: str, last_15m: pd.Series, last_1h: pd.Series, last_btc_1h: pd.Series,
                    df_15m: pd.DataFrame) -> Tuple[int, List[str], Dict[str, Any]]:
    price = safe_float(last_15m["close"])
    atr = safe_float(last_15m["atr"])
    rsi = safe_float(last_15m["rsi"])
    adx = safe_float(last_15m["adx"])
    volume_ratio = safe_float(last_15m["volume_ratio"])

    distance_from_ema20 = abs(price - safe_float(last_15m["ema20"]))
    late_entry_atr = distance_from_ema20 / atr if atr else 999

    last_3_move = abs(safe_float(df_15m["close"].iloc[-2]) - safe_float(df_15m["close"].iloc[-5]))
    fast_move_atr = last_3_move / atr if atr else 999

    btc_bull = last_btc_1h["ema20"] > last_btc_1h["ema50"] and last_btc_1h["close"] > last_btc_1h["ema200"]
    btc_bear = last_btc_1h["ema20"] < last_btc_1h["ema50"] and last_btc_1h["close"] < last_btc_1h["ema200"]
    btc_confirmed = (direction == "LONG" and btc_bull) or (direction == "SHORT" and btc_bear)

    score = 0
    reasons = []

    # 1H trend quality: 25
    if direction == "LONG" and last_1h["close"] > last_1h["ema200"] and last_1h["ema20"] > last_1h["ema50"]:
        score += 25
        reasons.append("1H тренд подтверждает LONG (+25)")
    elif direction == "SHORT" and last_1h["close"] < last_1h["ema200"] and last_1h["ema20"] < last_1h["ema50"]:
        score += 25
        reasons.append("1H тренд подтверждает SHORT (+25)")

    # 15M structure: 20
    if direction == "LONG" and last_15m["close"] > last_15m["ema200"] and last_15m["ema20"] > last_15m["ema50"]:
        score += 20
        reasons.append("15M структура сильная вверх (+20)")
    elif direction == "SHORT" and last_15m["close"] < last_15m["ema200"] and last_15m["ema20"] < last_15m["ema50"]:
        score += 20
        reasons.append("15M структура сильная вниз (+20)")

    # RSI pullback zone: 15
    if direction == "LONG" and 50 <= rsi <= 62:
        score += 15
        reasons.append(f"RSI в зоне LONG pullback: {round(rsi, 2)} (+15)")
    elif direction == "SHORT" and 38 <= rsi <= 50:
        score += 15
        reasons.append(f"RSI в зоне SHORT pullback: {round(rsi, 2)} (+15)")

    # Volume: 10
    if volume_ratio >= MIN_VOLUME_MULTIPLIER:
        vol_points = 10 if volume_ratio >= 1.25 else 7
        score += vol_points
        reasons.append(f"Объём выше среднего x{round(volume_ratio, 2)} (+{vol_points})")

    # ADX: 10
    if adx >= 25:
        score += 10
        reasons.append(f"ADX сильный: {round(adx, 2)} (+10)")
    elif adx >= 18:
        score += 5
        reasons.append(f"ADX средний: {round(adx, 2)} (+5)")

    # BTC confirmation: 10
    if btc_confirmed:
        score += 10
        reasons.append("BTC подтверждает направление (+10)")

    # Not late: 5
    if late_entry_atr <= MAX_LATE_ENTRY_ATR:
        score += 5
        reasons.append(f"Вход не поздний: {round(late_entry_atr, 2)} ATR от EMA20 (+5)")

    # No fast pump/dump: 5
    if fast_move_atr <= MAX_FAST_MOVE_ATR:
        score += 5
        reasons.append(f"Нет слишком резкого движения: {round(fast_move_atr, 2)} ATR (+5)")

    metrics = {
        "price": round(price, 2),
        "rsi": round(rsi, 2),
        "atr": round(atr, 2),
        "adx": round(adx, 2),
        "volume_ratio": round(volume_ratio, 2),
        "btc_confirmed": btc_confirmed,
        "late_entry_atr": round(late_entry_atr, 2),
        "fast_move_atr": round(fast_move_atr, 2),
    }

    return min(score, 100), reasons, metrics


def no_trade_checks(alpha_score: int, metrics: Dict[str, Any]) -> List[str]:
    blocks = []
    if has_open_trade():
        blocks.append("Уже есть открытая сделка")
    if today_signal_count() >= MAX_DAILY_SIGNALS:
        blocks.append(f"Дневной лимит сделок достигнут: {MAX_DAILY_SIGNALS}")
    if alpha_score < MIN_ALPHA_SCORE:
        blocks.append(f"Alpha Score низкий: {alpha_score} < {MIN_ALPHA_SCORE}")
    if not metrics.get("btc_confirmed", False):
        blocks.append("BTC не подтверждает направление")
    if safe_float(metrics.get("volume_ratio")) < MIN_VOLUME_MULTIPLIER:
        blocks.append("Объём недостаточный")
    if safe_float(metrics.get("late_entry_atr")) > MAX_LATE_ENTRY_ATR:
        blocks.append("Поздний вход")
    if safe_float(metrics.get("fast_move_atr")) > MAX_FAST_MOVE_ATR:
        blocks.append("Слишком резкое движение до сигнала")
    if safe_float(metrics.get("atr")) <= 0:
        blocks.append("ATR некорректный")
    return blocks


# -----------------------------
# Signal analysis
# -----------------------------

def analyze_strong_signal() -> Tuple[str, str]:
    update_trade_results()

    df_15m = get_data(SYMBOL, "15m", 250)
    df_1h = get_data(SYMBOL, "1h", 250)
    df_btc_1h = get_data(BTC_SYMBOL, "1h", 250)

    # Use closed candles only
    last_15m = df_15m.iloc[-2]
    last_1h = df_1h.iloc[-2]
    last_btc_1h = df_btc_1h.iloc[-2]

    price = round(safe_float(last_15m["close"]), 2)
    rsi = round(safe_float(last_15m["rsi"]), 2)
    atr = round(safe_float(last_15m["atr"]), 2)
    adx = round(safe_float(last_15m["adx"]), 2)

    signal = "NO TRADE"
    stop = "-"
    tp1 = "-"
    tp2 = "-"

    candle_green = last_15m["close"] > last_15m["open"]
    candle_red = last_15m["close"] < last_15m["open"]

    long_alpha, long_reasons, long_metrics = calculate_alpha("LONG", last_15m, last_1h, last_btc_1h, df_15m)
    short_alpha, short_reasons, short_metrics = calculate_alpha("SHORT", last_15m, last_1h, last_btc_1h, df_15m)

    # Candle confirmation adds a final practical condition
    if candle_green:
        long_alpha = min(long_alpha + 3, 100)
        long_reasons.append("Свеча закрылась зелёной (+3)")
    if candle_red:
        short_alpha = min(short_alpha + 3, 100)
        short_reasons.append("Свеча закрылась красной (+3)")

    # Choose only the better direction
    if long_alpha >= short_alpha:
        candidate = "LONG"
        alpha_score = long_alpha
        reasons = long_reasons
        metrics = long_metrics
    else:
        candidate = "SHORT"
        alpha_score = short_alpha
        reasons = short_reasons
        metrics = short_metrics

    metrics["rr"] = 2.0
    blocks = no_trade_checks(alpha_score, metrics)

    # Candle must match direction
    if candidate == "LONG" and not candle_green:
        blocks.append("Для LONG нет зелёного подтверждения свечи")
    if candidate == "SHORT" and not candle_red:
        blocks.append("Для SHORT нет красного подтверждения свечи")

    if not blocks:
        signal = candidate
        if signal == "LONG":
            stop = round(price - atr * 1.15, 2)
            tp1 = round(price + atr * 1.0, 2)
            tp2 = round(price + atr * 2.1, 2)
        else:
            stop = round(price + atr * 1.15, 2)
            tp1 = round(price - atr * 1.0, 2)
            tp2 = round(price - atr * 2.1, 2)

        save_trade(signal, price, rsi, atr, adx, alpha_score, stop, tp1, tp2, reasons, metrics)
        log_decision(signal, alpha_score, "OPEN", reasons, metrics)
    else:
        reasons = ["NO TRADE"] + blocks + ["Бот ждёт вход с подтверждённым преимуществом"]
        log_decision(candidate, alpha_score, "SKIP", reasons, metrics)

    text = f"""
ETHUSDT STRONG SIGNAL

Сигнал: {signal}
Кандидат: {candidate}
Alpha Score: {alpha_score}/100
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


# -----------------------------
# Telegram commands
# -----------------------------

@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "Привет! Бот работает.\n\nКоманды:\n/price\n/strong_signal\n/stats\n/market\n/decisions"
    )


@bot.message_handler(commands=["price"])
def price(message):
    ticker = exchange.fetch_ticker(SYMBOL)
    eth_price = ticker["last"]
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
    tp2 = len(df[df.get("tp2_hit", "") == "YES"]) if "tp2_hit" in df.columns else len(df[df["status"] == "TP2_HIT"])
    stops = len(df[df["status"] == "STOP"])
    breakeven = len(df[df["status"] == "BREAKEVEN"])

    closed = df[df["status"].isin(["TP2_HIT", "STOP", "BREAKEVEN"])]
    avg_alpha = round(df["alpha_score"].mean(), 2) if "alpha_score" in df.columns else "-"
    avg_r = round(pd.to_numeric(closed.get("result_r", pd.Series(dtype=float)), errors="coerce").mean(), 2) if not closed.empty else "-"

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
Безубыток: {breakeven}
Открытых: {open_trades}

Средний Alpha Score: {avg_alpha}
Средний результат R: {avg_r}
"""
    )


@bot.message_handler(commands=["decisions"])
def decisions(message):
    if not os.path.exists(DECISIONS_FILE):
        bot.reply_to(message, "Журнала решений пока нет.")
        return

    df = pd.read_csv(DECISIONS_FILE)
    if df.empty:
        bot.reply_to(message, "Журнала решений пока нет.")
        return

    last = df.tail(10)
    opened = len(df[df["decision"] == "OPEN"])
    skipped = len(df[df["decision"] == "SKIP"])
    avg_alpha = round(pd.to_numeric(df["alpha_score"], errors="coerce").mean(), 2)

    lines = []
    for _, row in last.iterrows():
        lines.append(f"{row['time']} | {row['decision']} | {row['signal']} | Alpha {row['alpha_score']}")

    bot.reply_to(
        message,
        f"""
ЖУРНАЛ РЕШЕНИЙ

Всего решений: {len(df)}
Открыто: {opened}
Пропущено: {skipped}
Средний Alpha: {avg_alpha}

Последние 10:
{chr(10).join(lines)}
"""
    )


@bot.message_handler(commands=["market"])
def market(message):
    df15 = get_data(SYMBOL, "15m", 250)
    df1h = get_data(SYMBOL, "1h", 250)
    dfbtc = get_data(BTC_SYMBOL, "1h", 250)

    last15 = df15.iloc[-2]
    last1h = df1h.iloc[-2]
    lastbtc = dfbtc.iloc[-2]

    trend1h = "BULLISH" if last1h["ema20"] > last1h["ema50"] else "BEARISH"
    trend15 = "BULLISH" if last15["ema20"] > last15["ema50"] else "BEARISH"
    trendbtc = "BULLISH" if lastbtc["ema20"] > lastbtc["ema50"] else "BEARISH"

    volume_status = "Высокий" if last15["volume"] > last15["avg_volume"] else "Низкий"

    text = f"""
РЫНОК ETH

Тренд 1H: {trend1h}
Тренд 15M: {trend15}
BTC 1H: {trendbtc}

Цена: {round(last15['close'], 2)}
RSI: {round(last15['rsi'], 2)}
ADX: {round(last15['adx'], 2)}
ATR: {round(last15['atr'], 2)}

Объем: {volume_status}
Volume Ratio: {round(last15['volume_ratio'], 2)}

EMA20: {round(last15['ema20'], 2)}
EMA50: {round(last15['ema50'], 2)}
EMA200: {round(last15['ema200'], 2)}
VWAP: {round(last15['vwap'], 2)}
"""
    bot.reply_to(message, text)


# -----------------------------
# Auto loop
# -----------------------------

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
