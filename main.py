"""
main_portfolio_v1.py

Портфельная версия бота: торгует НЕСКОЛЬКИМИ монетами одновременно (по умолчанию ADA+ARB),
используя ту же проверенную логику из strategy_core.py, что и одиночный бот
main_FLEX_ADX_EXT15_v2.py и бэктесты (backtest.py, backtest_portfolio.py).

Конфигурация подтверждена бэктестом на этих двух активах:
- Риск на сделку: strategy_core.RISK_PER_TRADE_PCT (сейчас 5%)
- Стоп: STOP_DISTANCE_MULT=0.6 (протестированный оптимум)
- ADX_RANGE_MAX=30

ВАЖНО (то же ограничение, что и в бэктесте): риск считается независимо на каждую монету —
если ADA и ARB одновременно откроют сделки, суммарный риск в моменте может быть
2×RISK_PER_TRADE_PCT, а не RISK_PER_TRADE_PCT. Портфельного лимита нет.

Каждая монета ведёт СВОЙ отдельный файл сделок (trades_log_<symbol>.csv) и решений
(decisions_log_<symbol>.csv), чтобы не путать логи разных активов.
"""

import os
import time
import threading
from datetime import datetime

import ccxt
import pandas as pd
import telebot
from dotenv import load_dotenv

import strategy_core as sc

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

SYMBOLS = ["ADA/USDT:USDT", "ARB/USDT:USDT"]
BTC_SYMBOL = "BTC/USDT:USDT"

SLEEP_SECONDS = 300
LIVE_TRADING_ENABLED = False  # НЕ включай без дополнительного тестирования на реальном исполнении


def symbol_name(symbol):
    return symbol.replace("/USDT:USDT", "USDT")


def trades_file(symbol):
    return f"trades_log_{symbol_name(symbol)}.csv"


def decisions_file(symbol):
    return f"decisions_log_{symbol_name(symbol)}.csv"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_data(symbol, timeframe="15m", limit=250):
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    return sc.compute_indicators(df)


def has_open_trade(symbol):
    f = trades_file(symbol)
    if not os.path.exists(f):
        return False
    df = pd.read_csv(f)
    if df.empty or "status" not in df.columns:
        return False
    return not df[df["status"].isin(["OPEN", "TP1_HIT"])].empty


def save_decision(symbol, action, candidate, alpha, price, rsi, atr, adx, reasons):
    row = {
        "time": now_str(), "symbol": symbol_name(symbol), "action": action,
        "candidate": candidate, "alpha": alpha, "price": price, "rsi": rsi,
        "atr": atr, "adx": adx, "reasons": " | ".join(reasons),
    }
    df = pd.DataFrame([row])
    f = decisions_file(symbol)
    df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)


def save_trade(symbol, signal, price, rsi, atr, adx, alpha, stop, tp1, tp2, reasons, qty, risk_amount):
    row = {
        "time": now_str(), "symbol": symbol_name(symbol), "signal": signal, "entry": price,
        "rsi": rsi, "atr": atr, "adx": adx, "alpha": alpha, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": qty, "risk_amount": risk_amount, "status": "OPEN", "tp1_hit": "NO",
        "peak": price, "result": "", "pnl_usdt": "", "reasons": " | ".join(reasons),
    }
    df = pd.DataFrame([row])
    f = trades_file(symbol)
    df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)


def update_trade_results(symbol):
    """Проверяет открытые сделки ПО ЭТОЙ монете, используя high/low последней закрытой свечи."""
    f = trades_file(symbol)
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    if df.empty:
        return

    df15 = get_data(symbol, "15m", 3)
    last_candle = df15.iloc[-2]
    candle_high = sc.safe_float(last_candle["high"])
    candle_low = sc.safe_float(last_candle["low"])
    current_price = float(exchange.fetch_ticker(symbol)["last"])
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
        atr_entry = float(row.get("atr", 0) or 0)
        qty = float(row.get("qty", 0) or 0)
        peak = float(row.get("peak", entry) or entry)

        def close_trade(new_status, exit_price):
            diff = (exit_price - entry) if signal == "LONG" else (entry - exit_price)
            pnl_usdt = sc.apply_fees(diff, entry, exit_price, qty)
            df.at[i, "status"] = new_status
            df.at[i, "result"] = round(diff, 2)
            df.at[i, "pnl_usdt"] = pnl_usdt

        if status == "OPEN":
            if signal == "LONG":
                if candle_high >= tp1 or current_price >= tp1:
                    df.at[i, "status"] = "TP1_HIT"
                    df.at[i, "stop"] = entry
                    df.at[i, "peak"] = max(entry, candle_high)
                    changed = True
                elif candle_low <= stop or current_price <= stop:
                    close_trade("STOP", stop)
                    changed = True
            else:
                if candle_low <= tp1 or current_price <= tp1:
                    df.at[i, "status"] = "TP1_HIT"
                    df.at[i, "stop"] = entry
                    df.at[i, "peak"] = min(entry, candle_low)
                    changed = True
                elif candle_high >= stop or current_price >= stop:
                    close_trade("STOP", stop)
                    changed = True

        elif status == "TP1_HIT":
            if sc.USE_TRAILING_EXIT:
                if signal == "LONG":
                    new_peak = max(peak, candle_high)
                    trail_stop = new_peak - sc.TRAIL_ATR_MULT * atr_entry
                    new_stop = max(stop, trail_stop)
                    df.at[i, "peak"] = new_peak
                    df.at[i, "stop"] = new_stop
                    if candle_low <= new_stop or current_price <= new_stop:
                        close_trade("TRAIL_EXIT", new_stop)
                    changed = True
                else:
                    new_peak = min(peak, candle_low)
                    trail_stop = new_peak + sc.TRAIL_ATR_MULT * atr_entry
                    new_stop = min(stop, trail_stop)
                    df.at[i, "peak"] = new_peak
                    df.at[i, "stop"] = new_stop
                    if candle_high >= new_stop or current_price >= new_stop:
                        close_trade("TRAIL_EXIT", new_stop)
                    changed = True
            else:
                if signal == "LONG":
                    if candle_high >= tp2 or current_price >= tp2:
                        close_trade("TP2_HIT", tp2)
                        changed = True
                    elif candle_low <= stop or current_price <= stop:
                        close_trade("TP1_BE", entry)
                        changed = True
                else:
                    if candle_low <= tp2 or current_price <= tp2:
                        close_trade("TP2_HIT", tp2)
                        changed = True
                    elif candle_high >= stop or current_price >= stop:
                        close_trade("TP1_BE", entry)
                        changed = True

    if changed:
        df.to_csv(f, index=False)


def funding_percentile(symbol):
    try:
        history = exchange.fetch_funding_rate_history(symbol, limit=90)
        if not history:
            return None
        rates = pd.Series([h["fundingRate"] for h in history])
        current = rates.iloc[-1]
        return (rates <= current).mean() * 100
    except Exception:
        return None


def btc_confirmation(candidate, btc_last_15m):
    if btc_last_15m is None:
        return None
    if candidate == "LONG":
        return bool(btc_last_15m["close"] > btc_last_15m["ema50"])
    if candidate == "SHORT":
        return bool(btc_last_15m["close"] < btc_last_15m["ema50"])
    return None


def get_current_equity(symbols):
    """Текущий 'виртуальный' капитал = стартовый + сумма PnL всех закрытых сделок по ВСЕМ монетам.
    Общий капитал, чтобы риск на новую сделку считался от реального текущего состояния портфеля
    (согласовано с логикой backtest_portfolio.py — компаундинг на общий капитал)."""
    equity = sc.ACCOUNT_EQUITY_USDT
    for symbol in symbols:
        f = trades_file(symbol)
        if os.path.exists(f):
            df = pd.read_csv(f)
            if not df.empty and "pnl_usdt" in df.columns:
                equity += pd.to_numeric(df["pnl_usdt"], errors="coerce").fillna(0).sum()
    return equity


def analyze_symbol(symbol, btc_last_15m, current_equity):
    update_trade_results(symbol)

    df_15m = get_data(symbol, "15m", 250)
    df_1h = get_data(symbol, "1h", 250)
    last_15m = df_15m.iloc[-2]
    last_1h = df_1h.iloc[-2]

    price = round(sc.safe_float(last_15m["close"]), 2)
    rsi = round(sc.safe_float(last_15m["rsi"]), 2)
    atr = round(sc.safe_float(last_15m["atr"]), 2)
    adx = round(sc.safe_float(last_15m["adx"]), 2)

    funding_pctile = funding_percentile(symbol)
    candidate = sc.detect_candidate(last_15m, last_1h, funding_pctile)
    btc_ok = btc_confirmation(candidate, btc_last_15m) if candidate in ("LONG", "SHORT") else None
    funding_ok = sc.funding_confirms(candidate, funding_pctile)
    alpha, reasons = sc.calculate_alpha(candidate, df_15m, last_15m, last_1h, btc_ok, funding_ok)

    signal = "NO TRADE"
    stop = tp1 = tp2 = "-"
    qty_text = ""

    if has_open_trade(symbol):
        reasons = ["Уже есть открытая сделка по этой монете"] + reasons
        save_decision(symbol, "SKIP", candidate, alpha, price, rsi, atr, adx, reasons)

    elif (
        candidate in ["LONG", "SHORT"]
        and alpha >= sc.ALPHA_THRESHOLD
        and sc.safe_float(last_15m["bb_mid"]) > 0
        and not (sc.USE_FUNDING_HARD_FILTER and funding_ok is False)
    ):
        signal = candidate
        stop, tp1, tp2 = sc.make_trade_levels(signal, price, atr, sc.safe_float(last_15m["bb_mid"]))
        qty, risk_amount = sc.position_size(price, stop, equity=current_equity)
        save_trade(symbol, signal, price, rsi, atr, adx, alpha, stop, tp1, tp2, reasons, qty, risk_amount)
        save_decision(symbol, "OPEN", candidate, alpha, price, rsi, atr, adx, reasons)
        qty_text = f"\nРазмер позиции: {qty} (риск {risk_amount} USDT, капитал портфеля {round(current_equity,2)} USDT)"

    else:
        reasons = ["NO TRADE"] + reasons
        save_decision(symbol, "SKIP", candidate, alpha, price, rsi, atr, adx, reasons)

    text = f"""
{symbol_name(symbol)} STRONG SIGNAL

Сигнал: {signal}
Кандидат: {candidate}
Alpha Score: {alpha}/100
Цена: {price} USDT
RSI: {rsi}
ATR: {atr}
ADX: {adx}

Стоп: {stop}
Take Profit 1: {tp1}
Take Profit 2: {tp2}{qty_text}

Причины:
- {chr(10).join(reasons)}

Деньги НЕ используем. Только собираем статистику.
"""
    return signal, text


def analyze_all_symbols():
    """Проверяет BTC-тренд один раз (общий для всех монет), затем анализирует каждую монету."""
    btc_last_15m = None
    try:
        btc_last_15m = get_data(BTC_SYMBOL, "15m", 120).iloc[-2]
    except Exception:
        pass

    current_equity = get_current_equity(SYMBOLS)
    results = []
    for symbol in SYMBOLS:
        try:
            signal, text = analyze_symbol(symbol, btc_last_15m, current_equity)
            results.append((symbol, signal, text))
        except Exception as e:
            results.append((symbol, "ERROR", f"Ошибка анализа {symbol}: {e}"))
    return results


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, "Привет! Портфельный бот работает.\n"
                           f"Монеты: {', '.join(symbol_name(s) for s in SYMBOLS)}\n\n"
                           "Команды:\n/price\n/strong_signal\n/results\n/history")


@bot.message_handler(commands=["price"])
def price(message):
    lines = []
    for symbol in SYMBOLS:
        p = exchange.fetch_ticker(symbol)["last"]
        lines.append(f"{symbol_name(symbol)}: {p} USDT")
    bot.reply_to(message, "\n".join(lines))


@bot.message_handler(commands=["strong_signal"])
def strong_signal(message):
    results = analyze_all_symbols()
    for symbol, signal, text in results:
        bot.reply_to(message, text)


@bot.message_handler(commands=["results"])
def results(message):
    equity = get_current_equity(SYMBOLS)
    lines = [f"РЕЗУЛЬТАТЫ ПОРТФЕЛЯ ({', '.join(symbol_name(s) for s in SYMBOLS)})\n"]
    total_trades, total_closed, total_open = 0, 0, 0
    for symbol in SYMBOLS:
        update_trade_results(symbol)
        f = trades_file(symbol)
        if not os.path.exists(f):
            lines.append(f"{symbol_name(symbol)}: сделок ещё не было")
            continue
        df = pd.read_csv(f)
        if df.empty:
            lines.append(f"{symbol_name(symbol)}: сделок ещё не было")
            continue
        closed = df[~df["status"].isin(["OPEN", "TP1_HIT"])]
        open_count = len(df) - len(closed)
        pnl_sum = pd.to_numeric(df["pnl_usdt"], errors="coerce").fillna(0).sum() if "pnl_usdt" in df.columns else 0
        win_rate = round((pd.to_numeric(closed["pnl_usdt"], errors="coerce").fillna(0) > 0).mean() * 100, 1) if len(closed) else 0
        lines.append(f"{symbol_name(symbol)}: {len(df)} сделок, {len(closed)} закрыто, "
                      f"{open_count} открыто, win rate {win_rate}%, PnL {round(pnl_sum,2)} USDT")
        total_trades += len(df)
        total_closed += len(closed)
        total_open += open_count

    lines.append(f"\nВсего сделок по портфелю: {total_trades} ({total_closed} закрыто, {total_open} открыто)")
    lines.append(f"Стартовый капитал: {sc.ACCOUNT_EQUITY_USDT} USDT")
    lines.append(f"Текущий капитал портфеля: {round(equity, 2)} USDT")
    lines.append(f"Доходность: {round((equity/sc.ACCOUNT_EQUITY_USDT - 1)*100, 2)}%")
    lines.append("\nДеньги НЕ используем. Это paper-статистика.")
    bot.reply_to(message, "\n".join(lines))


@bot.message_handler(commands=["history"])
def history(message):
    lines = ["ИСТОРИЯ ПОСЛЕДНИХ СДЕЛОК\n"]
    for symbol in SYMBOLS:
        update_trade_results(symbol)
        f = trades_file(symbol)
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        lines.append(f"--- {symbol_name(symbol)} ---")
        for _, row in df.tail(5).iterrows():
            lines.append(f"{row.get('time','-')} | {row.get('signal','-')} | {row.get('status','-')} | "
                          f"Entry: {row.get('entry','-')} | PnL: {row.get('pnl_usdt','')} USDT")
    bot.reply_to(message, "\n".join(lines) if len(lines) > 1 else "Истории сделок пока нет.")


def auto_check():
    while True:
        try:
            results = analyze_all_symbols()
            for symbol, signal, text in results:
                if signal in ["LONG", "SHORT"] and CHAT_ID:
                    bot.send_message(CHAT_ID, text)
                print(f"Автопроверка {symbol_name(symbol)} выполнена:", signal, flush=True)
        except Exception as e:
            print("Ошибка автопроверки:", e, flush=True)
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    if not TOKEN or CHAT_ID is None:
        raise RuntimeError("BOT_TOKEN или CHAT_ID не указаны в переменных окружения")
    threading.Thread(target=auto_check, daemon=True).start()
    print("Портфельный бот запущен на", SYMBOLS, "LIVE_TRADING_ENABLED =", LIVE_TRADING_ENABLED, flush=True)
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)
