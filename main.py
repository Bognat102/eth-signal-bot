"""
main.py

Railway-ready ETHUSDT Futures paper bot.

Verified design goals:
1) Binance Futures only.
2) Market check every 5 minutes, aligned to the next 5-minute boundary.
3) Detailed Railway console report for every check and every trade event.

This bot does NOT place real exchange orders. It records paper trades only.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt
import pandas as pd
import telebot
from dotenv import load_dotenv

import strategy_core as sc

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID_RAW = os.getenv("CHAT_ID")
CHAT_ID = int(CHAT_ID_RAW) if CHAT_ID_RAW else None

SYMBOL = "ETH/USDT:USDT"
TIMEFRAME_ENTRY = "5m"
TIMEFRAME_DIRECTION = "15m"

STATE_FILE = Path("state_eth_turtle20.json")
TRADES_FILE = Path("trades_log_ETHUSDT.csv")
DECISIONS_FILE = Path("decisions_log_ETHUSDT.csv")

LIVE_TRADING_ENABLED = False

bot = telebot.TeleBot(TOKEN) if TOKEN else None

exchange = ccxt.binance(
    {
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
        },
    }
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_str() -> str:
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def log_event(title: str, lines: List[str]) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"{now_str()} | {title}", flush=True)
    for line in lines:
        print(line, flush=True)
    print("=" * 72, flush=True)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def default_state() -> Dict[str, Any]:
    return {
        "equity": sc.START_EQUITY_USDT,
        "open_trade": None,
        "trades_today": 0,
        "trades_day": utc_now().strftime("%Y-%m-%d"),
        "last_processed_5m_close": None,
    }


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return default_state()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        base = default_state()
        base.update(data)
        return base
    except Exception as exc:
        log_event("STATE LOAD ERROR", [f"{type(exc).__name__}: {exc}", "Using a fresh paper state."])
        return default_state()


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def reset_daily_counter(state: Dict[str, Any]) -> None:
    today = utc_now().strftime("%Y-%m-%d")
    if state.get("trades_day") != today:
        state["trades_day"] = today
        state["trades_today"] = 0


def fetch_ohlcv(timeframe: str, limit: int) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
    return pd.DataFrame(
        candles,
        columns=["time", "open", "high", "low", "close", "volume"],
    )


def futures_last_price() -> float:
    return float(exchange.fetch_ticker(SYMBOL)["last"])


def funding_pnl_between(start_ms: int, end_ms: int, side: str, notional: float) -> float:
    if end_ms <= start_ms:
        return 0.0
    try:
        history = exchange.fetch_funding_rate_history(SYMBOL, since=start_ms, limit=1000)
    except Exception as exc:
        log_event("FUNDING WARNING", [f"Could not load funding history: {exc}", "Funding PnL set to 0 for this interval."])
        return 0.0

    total_rate = 0.0
    for item in history:
        ts = int(item.get("timestamp") or 0)
        if start_ms < ts <= end_ms:
            total_rate += float(item.get("fundingRate") or 0.0)

    return -notional * total_rate if side == "LONG" else notional * total_rate


def get_direction_context() -> Dict[str, float | str | None]:
    """V16-style context.

    Uses the currently forming 15m candle close and the highest/lowest values
    of the 20 completed 15m candles before it.
    """
    df15 = fetch_ohlcv(TIMEFRAME_DIRECTION, sc.TURTLE_LENGTH + 2)
    current = df15.iloc[-1]
    previous = df15.iloc[-(sc.TURTLE_LENGTH + 1):-1]

    current_close = float(current["close"])
    previous_high = float(previous["high"].max())
    previous_low = float(previous["low"].min())
    direction = sc.turtle_direction(current_close, previous_high, previous_low)

    return {
        "direction": direction,
        "current_15m_close": current_close,
        "previous_20_high": previous_high,
        "previous_20_low": previous_low,
        "current_15m_open_time": int(current["time"]),
    }


def latest_closed_5m() -> pd.Series:
    df5 = fetch_ohlcv(TIMEFRAME_ENTRY, 3)
    return df5.iloc[-2]


def current_equity(state: Dict[str, Any]) -> float:
    return float(state.get("equity", sc.START_EQUITY_USDT))


def open_trade(
    state: Dict[str, Any],
    side: str,
    entry_raw: float,
    signal_candle: pd.Series,
    context: Dict[str, Any],
) -> None:
    equity = current_equity(state)
    entry_exec = sc.execution_price(entry_raw, side, True)
    qty, margin, notional = sc.position_values(entry_exec, equity)
    levels = sc.make_levels(side, entry_exec)
    entry_fee = sc.fee(entry_exec * qty)

    trade = {
        "id": utc_now().strftime("%Y%m%d%H%M%S"),
        "status": "OPEN",
        "stage": 0,
        "side": side,
        "entry_time_ms": int(signal_candle["time"]) + 5 * 60 * 1000,
        "entry_time": now_str(),
        "entry_raw": entry_raw,
        "entry_exec": entry_exec,
        "qty_initial": qty,
        "qty_remaining": qty,
        "margin": margin,
        "notional_initial": notional,
        "notional_remaining": notional,
        "entry_fee": entry_fee,
        "fees_paid": entry_fee,
        "funding_pnl": 0.0,
        "realized_pnl": -entry_fee,
        "stop": levels.stop,
        "initial_stop": levels.stop,
        "tp1": levels.tp1,
        "tp2": levels.tp2,
        "tp3": levels.tp3,
        "tp1_hit": False,
        "tp2_hit": False,
        "signal_5m_open": float(signal_candle["open"]),
        "signal_5m_close": float(signal_candle["close"]),
        "direction_15m_close": context["current_15m_close"],
        "turtle_high_20": context["previous_20_high"],
        "turtle_low_20": context["previous_20_low"],
    }

    state["open_trade"] = trade
    state["trades_today"] = int(state.get("trades_today", 0)) + 1
    save_state(state)

    append_csv(
        TRADES_FILE,
        {
            "event_time": now_str(),
            "trade_id": trade["id"],
            "event": "OPEN",
            "side": side,
            "entry": round(entry_exec, 6),
            "qty": round(qty, 8),
            "margin_usdt": round(margin, 6),
            "notional_usdt": round(notional, 6),
            "stop": round(levels.stop, 6),
            "tp1": round(levels.tp1, 6),
            "tp2": round(levels.tp2, 6),
            "tp3": round(levels.tp3, 6),
            "fees_paid": round(entry_fee, 6),
            "funding_pnl": 0.0,
            "realized_pnl": round(-entry_fee, 6),
            "equity": round(equity, 6),
        },
    )

    lines = [
        f"SYMBOL: {SYMBOL}",
        "MARKET: Binance Futures",
        f"SIDE: {side}",
        f"ENTRY RAW: {entry_raw:.6f}",
        f"ENTRY EXEC (slippage included): {entry_exec:.6f}",
        f"EQUITY BEFORE: {equity:.6f} USDT",
        f"MARGIN: {margin:.6f} USDT ({sc.MARGIN_PCT:.2f}%)",
        f"LEVERAGE: {sc.LEVERAGE:.0f}x",
        f"NOTIONAL: {notional:.6f} USDT",
        f"QTY: {qty:.8f} ETH",
        f"STOP: {levels.stop:.6f} ({sc.INITIAL_STOP_PCT:.2f}%)",
        f"TP1: {levels.tp1:.6f} | close 50%",
        f"TP2: {levels.tp2:.6f} | close 30%",
        f"TP3: {levels.tp3:.6f} | close 20%",
        f"ENTRY FEE: {entry_fee:.6f} USDT",
        f"15M CLOSE: {context['current_15m_close']:.6f}",
        f"TURTLE HIGH 20: {context['previous_20_high']:.6f}",
        f"TURTLE LOW 20: {context['previous_20_low']:.6f}",
        f"5M SIGNAL O/C: {float(signal_candle['open']):.6f} / {float(signal_candle['close']):.6f}",
        "MODE: PAPER ONLY",
    ]
    log_event("TRADE OPENED", lines)

    if CHAT_ID:
        bot.send_message(
            CHAT_ID,
            "\n".join(
                [
                    f"ETHUSDT {side}",
                    f"Entry: {entry_exec:.2f}",
                    f"Stop: {levels.stop:.2f}",
                    f"TP1: {levels.tp1:.2f}",
                    f"TP2: {levels.tp2:.2f}",
                    f"TP3: {levels.tp3:.2f}",
                    f"Margin: {margin:.2f} USDT | {sc.LEVERAGE:.0f}x",
                    "Paper mode",
                ]
            ),
        )


def close_fraction(
    trade: Dict[str, Any],
    raw_exit: float,
    fraction: float,
    event_name: str,
    event_time_ms: int,
) -> None:
    qty = min(float(trade["qty_initial"]) * fraction, float(trade["qty_remaining"]))
    side = trade["side"]
    exit_exec = sc.execution_price(raw_exit, side, False)
    exit_fee = sc.fee(exit_exec * qty)

    funding = funding_pnl_between(
        int(trade["entry_time_ms"]),
        event_time_ms,
        side,
        float(trade["notional_remaining"]),
    )
    trade["funding_pnl"] = float(trade.get("funding_pnl", 0.0)) + funding
    trade["fees_paid"] = float(trade["fees_paid"]) + exit_fee
    trade["realized_pnl"] = float(trade["realized_pnl"]) + sc.gross_pnl(
        side,
        float(trade["entry_exec"]),
        exit_exec,
        qty,
    ) - exit_fee

    trade["qty_remaining"] = float(trade["qty_remaining"]) - qty
    trade["notional_remaining"] = float(trade["qty_remaining"]) * float(trade["entry_exec"])

    append_csv(
        TRADES_FILE,
        {
            "event_time": now_str(),
            "trade_id": trade["id"],
            "event": event_name,
            "side": side,
            "entry": round(float(trade["entry_exec"]), 6),
            "qty": round(qty, 8),
            "margin_usdt": round(float(trade["margin"]), 6),
            "notional_usdt": round(float(trade["notional_remaining"]), 6),
            "stop": round(float(trade["stop"]), 6),
            "tp1": round(float(trade["tp1"]), 6),
            "tp2": round(float(trade["tp2"]), 6),
            "tp3": round(float(trade["tp3"]), 6),
            "fees_paid": round(float(trade["fees_paid"]), 6),
            "funding_pnl": round(float(trade["funding_pnl"]), 6),
            "realized_pnl": round(float(trade["realized_pnl"]), 6),
            "equity": "",
        },
    )

    log_event(
        event_name,
        [
            f"TRADE ID: {trade['id']}",
            f"SIDE: {side}",
            f"EXIT EXEC: {exit_exec:.6f}",
            f"CLOSED QTY: {qty:.8f}",
            f"QTY REMAINING: {float(trade['qty_remaining']):.8f}",
            f"TOTAL FEES: {float(trade['fees_paid']):.6f} USDT",
            f"TOTAL FUNDING: {float(trade['funding_pnl']):.6f} USDT",
            f"REALIZED PNL SO FAR: {float(trade['realized_pnl']):.6f} USDT",
        ],
    )


def finalize_trade(
    state: Dict[str, Any],
    raw_exit: float,
    status: str,
    event_time_ms: int,
) -> None:
    trade = state["open_trade"]
    side = trade["side"]
    qty = float(trade["qty_remaining"])
    exit_exec = sc.execution_price(raw_exit, side, False)
    exit_fee = sc.fee(exit_exec * qty)

    funding = funding_pnl_between(
        int(trade["entry_time_ms"]),
        event_time_ms,
        side,
        float(trade["notional_remaining"]),
    )
    trade["funding_pnl"] = float(trade.get("funding_pnl", 0.0)) + funding
    trade["fees_paid"] = float(trade["fees_paid"]) + exit_fee

    net_pnl = (
        float(trade["realized_pnl"])
        + sc.gross_pnl(side, float(trade["entry_exec"]), exit_exec, qty)
        - exit_fee
        + float(trade["funding_pnl"])
    )

    equity_before = current_equity(state)
    equity_after = equity_before + net_pnl
    state["equity"] = equity_after

    append_csv(
        TRADES_FILE,
        {
            "event_time": now_str(),
            "trade_id": trade["id"],
            "event": status,
            "side": side,
            "entry": round(float(trade["entry_exec"]), 6),
            "qty": round(qty, 8),
            "margin_usdt": round(float(trade["margin"]), 6),
            "notional_usdt": 0.0,
            "stop": round(float(trade["stop"]), 6),
            "tp1": round(float(trade["tp1"]), 6),
            "tp2": round(float(trade["tp2"]), 6),
            "tp3": round(float(trade["tp3"]), 6),
            "fees_paid": round(float(trade["fees_paid"]), 6),
            "funding_pnl": round(float(trade["funding_pnl"]), 6),
            "realized_pnl": round(net_pnl, 6),
            "equity": round(equity_after, 6),
        },
    )

    log_event(
        "TRADE CLOSED",
        [
            f"TRADE ID: {trade['id']}",
            f"STATUS: {status}",
            f"SIDE: {side}",
            f"ENTRY: {float(trade['entry_exec']):.6f}",
            f"EXIT: {exit_exec:.6f}",
            f"TOTAL FEES: {float(trade['fees_paid']):.6f} USDT",
            f"TOTAL FUNDING: {float(trade['funding_pnl']):.6f} USDT",
            f"NET PNL: {net_pnl:.6f} USDT",
            f"EQUITY BEFORE: {equity_before:.6f} USDT",
            f"EQUITY AFTER: {equity_after:.6f} USDT",
            f"RETURN FROM START: {(equity_after / sc.START_EQUITY_USDT - 1) * 100:.4f}%",
        ],
    )

    if CHAT_ID:
        bot.send_message(
            CHAT_ID,
            "\n".join(
                [
                    f"ETHUSDT trade closed: {status}",
                    f"Net PnL: {net_pnl:.2f} USDT",
                    f"Equity: {equity_after:.2f} USDT",
                ]
            ),
        )

    state["open_trade"] = None
    save_state(state)


def manage_open_trade(state: Dict[str, Any], candle: pd.Series) -> None:
    trade = state.get("open_trade")
    if not trade:
        return

    side = trade["side"]
    high = float(candle["high"])
    low = float(candle["low"])
    event_time_ms = int(candle["time"]) + 5 * 60 * 1000

    stop = float(trade["stop"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])
    tp3 = float(trade["tp3"])
    stage = int(trade["stage"])

    if stage == 0:
        stop_hit = low <= stop if side == "LONG" else high >= stop
        tp1_hit = high >= tp1 if side == "LONG" else low <= tp1

        # Conservative same-candle rule: stop first.
        if stop_hit:
            finalize_trade(state, stop, "STOP", event_time_ms)
            return

        if tp1_hit:
            close_fraction(trade, tp1, sc.TP1_CLOSE_FRACTION, "TP1_HIT", event_time_ms)
            trade["stage"] = 1
            trade["tp1_hit"] = True
            trade["stop"] = sc.remaining_position_break_even(side, float(trade["entry_exec"]))
            save_state(state)
            log_event(
                "STOP MOVED AFTER TP1",
                [
                    "50% position is already closed with profit.",
                    f"Remaining 50% stop moved to its own cost-covered BE: {float(trade['stop']):.6f}",
                ],
            )
            return

    elif stage == 1:
        stop_hit = low <= stop if side == "LONG" else high >= stop
        tp2_hit = high >= tp2 if side == "LONG" else low <= tp2

        if stop_hit:
            finalize_trade(state, stop, "TP1_BE", event_time_ms)
            return

        if tp2_hit:
            close_fraction(trade, tp2, sc.TP2_CLOSE_FRACTION, "TP2_HIT", event_time_ms)
            trade["stage"] = 2
            trade["tp2_hit"] = True
            save_state(state)
            return

    else:
        stop_hit = low <= stop if side == "LONG" else high >= stop
        tp3_hit = high >= tp3 if side == "LONG" else low <= tp3

        if stop_hit:
            finalize_trade(state, stop, "TP2_BE", event_time_ms)
            return

        if tp3_hit:
            finalize_trade(state, tp3, "TP3_HIT", event_time_ms)
            return


def analyze_market(state: Dict[str, Any]) -> None:
    reset_daily_counter(state)
    candle = latest_closed_5m()
    candle_close_ms = int(candle["time"]) + 5 * 60 * 1000

    if state.get("last_processed_5m_close") == candle_close_ms:
        return

    state["last_processed_5m_close"] = candle_close_ms

    manage_open_trade(state, candle)

    context = get_direction_context()
    direction = context["direction"]
    signal_ok = sc.five_minute_entry_signal(
        direction,
        float(candle["open"]),
        float(candle["close"]),
    )
    price = futures_last_price()

    append_csv(
        DECISIONS_FILE,
        {
            "time": now_str(),
            "market": "Binance Futures",
            "symbol": SYMBOL,
            "five_minute_candle_time": candle_close_ms,
            "direction": direction or "NONE",
            "current_15m_close": round(float(context["current_15m_close"]), 6),
            "turtle_high_20": round(float(context["previous_20_high"]), 6),
            "turtle_low_20": round(float(context["previous_20_low"]), 6),
            "five_minute_open": round(float(candle["open"]), 6),
            "five_minute_close": round(float(candle["close"]), 6),
            "entry_signal": signal_ok,
            "futures_last_price": round(price, 6),
            "open_trade": bool(state.get("open_trade")),
            "trades_today": int(state.get("trades_today", 0)),
        },
    )

    log_event(
        "5-MINUTE MARKET CHECK",
        [
            f"MARKET: Binance Futures",
            f"SYMBOL: {SYMBOL}",
            f"5M CLOSED CANDLE: {pd.to_datetime(candle_close_ms, unit='ms', utc=True)}",
            f"5M O/H/L/C: {float(candle['open']):.6f} / {float(candle['high']):.6f} / {float(candle['low']):.6f} / {float(candle['close']):.6f}",
            f"CURRENT FUTURES PRICE: {price:.6f}",
            f"15M CURRENT CLOSE: {float(context['current_15m_close']):.6f}",
            f"TURTLE HIGH 20: {float(context['previous_20_high']):.6f}",
            f"TURTLE LOW 20: {float(context['previous_20_low']):.6f}",
            f"DIRECTION: {direction or 'NONE'}",
            f"5M ENTRY CONFIRMATION: {signal_ok}",
            f"OPEN TRADE: {bool(state.get('open_trade'))}",
            f"TRADES TODAY: {int(state.get('trades_today', 0))}/{sc.MAX_TRADES_PER_DAY}",
            f"EQUITY: {current_equity(state):.6f} USDT",
        ],
    )

    if (
        not state.get("open_trade")
        and signal_ok
        and direction in {"LONG", "SHORT"}
        and int(state.get("trades_today", 0)) < sc.MAX_TRADES_PER_DAY
    ):
        open_trade(state, direction, price, candle, context)

    save_state(state)


def seconds_to_next_five_minute_boundary(buffer_seconds: int = 3) -> float:
    now = time.time()
    next_boundary = ((int(now) // 300) + 1) * 300 + buffer_seconds
    return max(1.0, next_boundary - now)


def auto_check() -> None:
    state = load_state()

    while True:
        try:
            analyze_market(state)
        except Exception as exc:
            log_event(
                "AUTO CHECK ERROR",
                [
                    f"{type(exc).__name__}: {exc}",
                    "The bot will retry at the next 5-minute boundary.",
                ],
            )

        time.sleep(seconds_to_next_five_minute_boundary())


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "ETH Turtle 20 paper bot is running.\n"
        "Market: Binance Futures\n"
        "Check interval: every 5 minutes\n"
        "Commands: /price /status /history",
    )


@bot.message_handler(commands=["price"])
def price(message):
    bot.reply_to(message, f"ETHUSDT Futures: {futures_last_price():.2f} USDT")


@bot.message_handler(commands=["status"])
def status(message):
    state = load_state()
    trade = state.get("open_trade")
    lines = [
        f"Equity: {current_equity(state):.2f} USDT",
        f"Trades today: {int(state.get('trades_today', 0))}/{sc.MAX_TRADES_PER_DAY}",
        f"Open trade: {'YES' if trade else 'NO'}",
    ]
    if trade:
        lines.extend(
            [
                f"Side: {trade['side']}",
                f"Entry: {float(trade['entry_exec']):.2f}",
                f"Stop: {float(trade['stop']):.2f}",
                f"TP1/TP2/TP3: {float(trade['tp1']):.2f} / {float(trade['tp2']):.2f} / {float(trade['tp3']):.2f}",
                f"Remaining qty: {float(trade['qty_remaining']):.6f}",
            ]
        )
    bot.reply_to(message, "\n".join(lines))


@bot.message_handler(commands=["history"])
def history(message):
    if not TRADES_FILE.exists():
        bot.reply_to(message, "No trades yet.")
        return
    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        bot.reply_to(message, "No trades yet.")
        return
    lines = ["LAST TRADE EVENTS"]
    for _, row in df.tail(10).iterrows():
        lines.append(
            f"{row.get('event_time','-')} | {row.get('event','-')} | "
            f"{row.get('side','-')} | PnL: {row.get('realized_pnl','-')}"
        )
    bot.reply_to(message, "\n".join(lines))


if __name__ == "__main__":
    if not TOKEN or CHAT_ID is None:
        raise RuntimeError("BOT_TOKEN or CHAT_ID is missing in Railway Variables.")

    log_event(
        "BOT START",
        [
            f"SYMBOL: {SYMBOL}",
            "MARKET: Binance Futures",
            "CHECK: every 5 minutes, aligned to the next boundary",
            f"STRATEGY: Turtle {sc.TURTLE_LENGTH}, 15m direction + 5m entry",
            f"MARGIN: {sc.MARGIN_PCT}% of current equity",
            f"LEVERAGE: {sc.LEVERAGE}x",
            f"LIVE_TRADING_ENABLED: {LIVE_TRADING_ENABLED}",
            "MODE: PAPER ONLY",
        ],
    )

    threading.Thread(target=auto_check, daemon=True).start()
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)
