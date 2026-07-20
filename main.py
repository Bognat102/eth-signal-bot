"""
main.py

ETH Turtle 5 paper bot for Railway.

This file intentionally contains BOTH:
- the already-working Railway / Telegram / Binance Futures infrastructure pattern;
- the full Turtle 5 trading logic.

Only this one file needs to replace the current main.py in GitHub.

IMPORTANT:
- PAPER MODE ONLY. No real Binance orders are placed.
- Binance Futures data only.
- Market check every 60 seconds.
- Detailed Railway log for every market check and every trade event.
"""

from __future__ import annotations

import csv
import json
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import ccxt
import pandas as pd
import telebot
from dotenv import load_dotenv


# ============================================================================
# ENVIRONMENT / INFRASTRUCTURE
# ============================================================================

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID_RAW = os.getenv("CHAT_ID")
CHAT_ID = int(CHAT_ID_RAW) if CHAT_ID_RAW else None

bot = telebot.TeleBot(TOKEN) if TOKEN else None

# Preserve the proven Binance Futures connection pattern from the old bot.
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"
    }
})

SYMBOL = "ETH/USDT:USDT"
SLEEP_SECONDS = 60
LIVE_TRADING_ENABLED = False

TRADES_FILE = Path("trades_log_ETHUSDT.csv")
DECISIONS_FILE = Path("decisions_log_ETHUSDT.csv")
STATE_FILE = Path("state_ETHUSDT.json")


# ============================================================================
# V16 STRATEGY CONSTANTS — DO NOT CHANGE WITHOUT A NEW BACKTEST
# ============================================================================

START_EQUITY_USDT = 1000.0
POSITION_MARGIN_PCT = 5.0
LEVERAGE = 10.0

TAKER_FEE_PCT = 0.045
SLIPPAGE_PCT = 0.02

TURTLE_LENGTH = 5

INITIAL_STOP_PCT = 0.60
TP1_PCT = 0.50
TP2_PCT = 1.00
TP3_PCT = 1.50

TP1_CLOSE_FRACTION = 0.50
TP2_CLOSE_FRACTION = 0.30
TP3_CLOSE_FRACTION = 0.20

MAX_TRADES_PER_DAY = 40


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_str() -> str:
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def log_block(title: str, **values: Any) -> None:
    print("\n" + "=" * 74, flush=True)
    print(f"{now_str()} | {title}", flush=True)
    for key, value in values.items():
        print(f"{key}: {value}", flush=True)
    print("=" * 74, flush=True)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def execution_price(raw_price: float, side: str, is_entry: bool) -> float:
    slip = SLIPPAGE_PCT / 100.0
    is_buy = (side == "LONG" and is_entry) or (
        side == "SHORT" and not is_entry
    )
    return raw_price * (1 + slip if is_buy else 1 - slip)


def fee(notional: float) -> float:
    return abs(notional) * TAKER_FEE_PCT / 100.0


def gross_pnl(side: str, entry: float, exit_price: float, qty: float) -> float:
    if side == "LONG":
        return (exit_price - entry) * qty
    return (entry - exit_price) * qty


def position_values(entry: float, equity: float):
    margin = equity * POSITION_MARGIN_PCT / 100.0
    notional = margin * LEVERAGE
    qty = notional / entry if entry > 0 else 0.0
    return qty, margin, notional


def make_levels(side: str, entry: float):
    if side == "LONG":
        return {
            "stop": entry * (1 - INITIAL_STOP_PCT / 100.0),
            "tp1": entry * (1 + TP1_PCT / 100.0),
            "tp2": entry * (1 + TP2_PCT / 100.0),
            "tp3": entry * (1 + TP3_PCT / 100.0),
        }
    return {
        "stop": entry * (1 + INITIAL_STOP_PCT / 100.0),
        "tp1": entry * (1 - TP1_PCT / 100.0),
        "tp2": entry * (1 - TP2_PCT / 100.0),
        "tp3": entry * (1 - TP3_PCT / 100.0),
    }


def remaining_position_break_even(side: str, entry_exec: float) -> float:
    """
    Break-even for ONLY the remaining position after TP1.

    The already-realized TP1 profit is preserved.
    The remaining position covers its own future exit fee and modeled slippage.
    """
    fee_rate = TAKER_FEE_PCT / 100.0
    slip_rate = SLIPPAGE_PCT / 100.0

    if side == "LONG":
        target_exec = entry_exec / (1 - fee_rate)
        return target_exec / (1 - slip_rate)

    target_exec = entry_exec / (1 + fee_rate)
    return target_exec / (1 + slip_rate)


# ============================================================================
# STATE
# ============================================================================

def default_state() -> Dict[str, Any]:
    return {
        "equity": START_EQUITY_USDT,
        "open_trade": None,
        "trades_today": 0,
        "trades_day": utc_now().strftime("%Y-%m-%d"),
        "last_processed_1m_open_time": None,
        "last_check_time": None,
        "last_direction": "NONE",
        "last_confirmation": False,
        "last_signal": "NO TRADE",
        "last_futures_price": None,
        "last_reason": "Бот ещё не выполнил первую проверку",
        "checks_today": 0,
        "checks_day": utc_now().strftime("%Y-%m-%d"),
        "last_completed_direction": "NONE",
        "entry_lock_direction": "NONE",
    }


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return default_state()

    try:
        loaded = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state = default_state()
        state.update(loaded)
        return state
    except Exception as exc:
        log_block(
            "STATE LOAD ERROR",
            error=f"{type(exc).__name__}: {exc}",
            action="A new paper state will be used",
        )
        return default_state()


def save_state(state: Dict[str, Any]) -> None:
    temp_file = STATE_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(STATE_FILE)


def reset_daily_counter(state: Dict[str, Any]) -> None:
    today = utc_now().strftime("%Y-%m-%d")
    if state.get("trades_day") != today:
        state["trades_day"] = today
        state["trades_today"] = 0

    if state.get("checks_day") != today:
        state["checks_day"] = today
        state["checks_today"] = 0


# ============================================================================
# BINANCE FUTURES DATA
# ============================================================================

def get_data(timeframe: str, limit: int) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe=timeframe,
        limit=limit,
    )
    return pd.DataFrame(
        candles,
        columns=["time", "open", "high", "low", "close", "volume"],
    )


def current_futures_price() -> float:
    return float(exchange.fetch_ticker(SYMBOL)["last"])


def latest_closed_1m() -> pd.Series:
    df = get_data("1m", 3)
    if len(df) < 3:
        raise RuntimeError("Not enough 1m candles")
    return df.iloc[-2]


def turtle_15m_context() -> Dict[str, Any]:
    """
    Strategy #2:
    Direction is determined by breakout of the previous closed 15m candle.
    """
    df = get_data("15m", 3)
    if len(df) < 2:
        raise RuntimeError("Not enough 15m candles")

    current = df.iloc[-1]
    previous = df.iloc[-2]

    current_close = float(current["close"])
    previous_high = float(previous["high"])
    previous_low = float(previous["low"])

    direction: Optional[str] = None
    if current_close > previous_high:
        direction = "LONG"
    elif current_close < previous_low:
        direction = "SHORT"

    return {
        "direction": direction,
        "current_close": current_close,
        "previous_high": previous_high,
        "previous_low": previous_low,
        "current_15m_open_time": int(current["time"]),
    }


def update_funding(trade: Dict[str, Any], event_time_ms: int) -> None:
    """
    Add funding only for the interval not processed before.

    This prevents double-counting after TP1 and TP2.
    """
    last_time = int(trade.get("last_funding_time_ms", trade["entry_time_ms"]))
    if event_time_ms <= last_time:
        return

    try:
        history = exchange.fetch_funding_rate_history(
            SYMBOL,
            since=last_time,
            limit=1000,
        )
    except Exception as exc:
        log_block(
            "FUNDING WARNING",
            error=f"{type(exc).__name__}: {exc}",
            action="Funding for this interval is temporarily recorded as 0",
        )
        trade["last_funding_time_ms"] = event_time_ms
        return

    rate_sum = 0.0
    for item in history:
        timestamp = int(item.get("timestamp") or 0)
        if last_time < timestamp <= event_time_ms:
            rate_sum += float(item.get("fundingRate") or 0.0)

    notional = float(trade["notional_remaining"])
    side = trade["side"]
    funding_piece = -notional * rate_sum if side == "LONG" else notional * rate_sum

    trade["funding_pnl"] = float(trade.get("funding_pnl", 0.0)) + funding_piece
    trade["last_funding_time_ms"] = event_time_ms


# ============================================================================
# TRADE EVENTS
# ============================================================================

def send_telegram(text: str) -> None:
    if CHAT_ID and bot:
        bot.send_message(CHAT_ID, text)


def save_trade_event(trade: Dict[str, Any], event: str, event_pnl: Any = "") -> None:
    append_csv(
        TRADES_FILE,
        {
            "event_time": now_str(),
            "trade_id": trade["id"],
            "event": event,
            "side": trade["side"],
            "entry": round(float(trade["entry_exec"]), 6),
            "stop": round(float(trade["stop"]), 6),
            "tp1": round(float(trade["tp1"]), 6),
            "tp2": round(float(trade["tp2"]), 6),
            "tp3": round(float(trade["tp3"]), 6),
            "qty_initial": round(float(trade["qty_initial"]), 8),
            "qty_remaining": round(float(trade["qty_remaining"]), 8),
            "margin_usdt": round(float(trade["margin"]), 6),
            "notional_remaining": round(float(trade["notional_remaining"]), 6),
            "fees_paid": round(float(trade["fees_paid"]), 6),
            "funding_pnl": round(float(trade["funding_pnl"]), 6),
            "realized_pnl": round(float(trade["realized_pnl"]), 6),
            "event_pnl": event_pnl,
        },
    )


def open_trade(
    state: Dict[str, Any],
    side: str,
    raw_entry: float,
    candle_1m: pd.Series,
    context: Dict[str, Any],
) -> None:
    equity = float(state["equity"])
    entry_exec = execution_price(raw_entry, side, True)
    qty, margin, notional = position_values(entry_exec, equity)
    levels = make_levels(side, entry_exec)
    entry_fee = fee(entry_exec * qty)

    trade = {
        "id": utc_now().strftime("%Y%m%d%H%M%S"),
        "status": "OPEN",
        "stage": 0,
        "side": side,
        "entry_time": now_str(),
        "entry_time_ms": int(time.time() * 1000),
        "last_funding_time_ms": int(time.time() * 1000),
        "entry_raw": raw_entry,
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
        "stop": levels["stop"],
        "initial_stop": levels["stop"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "tp3": levels["tp3"],
        "turtle_current_15m_close": context["current_close"],
        "turtle_previous_high_5": context["previous_high"],
        "turtle_previous_low_5": context["previous_low"],
        "signal_1m_open": float(candle_1m["open"]),
        "signal_1m_close": float(candle_1m["close"]),
    }

    state["open_trade"] = trade
    state["trades_today"] = int(state["trades_today"]) + 1
    save_state(state)
    save_trade_event(trade, "OPEN")

    log_block(
        "TRADE OPENED",
        market="Binance Futures",
        symbol=SYMBOL,
        side=side,
        entry_raw=f"{raw_entry:.6f}",
        entry_exec=f"{entry_exec:.6f}",
        equity_before=f"{equity:.6f} USDT",
        margin=f"{margin:.6f} USDT ({POSITION_MARGIN_PCT:.2f}%)",
        leverage=f"{LEVERAGE:.0f}x",
        notional=f"{notional:.6f} USDT",
        qty=f"{qty:.8f} ETH",
        stop=f"{levels['stop']:.6f}",
        tp1=f"{levels['tp1']:.6f} | close 50%",
        tp2=f"{levels['tp2']:.6f} | close 30%",
        tp3=f"{levels['tp3']:.6f} | close 20%",
        entry_fee=f"{entry_fee:.6f} USDT",
        turtle_15m_close=f"{context['current_close']:.6f}",
        turtle_high_5=f"{context['previous_high']:.6f}",
        turtle_low_5=f"{context['previous_low']:.6f}",
        signal_1m=f"{float(candle_1m['open']):.6f} -> {float(candle_1m['close']):.6f}",
        mode="PAPER ONLY",
    )

    send_telegram(
        "\n".join([
            f"ETHUSDT {side}",
            f"Entry: {entry_exec:.2f}",
            f"Stop: {levels['stop']:.2f}",
            f"TP1: {levels['tp1']:.2f} (50%)",
            f"TP2: {levels['tp2']:.2f} (30%)",
            f"TP3: {levels['tp3']:.2f} (20%)",
            f"Margin: {margin:.2f} USDT",
            f"Leverage: {LEVERAGE:.0f}x",
            "Paper mode",
        ])
    )


def close_part(
    state: Dict[str, Any],
    raw_exit: float,
    fraction: float,
    event: str,
    event_time_ms: int,
) -> None:
    trade = state["open_trade"]
    update_funding(trade, event_time_ms)

    side = trade["side"]
    qty = min(
        float(trade["qty_initial"]) * fraction,
        float(trade["qty_remaining"]),
    )
    exit_exec = execution_price(raw_exit, side, False)
    exit_fee = fee(exit_exec * qty)
    pnl_piece = gross_pnl(
        side,
        float(trade["entry_exec"]),
        exit_exec,
        qty,
    ) - exit_fee

    trade["fees_paid"] = float(trade["fees_paid"]) + exit_fee
    trade["realized_pnl"] = float(trade["realized_pnl"]) + pnl_piece
    trade["qty_remaining"] = float(trade["qty_remaining"]) - qty
    trade["notional_remaining"] = (
        float(trade["qty_remaining"]) * float(trade["entry_exec"])
    )

    save_trade_event(trade, event, round(pnl_piece, 6))
    save_state(state)

    log_block(
        event,
        trade_id=trade["id"],
        side=side,
        exit_exec=f"{exit_exec:.6f}",
        closed_qty=f"{qty:.8f}",
        qty_remaining=f"{float(trade['qty_remaining']):.8f}",
        event_pnl=f"{pnl_piece:.6f} USDT",
        total_fees=f"{float(trade['fees_paid']):.6f} USDT",
        total_funding=f"{float(trade['funding_pnl']):.6f} USDT",
        realized_pnl_so_far=f"{float(trade['realized_pnl']):.6f} USDT",
    )


def finalize_trade(
    state: Dict[str, Any],
    raw_exit: float,
    status: str,
    event_time_ms: int,
) -> None:
    trade = state["open_trade"]
    update_funding(trade, event_time_ms)

    side = trade["side"]
    qty = float(trade["qty_remaining"])
    exit_exec = execution_price(raw_exit, side, False)
    exit_fee = fee(exit_exec * qty)
    final_piece = gross_pnl(
        side,
        float(trade["entry_exec"]),
        exit_exec,
        qty,
    ) - exit_fee

    trade["fees_paid"] = float(trade["fees_paid"]) + exit_fee
    trade["realized_pnl"] = float(trade["realized_pnl"]) + final_piece

    net_pnl = float(trade["realized_pnl"]) + float(trade["funding_pnl"])
    equity_before = float(state["equity"])
    equity_after = equity_before + net_pnl
    state["equity"] = equity_after

    trade["qty_remaining"] = 0.0
    trade["notional_remaining"] = 0.0
    trade["status"] = status

    save_trade_event(trade, status, round(final_piece, 6))

    log_block(
        "TRADE CLOSED",
        trade_id=trade["id"],
        status=status,
        side=side,
        entry=f"{float(trade['entry_exec']):.6f}",
        exit=f"{exit_exec:.6f}",
        total_fees=f"{float(trade['fees_paid']):.6f} USDT",
        total_funding=f"{float(trade['funding_pnl']):.6f} USDT",
        net_pnl=f"{net_pnl:.6f} USDT",
        equity_before=f"{equity_before:.6f} USDT",
        equity_after=f"{equity_after:.6f} USDT",
        return_from_start=f"{(equity_after / START_EQUITY_USDT - 1) * 100:.4f}%",
    )

    send_telegram(
        "\n".join([
            f"ETHUSDT closed: {status}",
            f"Net PnL: {net_pnl:.2f} USDT",
            f"Equity: {equity_after:.2f} USDT",
        ])
    )

    state["last_completed_direction"] = trade["side"]
    state["entry_lock_direction"] = trade["side"]
    state["open_trade"] = None
    save_state(state)


def manage_open_trade(state: Dict[str, Any], candle: pd.Series) -> None:
    trade = state.get("open_trade")
    if not trade:
        return

    side = trade["side"]
    high = float(candle["high"])
    low = float(candle["low"])
    event_time_ms = int(candle["time"]) + 1 * 60 * 1000

    stop = float(trade["stop"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])
    tp3 = float(trade["tp3"])
    stage = int(trade["stage"])

    if stage == 0:
        stop_hit = low <= stop if side == "LONG" else high >= stop
        tp1_hit = high >= tp1 if side == "LONG" else low <= tp1

        # Same conservative rule as the backtest: if both are inside one candle,
        # assume the stop happened first.
        if stop_hit:
            finalize_trade(state, stop, "STOP", event_time_ms)
            return

        if tp1_hit:
            close_part(
                state,
                tp1,
                TP1_CLOSE_FRACTION,
                "TP1_HIT",
                event_time_ms,
            )
            trade = state["open_trade"]
            trade["stage"] = 1
            trade["stop"] = remaining_position_break_even(
                side,
                float(trade["entry_exec"]),
            )
            save_state(state)

            log_block(
                "STOP MOVED AFTER TP1",
                closed_position="50%",
                remaining_position="50%",
                new_stop=f"{float(trade['stop']):.6f}",
                rule="Remaining 50% has its own cost-covered break-even",
                tp1_profit="Preserved",
            )
            return

    elif stage == 1:
        stop_hit = low <= stop if side == "LONG" else high >= stop
        tp2_hit = high >= tp2 if side == "LONG" else low <= tp2

        if stop_hit:
            finalize_trade(state, stop, "TP1_BE", event_time_ms)
            return

        if tp2_hit:
            close_part(
                state,
                tp2,
                TP2_CLOSE_FRACTION,
                "TP2_HIT",
                event_time_ms,
            )
            trade = state["open_trade"]
            trade["stage"] = 2
            # Stop deliberately remains at the TP1 break-even level.
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


# ============================================================================
# MARKET ANALYSIS
# ============================================================================

def market_snapshot() -> Dict[str, Any]:
    candle = latest_closed_1m()
    context = turtle_15m_context()
    direction = context["direction"]
    candle_open = float(candle["open"])
    candle_close = float(candle["close"])
    confirmation = (
        direction == "LONG" and candle_close > candle_open
    ) or (
        direction == "SHORT" and candle_close < candle_open
    )
    return {
        "candle": candle,
        "context": context,
        "direction": direction,
        "confirmation": confirmation,
        "futures_price": current_futures_price(),
    }


def no_trade_reason(state, direction, confirmation):
    if state.get("open_trade"):
        t = state["open_trade"]
        return f"Уже есть открытая сделка {t['side']} на стадии {t['stage']}"
    if int(state.get("trades_today", 0)) >= MAX_TRADES_PER_DAY:
        return f"Достигнут дневной лимит {MAX_TRADES_PER_DAY} сделок"
    if direction == state.get("entry_lock_direction","NONE") and direction in {"LONG","SHORT"}:
        return f"Повторный вход в тот же пробой {direction} запрещён до возврата направления в NONE"
    if direction is None:
        return (
            "Нет пробоя Turtle 5: текущая 15M цена находится "
            "между максимумом и минимумом предыдущих 5 свечей"
        )
    if not confirmation:
        if direction == "LONG":
            return "Есть направление LONG, но последняя закрытая 1M свеча не бычья"
        return "Есть направление SHORT, но последняя закрытая 1M свеча не медвежья"
    return "Все условия входа выполнены"


def analyze_market(state: Dict[str, Any]) -> None:
    reset_daily_counter(state)

    snapshot = market_snapshot()
    candle = snapshot["candle"]
    context = snapshot["context"]
    direction = snapshot["direction"]
    if direction is None and state.get("entry_lock_direction") != "NONE":
        state["entry_lock_direction"] = "NONE"
    one_minute_confirmation = snapshot["confirmation"]
    futures_price = snapshot["futures_price"]

    candle_open_time = int(candle["time"])

    if state.get("last_processed_1m_open_time") == candle_open_time:
        return

    state["last_processed_1m_open_time"] = candle_open_time
    manage_open_trade(state, candle)

    candle_open = float(candle["open"])
    candle_close = float(candle["close"])
    reason = no_trade_reason(state, direction, one_minute_confirmation)

    state["last_check_time"] = now_str()
    state["last_direction"] = direction or "NONE"
    state["last_confirmation"] = bool(one_minute_confirmation)
    state["last_futures_price"] = futures_price
    state["checks_today"] = int(state.get("checks_today", 0)) + 1

    append_csv(
        DECISIONS_FILE,
        {
            "time": now_str(),
            "market": "Binance Futures",
            "symbol": SYMBOL,
            "one_minute_open_time": candle_open_time,
            "one_minute_open": round(candle_open, 6),
            "one_minute_high": round(float(candle["high"]), 6),
            "one_minute_low": round(float(candle["low"]), 6),
            "one_minute_close": round(candle_close, 6),
            "current_15m_close": round(float(context["current_close"]), 6),
            "turtle_high_5": round(float(context["previous_high"]), 6),
            "turtle_low_5": round(float(context["previous_low"]), 6),
            "direction": direction or "NONE",
            "one_minute_confirmation": one_minute_confirmation,
            "futures_last_price": round(futures_price, 6),
            "open_trade": bool(state.get("open_trade")),
            "trades_today": int(state["trades_today"]),
            "equity": round(float(state["equity"]), 6),
        },
    )

    log_block(
        "1-MINUTE MARKET CHECK",
        market="Binance Futures",
        symbol=SYMBOL,
        candle_time=pd.to_datetime(
            candle_open_time,
            unit="ms",
            utc=True,
        ),
        candle_ohlc=(
            f"{candle_open:.6f} / {float(candle['high']):.6f} / "
            f"{float(candle['low']):.6f} / {candle_close:.6f}"
        ),
        current_futures_price=f"{futures_price:.6f}",
        current_15m_close=f"{float(context['current_close']):.6f}",
        turtle_high_5=f"{float(context['previous_high']):.6f}",
        turtle_low_5=f"{float(context['previous_low']):.6f}",
        direction=direction or "NONE",
        one_minute_confirmation=one_minute_confirmation,
        open_trade=bool(state.get("open_trade")),
        trades_today=f"{int(state['trades_today'])}/{MAX_TRADES_PER_DAY}",
        equity=f"{float(state['equity']):.6f} USDT",
    )

    opened_now = False

    if (
        not state.get("open_trade")
        and direction in {"LONG", "SHORT"}
        and direction != state.get("entry_lock_direction","NONE")
        and one_minute_confirmation
        and int(state["trades_today"]) < MAX_TRADES_PER_DAY
    ):
        open_trade(
            state,
            direction,
            futures_price,
            candle,
            context,
        )
        opened_now = True

    if opened_now:
        state["last_signal"] = direction
        state["last_reason"] = "Все условия выполнены — paper-сделка открыта"
    else:
        state["last_signal"] = "NO TRADE"
        state["last_reason"] = reason

    print(
        f"Автопроверка ETHUSDT выполнена: {state['last_signal']} | "
        f"Причина: {state['last_reason']}",
        flush=True,
    )
    save_state(state)


def auto_check() -> None:
    state = load_state()

    while True:
        cycle_start = time.time()

        try:
            analyze_market(state)
        except Exception as exc:
            log_block(
                "AUTO CHECK ERROR",
                error=f"{type(exc).__name__}: {exc}",
                retry=f"After {SLEEP_SECONDS} seconds",
            )

        elapsed = time.time() - cycle_start
        time.sleep(max(1.0, SLEEP_SECONDS - elapsed))


# ============================================================================
# TELEGRAM COMMANDS
# ============================================================================

@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "ETH Turtle 5 paper bot работает.\n"
        "Рынок: Binance Futures\n"
        "Проверка: каждые 60 секунд\n"
        "Команды:\n"
        "/price — цена ETH Futures\n"
        "/strong_signal — состояние рынка и причина входа/отказа\n"
        "/status — состояние бота и открытой сделки\n"
        "/history — последние события по сделкам",
    )


@bot.message_handler(commands=["price"])
def price(message):
    bot.reply_to(message, f"ETHUSDT Futures: {current_futures_price():.2f} USDT")


@bot.message_handler(commands=["strong_signal"])
def strong_signal(message):
    try:
        state = load_state()
        reset_daily_counter(state)
        snap = market_snapshot()
        candle = snap["candle"]
        context = snap["context"]
        direction = snap["direction"]
        confirmation = snap["confirmation"]
        futures_price = snap["futures_price"]
        reason = no_trade_reason(state, direction, confirmation)

        can_open = (
            not state.get("open_trade")
            and direction in {"LONG", "SHORT"}
            and confirmation
            and int(state.get("trades_today", 0)) < MAX_TRADES_PER_DAY
        )
        signal = direction if can_open else "NO TRADE"

        lines = [
            "ETHUSDT PREVIOUS 15M BREAKOUT",
            "",
            f"Сигнал: {signal}",
            f"Направление 15M: {direction or 'NONE'}",
            f"Подтверждение 1M: {'ЕСТЬ' if confirmation else 'НЕТ'}",
            f"Цена Futures: {futures_price:.2f} USDT",
            f"Turtle High 5: {float(context['previous_high']):.2f}",
            f"Turtle Low 5: {float(context['previous_low']):.2f}",
            f"Текущая 15M цена: {float(context['current_close']):.2f}",
            f"Закрытая 1M свеча: {float(candle['open']):.2f} → {float(candle['close']):.2f}",
            "",
            f"Открытая сделка: {'ДА' if state.get('open_trade') else 'НЕТ'}",
            f"Сделок сегодня: {int(state.get('trades_today', 0))}/{MAX_TRADES_PER_DAY}",
            f"Капитал: {float(state.get('equity', START_EQUITY_USDT)):.2f} USDT",
            "",
            f"Причина: {reason}",
        ]

        if can_open:
            entry = execution_price(futures_price, direction, True)
            levels = make_levels(direction, entry)
            qty, margin, notional = position_values(
                entry,
                float(state.get("equity", START_EQUITY_USDT)),
            )
            lines.extend([
                "",
                "УСЛОВИЯ ПОТЕНЦИАЛЬНОЙ СДЕЛКИ",
                f"Вход: {entry:.2f}",
                f"Стоп: {levels['stop']:.2f}",
                f"TP1: {levels['tp1']:.2f} — закрыть 50%",
                f"TP2: {levels['tp2']:.2f} — закрыть 30%",
                f"TP3: {levels['tp3']:.2f} — закрыть 20%",
                f"Маржа: {margin:.2f} USDT",
                f"Номинал: {notional:.2f} USDT",
                f"Количество: {qty:.6f} ETH",
                f"Плечо: {LEVERAGE:.0f}x",
            ])

        bot.reply_to(message, "\n".join(lines))
    except Exception as exc:
        bot.reply_to(message, f"Ошибка ручной проверки: {type(exc).__name__}: {exc}")


@bot.message_handler(commands=["status"])
def status(message):
    state = load_state()
    trade = state.get("open_trade")

    lines = [
        "СОСТОЯНИЕ БОТА",
        "",
        "Бот работает: ДА",
        "Рынок: ETHUSDT Binance Futures",
        "Проверка: каждые 60 секунд",
        f"Последняя проверка: {state.get('last_check_time') or 'ещё не было'}",
        f"Проверок сегодня: {int(state.get('checks_today', 0))}",
        f"Последний сигнал: {state.get('last_signal', 'NO TRADE')}",
        f"Последняя причина: {state.get('last_reason', '-')}",
        f"Последняя цена Futures: {state.get('last_futures_price') or '-'}",
        f"Капитал: {float(state.get('equity', START_EQUITY_USDT)):.2f} USDT",
        f"Сделок сегодня: {int(state.get('trades_today', 0))}/{MAX_TRADES_PER_DAY}",
        "",
        f"Открытая сделка: {'ДА' if trade else 'НЕТ'}",
    ]

    if trade:
        stage_names = {
            0: "до TP1",
            1: "TP1 выполнен, осталось 50%",
            2: "TP2 выполнен, осталось 20%",
        }
        lines.extend([
            f"Направление: {trade['side']}",
            f"Стадия: {stage_names.get(int(trade['stage']), trade['stage'])}",
            f"Вход: {float(trade['entry_exec']):.2f}",
            f"Текущий стоп: {float(trade['stop']):.2f}",
            f"TP1: {float(trade['tp1']):.2f}",
            f"TP2: {float(trade['tp2']):.2f}",
            f"TP3: {float(trade['tp3']):.2f}",
            f"Остаток позиции: {float(trade['qty_remaining']):.6f} ETH",
            f"Зафиксированный PnL: {float(trade['realized_pnl']):.4f} USDT",
            f"Комиссии: {float(trade['fees_paid']):.4f} USDT",
            f"Funding: {float(trade['funding_pnl']):.4f} USDT",
        ])

    bot.reply_to(message, "\n".join(lines))


@bot.message_handler(commands=["history"])
def history(message):
    if not TRADES_FILE.exists():
        bot.reply_to(message, "Истории сделок пока нет.")
        return

    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        bot.reply_to(message, "Истории сделок пока нет.")
        return

    lines = ["ПОСЛЕДНИЕ 10 СОБЫТИЙ ПО СДЕЛКАМ"]
    for _, row in df.tail(10).iterrows():
        lines.append(
            f"{row.get('event_time', '-')} | "
            f"{row.get('event', '-')} | "
            f"{row.get('side', '-')} | "
            f"PnL события: {row.get('event_pnl', '-')}"
        )

    bot.reply_to(message, "\n".join(lines))


# ============================================================================
# STARTUP
# ============================================================================

def startup_self_check() -> None:
    checks = {
        "symbol_is_eth_futures": SYMBOL == "ETH/USDT:USDT",
        "check_every_60_seconds": SLEEP_SECONDS == 60,
        "turtle_length_5": TURTLE_LENGTH == 5,
        "margin_5_percent": POSITION_MARGIN_PCT == 5.0,
        "leverage_10x": LEVERAGE == 10.0,
        "stop_0_60_percent": INITIAL_STOP_PCT == 0.60,
        "tp1_0_50_percent": TP1_PCT == 0.50,
        "tp2_1_00_percent": TP2_PCT == 1.00,
        "tp3_1_50_percent": TP3_PCT == 1.50,
        "tp_split_50_30_20": (
            TP1_CLOSE_FRACTION,
            TP2_CLOSE_FRACTION,
            TP3_CLOSE_FRACTION,
        ) == (0.50, 0.30, 0.20),
        "paper_mode": LIVE_TRADING_ENABLED is False,
    }

    failed = [name for name, passed in checks.items() if not passed]

    log_block(
        "STARTUP SELF-CHECK",
        **{name: "PASS" if passed else "FAIL" for name, passed in checks.items()},
    )

    if failed:
        raise RuntimeError(
            "Startup self-check failed: " + ", ".join(failed)
        )


if __name__ == "__main__":
    if not TOKEN or CHAT_ID is None:
        raise RuntimeError(
            "BOT_TOKEN or CHAT_ID is missing in Railway Variables"
        )

    startup_self_check()

    log_block(
        "BOT START",
        symbol=SYMBOL,
        market="Binance Futures",
        check_interval="Every 60 seconds",
        strategy="Previous closed 15m breakout + closed 1m confirmation",
        margin=f"{POSITION_MARGIN_PCT}% of current equity",
        leverage=f"{LEVERAGE}x",
        stop=f"{INITIAL_STOP_PCT}%",
        tp1=f"{TP1_PCT}% | close 50%",
        tp2=f"{TP2_PCT}% | close 30%",
        tp3=f"{TP3_PCT}% | close 20%",
        live_trading=LIVE_TRADING_ENABLED,
        mode="PAPER ONLY",
    )

    threading.Thread(
        target=auto_check,
        daemon=True,
    ).start()

    bot.infinity_polling(
        skip_pending=True,
        timeout=30,
        long_polling_timeout=30,
    )
