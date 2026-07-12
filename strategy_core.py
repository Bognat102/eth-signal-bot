"""
strategy_core.py

ETHUSDT Futures paper-trading strategy based on the chosen Turtle 20 V16 logic.

Important:
- ETH only.
- Binance Futures data only.
- 15m Turtle 20 defines direction.
- 5m candle gives entry in that direction.
- No EMA/ADX/RSI/Bollinger/VWAP/score/funding filter.
- 5% of current equity is used as margin.
- 10x leverage.
- Stop 0.40%.
- TP1 0.30% closes 50%.
- TP2 0.60% closes 30%.
- TP3 1.00% closes 20%.
- After TP1, the remaining position moves to its own cost-covered break-even.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

START_EQUITY_USDT = 1000.0
MARGIN_PCT = 5.0
LEVERAGE = 10.0

TAKER_FEE_PCT = 0.045
SLIPPAGE_PCT = 0.02

TURTLE_LENGTH = 20

INITIAL_STOP_PCT = 0.40
TP1_PCT = 0.30
TP2_PCT = 0.60
TP3_PCT = 1.00

TP1_CLOSE_FRACTION = 0.50
TP2_CLOSE_FRACTION = 0.30
TP3_CLOSE_FRACTION = 0.20

MAX_TRADES_PER_DAY = 20


@dataclass(frozen=True)
class Levels:
    stop: float
    tp1: float
    tp2: float
    tp3: float


def execution_price(raw_price: float, side: str, is_entry: bool) -> float:
    slip = SLIPPAGE_PCT / 100.0
    is_buy = (side == "LONG" and is_entry) or (side == "SHORT" and not is_entry)
    return raw_price * (1 + slip if is_buy else 1 - slip)


def fee(notional: float) -> float:
    return abs(notional) * TAKER_FEE_PCT / 100.0


def gross_pnl(side: str, entry: float, exit_price: float, qty: float) -> float:
    if side == "LONG":
        return (exit_price - entry) * qty
    return (entry - exit_price) * qty


def position_values(entry: float, equity: float):
    margin = equity * MARGIN_PCT / 100.0
    notional = margin * LEVERAGE
    qty = notional / entry if entry > 0 else 0.0
    return qty, margin, notional


def make_levels(side: str, entry: float) -> Levels:
    if side == "LONG":
        return Levels(
            stop=entry * (1 - INITIAL_STOP_PCT / 100.0),
            tp1=entry * (1 + TP1_PCT / 100.0),
            tp2=entry * (1 + TP2_PCT / 100.0),
            tp3=entry * (1 + TP3_PCT / 100.0),
        )

    return Levels(
        stop=entry * (1 + INITIAL_STOP_PCT / 100.0),
        tp1=entry * (1 - TP1_PCT / 100.0),
        tp2=entry * (1 - TP2_PCT / 100.0),
        tp3=entry * (1 - TP3_PCT / 100.0),
    )


def remaining_position_break_even(side: str, entry_exec: float) -> float:
    """Cost-covered break-even for the remaining position only.

    TP1 profit is preserved. The remaining part covers its own future
    exit fee and modeled exit slippage.
    """
    fee_rate = TAKER_FEE_PCT / 100.0
    slip_rate = SLIPPAGE_PCT / 100.0

    if side == "LONG":
        target_exec = entry_exec / (1 - fee_rate)
        return target_exec / (1 - slip_rate)

    target_exec = entry_exec / (1 + fee_rate)
    return target_exec / (1 + slip_rate)


def turtle_direction(
    current_15m_close: float,
    previous_20_high: float,
    previous_20_low: float,
) -> Optional[str]:
    if current_15m_close > previous_20_high:
        return "LONG"
    if current_15m_close < previous_20_low:
        return "SHORT"
    return None


def five_minute_entry_signal(side: Optional[str], open_price: float, close_price: float) -> bool:
    if side == "LONG":
        return close_price > open_price
    if side == "SHORT":
        return close_price < open_price
    return False
