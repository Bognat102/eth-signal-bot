# ETH Turtle 20 Railway Paper Bot

Files:
- `main.py`
- `strategy_core.py`
- `requirements.txt`

## Verified behavior

1. Uses Binance Futures:
   - `defaultType = future`
   - symbol `ETH/USDT:USDT`

2. Checks the market every 5 minutes:
   - aligned to the next 5-minute candle boundary

3. Prints detailed Railway logs:
   - every 5-minute market check
   - trade open
   - TP1 / TP2 / TP3
   - break-even move
   - stop
   - final PnL, fees, funding, equity

## Strategy

- Turtle 20 on 15m defines direction
- 5m bullish/bearish candle gives entry
- stop 0.40%
- TP1 0.30%, closes 50%
- TP2 0.60%, closes 30%
- TP3 1.00%, closes 20%
- after TP1 the remaining 50% moves to its own cost-covered break-even
- 5% margin from current equity
- 10x leverage
- paper mode only

## Railway Variables

- `BOT_TOKEN`
- `CHAT_ID`

## Important

Railway local CSV/JSON files may be lost on redeploy unless a persistent Volume is attached.
