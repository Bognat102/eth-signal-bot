"""
strategy_core.py

Общая логика стратегии: индикаторы, alpha-скоринг, торговые уровни, позиционный сайзинг.
Используется ОДНОВРЕМЕННО в live-боте (main_FLEX_ADX_EXT15_v2.py) и в backtest.py,
чтобы бэктест реально отражал то, что будет делать бот, а не отдельную "бумажную" логику.

Главные изменения относительно оригинала:
1. TP1/TP2 пересчитаны так, чтобы R:R был > 1 уже на первом тейке (было 0.91:1 — математически убыточно).
2. RSI/ATR/ADX считаются по Wilder smoothing (стандарт индустрии), а не по простому rolling mean.
3. VWAP привязан к UTC-дню (anchored), а не считается cumsum по всему загруженному окну.
4. Добавлен позиционный сайзинг по % риска капитала.
5. Добавлен учёт комиссий (taker fee) в расчёте PnL.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Параметры стратегии
# ---------------------------------------------------------------------------
ADX_MIN = 35
ADX_SOFT_MIN = 25
ALPHA_THRESHOLD = 65           # порог под новую mean-reversion шкалу (макс ~105 баллов)
ALPHA_STRONG_THRESHOLD = 85
MAX_EXT_ATR = 1.5
REQUIRE_FULL_ADX = True       # если True — "мягкий" вход при ADX 25-35 отключён полностью

# --- Mean-reversion параметры (v2 стратегии, после того как trend-following
#     показал win rate на уровне случайного входа — см. историю бэктестов) ---
# --- Funding-first стратегия (v3) ---
# Находка после 3 независимых периодов на ETH и ADA: mean-reversion по Bollinger/RSI
# проваливается на ~1 из 3 периодов (win rate падает до 11-14%, PF 0.2-0.4), причём
# ПРОБЛЕМА НЕ В АКТИВЕ — она системная. Единственный элемент, который стабильно помогал
# во всех предыдущих тестах — funding rate. Новая версия делает его ОСНОВНЫМ триггером
# (не одним из многих баллов в скоринге), а RSI/Bollinger — второстепенным подтверждением.
FUNDING_LONG_MAX_PCTILE = 40   # LONG поддерживаем, если funding в нижних 40% (шорты перегружены)
FUNDING_SHORT_MIN_PCTILE = 60  # SHORT поддерживаем, если funding в верхних 40% (лонги перегружены)
USE_FUNDING_HARD_FILTER = True # если True — сделка блокируется, если funding явно ПРОТИВ направления

# --- Funding-first эксперимент (v3) — ПРОТЕСТИРОВАН И ОТКЛОНЁН ---
# Ужесточение funding-порога до топ/низ 10% почти обнулило число сделок (50→12, 27→2)
# и резко ухудшило результат на всех проверенных периодах ADA. Экстремальный funding
# слишком редкое явление, чтобы быть единственным триггером. funding как БОНУС в скоринге
# (см. FUNDING_LONG_MAX_PCTILE/FUNDING_SHORT_MIN_PCTILE выше, порог 40%) работает лучше.
FUNDING_EXTREME_LONG_PCTILE = 10
FUNDING_EXTREME_SHORT_PCTILE = 90
USE_FUNDING_FIRST_MODE = False  # ВЫКЛЮЧЕНО — см. пояснение выше

ADX_RANGE_MAX = 30    # ПРОТЕСТИРОВАНО: сужение до 22 (на основе ADX на входе у убыточных
                       # сделок) снизило убыток на плохом периоде (-38%→-14%), но полностью
                       # разрушило лучший период (+36.91%→-8.7%) — тот же паттерн неудачи,
                       # что и с REGIME_BTC_ADX_MAX: фильтр режет выборку, а не улучшает качество.
                       # Возвращено на исходное значение.
RSI_OVERSOLD = 32
RSI_OVERBOUGHT = 68

USE_TRAILING_EXIT = False  # ПРОТЕСТИРОВАНО: на обоих периодах (последние 180д и 180-360д назад)
                            # trailing 1×ATR дал хуже результат, чем фиксированный TP2
                            # (win rate вырос, но средняя прибыльная сделка стала меньше — net хуже).
                            # Оставляю флаг на будущее для повторных экспериментов с другим TRAIL_ATR_MULT.
TRAIL_ATR_MULT = 1.0      # дистанция трейлинга в ATR (посчитанном на момент входа)

USE_5M_CONFIRMATION = False  # ПРОТЕСТИРОВАНО: ухудшило результат на обоих периодах
                              # (win rate упал с ~30% до ~23-25%, PF упал ниже 1 на обоих окнах).
                              # Ожидание подтверждающей 5m свечи запаздывает относительно разворота,
                              # а не улучшает точность входа. Оставляю флаг для будущих экспериментов.
MAX_5M_WAIT_CANDLES = 3      # сколько 5m свечей (=15 минут) ждём подтверждение, прежде чем пропустить сигнал

USE_DIVERGENCE_FILTER = False  # ПРОТЕСТИРОВАНО: слишком жёсткий фильтр — подтверждал направление
                                # лишь в ~5% случаев (12-15 раз из ~260), обрушил число сделок до 3
                                # и почти до 0 на разных периодах. Выборка стала статистически
                                # незначимой. Оставляю флаг для будущих экспериментов (например,
                                # с более коротким DIVERGENCE_LOOKBACK или мягким бонусом вместо
                                # жёсткого блокирующего требования).
DIVERGENCE_LOOKBACK = 20       # окно поиска предыдущего экстремума (баров по 15m, ~5 часов)

# --- Фильтр рыночного режима ---
# Находка: mean-reversion работает в боковике и резко проседает в трендовом рынке
# (портфельный тест на 10 монетах: +38% в спокойный период, -65% в трендовый).
# Используем ADX по BTC как индикатор общего состояния рынка — если BTC сам
# в сильном тренде, весь рынок (включая альты) обычно тоже движется направленно,
# и фейд-сделки (mean-reversion) статистически чаще проигрывают.
USE_REGIME_FILTER = False  # ПРОТЕСТИРОВАНО: грубый ADX-порог резал и плохие, и хорошие периоды
                            # одинаково — просадка на плохом периоде снизилась (-68%→-59%), но
                            # прибыль на хорошем периоде почти исчезла (+38%→+4%). Не даёт чистого
                            # улучшения. Оставляю флаг для будущих экспериментов с более тонким
                            # определением режима (не жёсткий порог ADX, а что-то мягче/комбинированное).
REGIME_BTC_ADX_MAX = 25   # торгуем только если ADX BTC ниже этого порога (боковик/слабый тренд)


def market_regime_ok(btc_row):
    """True — рынок в подходящем для mean-reversion режиме (боковик).
    False — BTC в сильном тренде, лучше пропустить сигналы по всему портфелю.
    None (через btc_row=None) обрабатывается как 'разрешено' — не блокируем при нехватке данных."""
    if btc_row is None:
        return True
    adx = safe_float(btc_row.get("adx"), default=None) if hasattr(btc_row, "get") else safe_float(btc_row["adx"])
    if adx is None or pd.isna(adx):
        return True
    return bool(adx <= REGIME_BTC_ADX_MAX)


def detect_divergence(df_window, candidate, lookback=DIVERGENCE_LOOKBACK):
    """df_window: DataFrame последних (lookback+1) баров, где последняя строка — ТЕКУЩИЙ бар.
    Bullish-дивергенция (для LONG): цена обновляет минимум ниже недавнего, а RSI на этом
    минимуме ВЫШЕ, чем на предыдущем минимуме — движение вниз слабеет, несмотря на новый лоу.
    Bearish-дивергенция (для SHORT) — зеркально по максимумам."""
    if len(df_window) < lookback + 1:
        return False
    window = df_window.iloc[-(lookback + 1):-1]  # предыдущие бары, БЕЗ текущего
    current = df_window.iloc[-1]

    if candidate == "LONG":
        prior_idx = window["low"].idxmin()
        prior_low = window.loc[prior_idx, "low"]
        prior_rsi = window.loc[prior_idx, "rsi"]
        cur_low = safe_float(current["low"])
        cur_rsi = safe_float(current["rsi"])
        if pd.isna(prior_rsi) or pd.isna(prior_low):
            return False
        return bool(cur_low <= prior_low and cur_rsi > prior_rsi)

    if candidate == "SHORT":
        prior_idx = window["high"].idxmax()
        prior_high = window.loc[prior_idx, "high"]
        prior_rsi = window.loc[prior_idx, "rsi"]
        cur_high = safe_float(current["high"])
        cur_rsi = safe_float(current["rsi"])
        if pd.isna(prior_rsi) or pd.isna(prior_high):
            return False
        return bool(cur_high >= prior_high and cur_rsi < prior_rsi)

    return False


def confirm_entry_5m(candidate, five_min_candles):
    """five_min_candles: DataFrame/список 5m свечей (open, close), идущих ПОСЛЕ 15m сигнала,
    в пределах следующих MAX_5M_WAIT_CANDLES баров.
    Для LONG ждём первую бычью (зелёную) 5m свечу — признак остановки падения.
    Для SHORT — первую медвежью (красную) — признак остановки роста.
    Возвращает (confirmed: bool, entry_price: float|None)."""
    for _, c in five_min_candles.iterrows():
        is_bullish = c["close"] > c["open"]
        is_bearish = c["close"] < c["open"]
        if candidate == "LONG" and is_bullish:
            return True, float(c["close"])
        if candidate == "SHORT" and is_bearish:
            return True, float(c["close"])
    return False, None


def compute_funding_feature(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """Присоединяет funding rate к каждому 15m бару (forward-fill, funding обновляется раз в 8ч)
    и считает скользящий перцентиль относительно последних ~30 дней — чтобы понимать, экстремальный
    ли funding СЕЙЧАС относительно недавней истории, а не в абсолюте."""
    df = df.sort_values("time").copy()
    funding_df = funding_df[["time", "fundingRate"]].sort_values("time")
    merged = pd.merge_asof(df, funding_df, on="time", direction="backward")
    merged["funding_rate"] = merged["fundingRate"].ffill().fillna(0.0)
    window = 2880  # ~30 дней на 15m барах
    merged["funding_pctile"] = merged["funding_rate"].rolling(window, min_periods=200).rank(pct=True) * 100
    return merged.drop(columns=["fundingRate"])


def funding_confirms(candidate, funding_pctile):
    """True/False/None. None означает 'данных недостаточно' — не блокируем сделку, просто без бонуса.
    Пороги 40/60 (не 10/90) — проверенная версия, дающая лучший результат, чем extreme-пороги."""
    if funding_pctile is None or pd.isna(funding_pctile):
        return None
    if candidate == "LONG":
        return bool(funding_pctile <= FUNDING_LONG_MAX_PCTILE)
    if candidate == "SHORT":
        return bool(funding_pctile >= FUNDING_SHORT_MIN_PCTILE)
    return None

# Множители ATR для стопа/тейков.
# Было: stop=1.1, tp1=1.0 (R:R 0.91:1 — хуже монетки), tp2=1.8
STOP_ATR_MULT = 1.0
TP1_ATR_MULT = 1.4   # R:R на TP1 теперь 1.4:1
TP2_ATR_MULT = 2.4   # R:R на TP2 теперь 2.4:1

# ---------------------------------------------------------------------------
# Риск-менеджмент / комиссии
# ---------------------------------------------------------------------------
ACCOUNT_EQUITY_USDT = 1000.0     # виртуальный капитал для paper-статистики (поменяй под себя)
RISK_PER_TRADE_PCT = 5.0         # % капитала, которым рискуем на одну сделку
                                  # Поднято с 3% для теста. ВАЖНО: на "плохом" третьем периоде
                                  # (offset=360) убыток тоже вырастет пропорционально — при 3%
                                  # там было -38.11%, при 5% ожидай около -63% с ещё большей
                                  # просадкой. Риск масштабирует прибыль и убыток одинаково.
TAKER_FEE_PCT = 0.045            # Binance Futures taker fee за одну сторону сделки (~0.045%)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # --- RSI (Wilder smoothing) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- True Range / ATR (Wilder smoothing) ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    df["avg_volume"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["avg_volume"].replace(0, pd.NA)

    # --- ADX (Wilder smoothing) ---
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_dm_s = plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm_s / df["atr"].replace(0, pd.NA)
    minus_di = 100 * minus_dm_s / df["atr"].replace(0, pd.NA)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, pd.NA)) * 100
    df["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # --- VWAP, привязанный к UTC-дню (сбрасывается каждый день) ---
    dt = pd.to_datetime(df["time"], unit="ms", utc=True)
    day = dt.dt.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]
    df["vwap"] = tp_vol.groupby(day).cumsum() / df["volume"].replace(0, pd.NA).groupby(day).cumsum()

    # --- Bollinger Bands (20, 2) — нужны для mean-reversion логики ---
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    return df


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def detect_candidate(last_15m, last_1h, funding_pctile=None):
    """FUNDING-FIRST: экстремальный funding rate — ОБЯЗАТЕЛЬНОЕ условие входа (если
    USE_FUNDING_FIRST_MODE=True), а не просто бонус в скоринге. RSI/Bollinger остаются
    как подтверждение перегретости."""
    close = safe_float(last_15m["close"])
    rsi = safe_float(last_15m["rsi"])
    bb_lower = safe_float(last_15m["bb_lower"])
    bb_upper = safe_float(last_15m["bb_upper"])
    adx = safe_float(last_15m["adx"])
    ema200_1h = safe_float(last_1h["ema200"])

    if bb_lower <= 0 or bb_upper <= 0:
        return "NONE"

    if adx > ADX_RANGE_MAX:
        return "NONE"  # рынок слишком сильно трендит — фейд здесь опасен

    if USE_FUNDING_FIRST_MODE:
        if funding_pctile is None or pd.isna(funding_pctile):
            return "NONE"  # без funding-данных сигнала нет — теперь это ключевое условие

        funding_long_ok = funding_pctile <= FUNDING_EXTREME_LONG_PCTILE
        funding_short_ok = funding_pctile >= FUNDING_EXTREME_SHORT_PCTILE

        if funding_long_ok and close <= bb_lower and rsi <= RSI_OVERSOLD and close > ema200_1h * 0.97:
            return "LONG"
        if funding_short_ok and close >= bb_upper and rsi >= RSI_OVERBOUGHT and close < ema200_1h * 1.03:
            return "SHORT"
        return "NONE"

    # --- старый режим (funding как бонус в calculate_alpha, не требование) ---
    if close <= bb_lower and rsi <= RSI_OVERSOLD and close > ema200_1h * 0.97:
        return "LONG"
    if close >= bb_upper and rsi >= RSI_OVERBOUGHT and close < ema200_1h * 1.03:
        return "SHORT"
    return "NONE"


def calculate_alpha(candidate, df_15m, last_15m, last_1h, btc_ok=None, funding_ok=None):
    """MEAN-REVERSION скоринг: чем сильнее перепроданность/перекупленность и чем
    спокойнее рынок (низкий ADX = боковик), тем выше уверенность в развороте к средней."""
    if candidate == "NONE":
        return 0, ["Нет базового направления"]

    reasons = []
    score = 0

    close = safe_float(last_15m["close"])
    rsi = safe_float(last_15m["rsi"])
    adx = safe_float(last_15m["adx"])
    atr = safe_float(last_15m["atr"])
    volume_ratio = safe_float(last_15m["volume_ratio"], 1.0)
    bb_lower = safe_float(last_15m["bb_lower"])
    bb_upper = safe_float(last_15m["bb_upper"])
    bb_mid = safe_float(last_15m["bb_mid"])
    vwap = safe_float(last_15m["vwap"])

    # ADX: 25 points — чем ниже, тем чище боковик, тем надёжнее фейд
    if adx < 15:
        score += 25; reasons.append(f"ADX очень низкий: {round(adx,2)} — чистый боковик")
    elif adx < 20:
        score += 18; reasons.append(f"ADX низкий: {round(adx,2)} — боковик")
    elif adx < ADX_RANGE_MAX:
        score += 8; reasons.append(f"ADX умеренный: {round(adx,2)}")
    else:
        reasons.append(f"ADX слишком высокий для фейда: {round(adx,2)}")

    # RSI extremity: 25 points — чем дальше от 50, тем сильнее перекос
    if candidate == "LONG":
        depth = RSI_OVERSOLD - rsi
        if rsi <= 20:
            score += 25; reasons.append(f"RSI глубоко перепродан: {round(rsi,2)}")
        elif rsi <= 26:
            score += 18; reasons.append(f"RSI сильно перепродан: {round(rsi,2)}")
        elif rsi <= RSI_OVERSOLD:
            score += 10; reasons.append(f"RSI перепродан: {round(rsi,2)}")
    else:
        if rsi >= 80:
            score += 25; reasons.append(f"RSI глубоко перекуплен: {round(rsi,2)}")
        elif rsi >= 74:
            score += 18; reasons.append(f"RSI сильно перекуплен: {round(rsi,2)}")
        elif rsi >= RSI_OVERBOUGHT:
            score += 10; reasons.append(f"RSI перекуплен: {round(rsi,2)}")

    # Band penetration: 20 points — насколько цена вышла за полосу
    band_width = bb_upper - bb_lower if (bb_upper > 0 and bb_lower > 0) else 0
    if band_width > 0:
        if candidate == "LONG":
            penetration = (bb_lower - close) / band_width
        else:
            penetration = (close - bb_upper) / band_width
        if penetration >= 0.05:
            score += 20; reasons.append(f"Сильный выход за полосу Боллинджера: {round(penetration*100,1)}%")
        elif penetration >= 0:
            score += 12; reasons.append("Цена у границы полосы Боллинджера")
        else:
            reasons.append("Цена ещё не вышла за полосу")

    # Volume spike: 15 points — объёмный всплеск часто сопровождает разворот/капитуляцию
    if volume_ratio >= 1.5:
        score += 15; reasons.append(f"Сильный объёмный всплеск: x{round(volume_ratio,2)}")
    elif volume_ratio >= 1.1:
        score += 8; reasons.append(f"Повышенный объём: x{round(volume_ratio,2)}")

    # Расстояние от VWAP: 15 points — доп. подтверждение перегретости
    if vwap > 0:
        if candidate == "LONG" and close < vwap:
            dist_pct = (vwap - close) / vwap * 100
            if dist_pct >= 1.0:
                score += 15; reasons.append(f"Цена far ниже VWAP: -{round(dist_pct,2)}%")
            else:
                score += 8; reasons.append("Цена ниже VWAP")
        elif candidate == "SHORT" and close > vwap:
            dist_pct = (close - vwap) / vwap * 100
            if dist_pct >= 1.0:
                score += 15; reasons.append(f"Цена far выше VWAP: +{round(dist_pct,2)}%")
            else:
                score += 8; reasons.append("Цена выше VWAP")

    if btc_ok is True:
        score += 5; reasons.append("BTC не мешает развороту")
    elif btc_ok is False:
        score -= 10; reasons.append("BTC сильно против направления")
    else:
        reasons.append("BTC подтверждение недоступно")

    # Funding rate: 15 points — экстремальный funding против позиции толпы = доп. довод за разворот
    if funding_ok is True:
        score += 15; reasons.append("Funding rate подтверждает перекос позиционирования толпы")
    elif funding_ok is False:
        score -= 15; reasons.append("Funding rate НЕ подтверждает — рынок не перегружен в нужную сторону")
    else:
        reasons.append("Funding rate недоступен")

    score = max(0, min(100, int(round(score))))
    return score, reasons


STOP_DISTANCE_MULT = 0.6   # доля расстояния до средней (bb_mid), используемая как стоп
                            # ПРОТЕСТИРОВАНО (ADA+ARB, риск 5%): чёткий купол с пиком на 0.6.
                            #   0.4 → +291.21%, просадка -31.67%
                            #   0.6 → +341.83%, просадка -31.67%  (ЛУЧШИЙ)
                            #   0.7 → +239.97%, просадка -31.67%
                            #   0.8 → +177.44%, просадка -36.26% (тут и просадка выросла)
                            # 0.6 — устойчивый локальный оптимум, не случайность.
TP2_DISTANCE_MULT = 1.6    # доля расстояния до средней для TP2 (перелёт за среднюю)


def make_trade_levels(signal, price, atr, bb_mid, stop_mult=None, tp2_mult=None):
    """MEAN-REVERSION уровни: цель — возврат к средней (bb_mid), а не фиксированный ATR-множитель.
    Стоп — часть дистанции до средней (если цена идёт дальше от средней вместо разворота,
    тезис неверен — выходим быстро). TP2 — небольшой overshoot за среднюю."""
    stop_mult = STOP_DISTANCE_MULT if stop_mult is None else stop_mult
    tp2_mult = TP2_DISTANCE_MULT if tp2_mult is None else tp2_mult
    distance = abs(price - bb_mid)
    distance = max(distance, atr * 0.5) if atr > 0 else distance  # защита от слишком узкой цели

    if signal == "LONG":
        stop = round(price - distance * stop_mult, 2)
        tp1 = round(bb_mid, 2)
        tp2 = round(price + distance * tp2_mult, 2)
    else:
        stop = round(price + distance * stop_mult, 2)
        tp1 = round(bb_mid, 2)
        tp2 = round(price - distance * tp2_mult, 2)
    return stop, tp1, tp2


def position_size(entry, stop, equity=ACCOUNT_EQUITY_USDT, risk_pct=RISK_PER_TRADE_PCT):
    """Размер позиции (в базовом активе, напр. ETH) исходя из % риска капитала."""
    risk_amount = equity * (risk_pct / 100.0)
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 0.0, 0.0
    qty = risk_amount / stop_distance
    return round(qty, 4), round(risk_amount, 2)


def apply_fees(pnl_price_diff, entry, exit_price, qty):
    """PnL в USDT с учётом комиссий обеих сторон сделки.
    ВАЖНО: funding rate НЕ учтён (он может быть и +, и -, зависит от времени удержания
    и режима рынка) — для честной оценки добавь его отдельно, если держишь позиции
    через funding-события (00:00/08:00/16:00 UTC)."""
    if qty <= 0:
        return round(pnl_price_diff, 2)
    gross_usdt = pnl_price_diff * qty
    fee_usdt = (entry * qty + exit_price * qty) * (TAKER_FEE_PCT / 100.0)
    return round(gross_usdt - fee_usdt, 2)
