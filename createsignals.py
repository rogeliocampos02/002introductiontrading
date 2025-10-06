# createsignals.py
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def add_indicators(df: pd.DataFrame,
                   ema_fast: int = 12,
                   ema_slow: int = 26,
                   rsi_window: int = 14,
                   donch_window: int = 55,
                   ema_longterm: int = 720) -> pd.DataFrame:
    """
    Calcula EMA rápida/lenta, EMA de largo plazo, RSI, ADX, ATR y bandas Donchian.
    """
    out = df.copy()

    # EMAs
    out["ema_fast"] = EMAIndicator(close=out["Close"], window=ema_fast).ema_indicator()
    out["ema_slow"] = EMAIndicator(close=out["Close"], window=ema_slow).ema_indicator()
    out["ema_lt"]   = EMAIndicator(close=out["Close"], window=ema_longterm).ema_indicator()

    # RSI
    out["rsi"] = RSIIndicator(close=out["Close"], window=rsi_window).rsi()

    # ADX y ATR (volatilidad)
    adx = ADXIndicator(high=out["High"], low=out["Low"], close=out["Close"], window=14)
    out["adx"] = adx.adx()
    atr = AverageTrueRange(high=out["High"], low=out["Low"], close=out["Close"], window=14)
    out["atr"] = atr.average_true_range()

    # Donchian (breakout)
    out["donch_up"] = out["High"].rolling(donch_window).max()
    out["donch_dn"] = out["Low"].rolling(donch_window).min()

    return out


def add_signals(df: pd.DataFrame,
                rsi_upper: int = 58,
                rsi_lower: int = 45,
                adx_thr: float = 15.0,
                atr_q: float | None = 0.55) -> pd.DataFrame:
    """
    Señal 2-de-3 con filtros:
      - Tendencia: ema_fast > ema_slow (long) / < (short)
      - Momentum: RSI > rsi_upper / < rsi_lower
      - Breakout: cruce de Donchian usando el canal de la barra PREVIA
    Filtros:
      - ADX >= adx_thr
      - ATR por encima del cuantil 'atr_q' (si es None, no se filtra por ATR)
      - Cortos sólo si Close < ema_lt (macro bajista)
    """
    out = df.copy()

    # 1) Reglas base
    s1_long  = (out["ema_fast"] > out["ema_slow"]).astype(int)
    s1_short = (out["ema_fast"] < out["ema_slow"]).astype(int)

    s2_long  = (out["rsi"] > rsi_upper).astype(int)
    s2_short = (out["rsi"] < rsi_lower).astype(int)

    # 2) Donchian: comparamos con el canal de la barra previa (más realista)
    up_prev = out["donch_up"].shift(1)
    dn_prev = out["donch_dn"].shift(1)
    s3_long  = (out["High"] >= up_prev).astype(int)
    s3_short = (out["Low"]  <= dn_prev).astype(int)

    votes_long  = s1_long + s2_long + s3_long
    votes_short = s1_short + s2_short + s3_short

    # 3) Filtros de mercado
    adx_ok = out["adx"] >= adx_thr

    if atr_q is None:
        vol_ok = pd.Series(True, index=out.index)
    else:
        thr = out["atr"].quantile(float(atr_q))
        vol_ok = out["atr"] >= thr

    macro_down = out["Close"] < out["ema_lt"]  # cortos sólo con tendencia macro bajista

    long_raw  = (votes_long  >= 2) & adx_ok & vol_ok
    short_raw = (votes_short >= 2) & adx_ok & vol_ok & macro_down

    # 4) Desfase 1 barra para evitar look-ahead (sin fillna -> sin warnings)
    out["long_signal"]  = np.r_[False, long_raw.values[:-1]]
    out["short_signal"] = np.r_[False, short_raw.values[:-1]]

    # 5) Warm-up: eliminar NaN de indicadores
    out = out.dropna(subset=["ema_fast","ema_slow","rsi","atr","adx","donch_up","donch_dn","ema_lt"]).reset_index(drop=True)
    return out

   
