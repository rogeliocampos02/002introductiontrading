# createsignals.py
# ------------------------------------------------------------
# Cálculo de indicadores (EMA, RSI, ATR, Donchian) y señales
# Reglas: 2-de-3 
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(int(n), 1), adjust=False).mean()


def _rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    n = max(int(n), 1)
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    up = pd.Series(up, index=close.index)
    dn = pd.Series(dn, index=close.index)
    roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    n = max(int(n), 1)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    return atr


def add_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_window: int = 14,
    atr_window: int = 14,
    donch_window: int = 20,
) -> pd.DataFrame:
    """Devuelve una copia con columnas de indicadores."""
    out = df.copy()
    out["ema_fast"] = _ema(out["Close"], ema_fast)
    out["ema_slow"] = _ema(out["Close"], ema_slow)
    out["rsi"] = _rsi_wilder(out["Close"], rsi_window)
    out["atr"] = _atr_wilder(out["High"], out["Low"], out["Close"], atr_window)

    donch_window = max(int(donch_window), 2)
    out["donch_up"] = out["High"].rolling(donch_window).max()
    out["donch_dn"] = out["Low"].rolling(donch_window).min()

    return out


def add_signals(
    df: pd.DataFrame,
    rsi_upper: float = 60,
    rsi_lower: float = 40,
    atr_q: float = 0.5,
) -> pd.DataFrame:
    """
    Señales 2-de-3 (confirmación):
      1) Tendencia: ema_fast >= ema_slow (alcista) / <= (bajista)
      2) Momento:   rsi > rsi_upper (long) / rsi < rsi_lower (short)
      3) Ruptura:   Close > donch_up + atr_q*atr (long)
                    Close < donch_dn - atr_q*atr (short)
    Se desplaza 1 barra para evitar look-ahead.
    """
    out = df.copy()

    s1_long = (out["ema_fast"] >= out["ema_slow"]).astype(int)
    s1_short = (out["ema_fast"] <= out["ema_slow"]).astype(int)

    s2_long = (out["rsi"] > float(rsi_upper)).astype(int)
    s2_short = (out["rsi"] < float(rsi_lower)).astype(int)

    buf = out["atr"] * float(atr_q)
    s3_long = (out["Close"] > (out["donch_up"] + buf)).astype(int)
    s3_short = (out["Close"] < (out["donch_dn"] - buf)).astype(int)

    votes_long = s1_long + s2_long + s3_long
    votes_short = s1_short + s2_short + s3_short

    # --- FIX de FutureWarning ---
    # Evitamos fillna/astype después de shift usando fill_value en shift(),
    # lo que conserva dtype boolean y elimina el NaN inicial sin downcasting.
    long_raw = (votes_long >= 2)
    short_raw = (votes_short >= 2)

    out["long_signal"] = long_raw.shift(1, fill_value=False)
    out["short_signal"] = short_raw.shift(1, fill_value=False)
    # ----------------------------

    # Eliminar warm-up donde hay NaNs en indicadores
    out = out.dropna(
        subset=["ema_fast", "ema_slow", "rsi", "atr", "donch_up", "donch_dn"]
    ).reset_index(drop=True)

    return out
