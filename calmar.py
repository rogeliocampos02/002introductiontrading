# calmar.py
# ------------------------------------------------------------
# Métricas: Sharpe, Sortino, Max Drawdown, Calmar, WinRate
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import pandas as pd


HOURS_PER_YEAR = 24 * 365.25


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def max_drawdown(eq: pd.Series) -> float:
    eq = pd.Series(eq).astype(float)
    peak = eq.cummax()
    dd = (eq / peak - 1.0).fillna(0.0)
    return dd.min()  # negativo


def sharpe(rets: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    rets = pd.Series(rets).astype(float).dropna()
    if len(rets) < 2:
        return 0.0
    mu = rets.mean()
    sd = rets.std(ddof=0)
    if sd <= 0:
        return 0.0
    return _safe_float((mu / sd) * np.sqrt(periods_per_year))


def sortino(rets: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    rets = pd.Series(rets).astype(float).dropna()
    down = rets[rets < 0]
    if len(down) == 0:
        return 0.0
    mu = rets.mean()
    sd_down = down.std(ddof=0)
    if sd_down <= 0:
        return 0.0
    return _safe_float((mu / sd_down) * np.sqrt(periods_per_year))


def calmar(eq: pd.Series, dt: pd.Series) -> float:
    eq = pd.Series(eq).astype(float)
    if len(eq) < 2:
        return 0.0
    eq0 = eq.iloc[0]
    eqf = eq.iloc[-1]
    dt0 = pd.to_datetime(dt.iloc[0])
    dtf = pd.to_datetime(dt.iloc[-1])
    years = max((dtf - dt0).total_seconds() / (365.25 * 24 * 3600.0), 1e-9)
    cagr = _safe_float((eqf / eq0) ** (1.0 / years) - 1.0)
    mdd = max_drawdown(eq)
    if mdd >= 0:
        return 0.0
    return _safe_float(cagr / abs(mdd))


def win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for x in trade_pnls if x > 0)
    return wins / max(len(trade_pnls), 1)
