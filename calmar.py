# calmar.py
import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600

def _to_series(x):
    return pd.Series(x).astype(float)

def max_drawdown(equity) -> float:
    s = _to_series(equity)
    if len(s) < 2:
        return 0.0
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(-dd.min())

def sharpe(rets: pd.Series, periods_per_year: float = 24*365.25) -> float:
    r = pd.Series(rets).dropna()
    if len(r) < 2:
        return 0.0
    sigma = r.std()
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / (sigma + 1e-12))

def sortino(rets: pd.Series, periods_per_year: float = 24*365.25) -> float:
    r = pd.Series(rets).dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    sigma_d = downside.std()
    if sigma_d == 0 or np.isnan(sigma_d):
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / (sigma_d + 1e-12))

def cagr_from_equity(equity, dt_index: pd.Series) -> float:
    s = _to_series(equity)
    if len(s) < 2:
        return 0.0
    start, end = float(s.iloc[0]), float(s.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = (dt_index.iloc[-1] - dt_index.iloc[0]).total_seconds() / SECONDS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)

def calmar_ratio(equity, dt_index: pd.Series) -> float:
    cagr = cagr_from_equity(equity, dt_index)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)

def win_rate_from_pnls(trade_pnls) -> float:
    if not trade_pnls:
        return 0.0
    arr = np.array(trade_pnls, dtype=float)
    return float((arr > 0).mean())

def evaluate_bt(bt: pd.DataFrame) -> dict:
    eq = bt["equity"]
    rets = bt["ret"]
    dt = bt["dt"]
    metrics = {
        "Sharpe":  round(sharpe(rets), 4),
        "Sortino": round(sortino(rets), 4),
        "Calmar":  round(calmar_ratio(eq, dt), 4),
        "MaxDD":   round(max_drawdown(eq), 4),
        "WinRate": round(win_rate_from_pnls(bt.attrs.get("trade_pnls", [])), 4),
        "Trades":  len(bt.attrs.get("trade_pnls", [])),
        "Equity0": float(eq.iloc[0]),
        "EquityF": float(eq.iloc[-1]),
    }
    return metrics
