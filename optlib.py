# optlib.py
# ------------------------------------------------------------
# Split de datos, evaluación con parámetros y Random Search
# ------------------------------------------------------------
from __future__ import annotations
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import createsignals as signals
from backtesting import BTConfig, backtest
from calmar import sharpe, sortino, calmar, max_drawdown, win_rate


def split_60_20_20(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * 0.60)
    i2 = int(n * 0.80)
    train = df.iloc[:i1].reset_index(drop=True)
    test = df.iloc[i1:i2].reset_index(drop=True)
    valid = df.iloc[i2:].reset_index(drop=True)
    return train, test, valid


def evaluate_with_params(df: pd.DataFrame, params: Dict, start_capital: float = 10_000):
    # 1) Indicadores & señales
    dd = signals.add_indicators(
        df,
        ema_fast=int(params.get("ema_fast", 12)),
        ema_slow=int(params.get("ema_slow", 26)),
        rsi_window=int(params.get("rsi_window", 14)),
        atr_window=int(params.get("atr_window", 14)),
        donch_window=int(params.get("donch_window", 20)),
    )
    dd = signals.add_signals(
        dd,
        rsi_upper=float(params.get("rsi_upper", 60)),
        rsi_lower=float(params.get("rsi_lower", 40)),
        atr_q=float(params.get("atr_q", 0.5)),
    )

    # 2) Backtest
    cfg = BTConfig(
        fee=float(params.get("fee", 0.00125)),
        pos_frac=float(params.get("pos_frac", 0.25)),
        atr_sl=float(params.get("atr_sl", 3.0)),
        atr_tp=float(params.get("atr_tp", 5.0)),
        hold_min=int(params.get("hold_min", 12)),
        start_capital=float(params.get("start_capital", start_capital)),
    )
    bt = backtest(dd, cfg)

    # 3) Métricas
    eq, rets = bt["equity"], bt["ret"]
    trade_pnls = bt.attrs.get("trade_pnls", [])

    metrics = {
        "Sharpe": round(sharpe(rets), 4),
        "Sortino": round(sortino(rets), 4),
        "Calmar": round(calmar(eq, bt["dt"]), 4),
        "MaxDD": round(max_drawdown(eq), 4),
        "WinRate": round(win_rate(trade_pnls), 4),
        "Trades": len(trade_pnls),
        "Equity0": float(eq.iloc[0]),
        "EquityF": float(eq.iloc[-1]),
    }
    return bt, metrics


def random_search(
    train_df: pd.DataFrame,
    n_iter: int = 50,
    seed: int = 42,
) -> Dict:
    """
    Random Search simple sobre un espacio razonable.
    Devuelve el diccionario de mejores parámetros por Calmar.
    """
    random.seed(seed)
    best = None
    best_score = -1e18

    for _ in range(int(n_iter)):
        p = {
            "ema_fast": random.randint(8, 20),
            "ema_slow": random.randint(21, 60),
            "rsi_window": random.randint(10, 30),
            "atr_window": random.randint(10, 30),
            "donch_window": random.randint(20, 60),
            "rsi_upper": random.uniform(55, 70),
            "rsi_lower": random.uniform(30, 45),
            "atr_q": random.uniform(0.3, 0.9),
            "pos_frac": random.uniform(0.10, 0.40),
            "atr_sl": random.uniform(2.0, 4.5),
            "atr_tp": random.uniform(4.0, 7.0),
            "hold_min": random.randint(6, 36),
            "fee": 0.00125,
            "start_capital": 10_000,
        }

        _, m = evaluate_with_params(train_df, p)
        score = m["Calmar"]
        if score > best_score:
            best_score = score
            best = p

    return best
