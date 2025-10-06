# optimization.py
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from createsignals import add_indicators, add_signals
from backtesting import BTConfig, backtest
from calmar import evaluate_bt, calmar_ratio


# ===============================
# 60 / 20 / 20 split por tiempo
# ===============================
def split_60_20_20(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * 0.60)
    i2 = int(n * 0.80)
    train = df.iloc[:i1].copy()
    test  = df.iloc[i1:i2].copy()
    valid = df.iloc[i2:].copy()
    return train, test, valid


# ==========================================
# Helpers: indicadores + señales + backtest
# ==========================================
def _build_with_params(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
    """
    Calcula indicadores y señales con los parámetros p sobre df.
    """
    x = add_indicators(
        df,
        ema_fast=p["ema_fast"],
        ema_slow=p["ema_slow"],
        rsi_window=p["rsi_window"],
        donch_window=p["donch_window"],
        ema_longterm=720,          # macro filter para cortos
    )
    x = add_signals(
        x,
        rsi_upper=p["rsi_upper"],
        rsi_lower=p["rsi_lower"],
        adx_thr=p["adx_thr"],
        atr_q=p["atr_q"],
    )
    return x


def _run_with_params(df: pd.DataFrame, p: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Ejecuta backtest con parámetros p y devuelve (bt_df, metrics_dict).
    """
    x = _build_with_params(df, p)

    cfg = BTConfig(
        fee=0.00125,               # 0.125%
        capital0=10_000,
        risk_frac=p["risk_frac"],  # % del equity arriesgado por operación
        atr_sl=p["atr_sl"],
        atr_tp=p["atr_tp"],
        hold_min=p["hold_min"],
        pos_cap=p["pos_cap"],      # tope de exposición sobre cash
    )
    bt = backtest(x, cfg)
    met = evaluate_bt(bt)
    return bt, met


# =====================================================
# Walk-forward score dentro del TRAIN (evitar overfit)
# =====================================================
def walk_forward_score(train_df: pd.DataFrame, p: Dict, folds: int = 3) -> float:
    """
    Divide train en 'folds' secuenciales y evalúa Calmar en los tramos
    out-of-sample, promediando el resultado. Devuelve el promedio de Calmar.
    """
    n = len(train_df)
    if n < folds + 10:  # sanity check
        return -1e9

    bounds = np.linspace(0, n, folds + 1, dtype=int)
    calmars = []

    for i in range(folds - 1):
        is_end = bounds[i + 1]   # fin del in-sample
        os_end = bounds[i + 2]   # fin del out-of-sample

        ins = train_df.iloc[:is_end].copy()
        oos = train_df.iloc[is_end:os_end].copy()

        # evitar segmentos OOS demasiado cortos
        if len(oos) < 300:
            continue

        # Construimos indicadores con ins+oos (para no truncar rolling),
        # pero medimos métricas sólo en el tramo OOS.
        both = pd.concat([ins, oos], axis=0, ignore_index=True)
        bt, _ = _run_with_params(both, p)

        # recorte OOS por dt
        start_oos = oos["dt"].iloc[0]
        end_oos   = oos["dt"].iloc[-1]
        msk = (bt["dt"] >= start_oos) & (bt["dt"] <= end_oos)
        bt_oos = bt.loc[msk].reset_index(drop=True)

        if len(bt_oos) > 20:
            calmars.append(calmar_ratio(bt_oos["equity"], bt_oos["dt"]))

    if not calmars:
        return -1e9
    return float(np.mean(calmars))


def _sample_params() -> Dict:
    # Tendencia (EMAs)
    ema_fast  = random.randint(10, 30)
    ema_slow  = random.randint(80, 220)
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + random.randint(20, 80)

    # Momentum (RSI)
    rsi_window = random.randint(12, 20)
    rsi_lower  = random.randint(42, 48)     # menos estricto
    rsi_upper  = random.randint(55, 62)     # menos estricto

    # Breakout (Donchian)
    donch_window = random.randint(40, 90)

    # Filtros
    adx_thr = random.uniform(12.0, 22.0)    # más permisivo
    # 30%–75%: deja pasar más barras; si quieres desactivar, pondremos atr_q=None abajo
    atr_q   = random.uniform(0.30, 0.75)
    # ~20% de las veces no filtrar por ATR:
    if random.random() < 0.2:
        atr_q = None

    # Gestión del riesgo (prudente)
    risk_frac = random.uniform(0.005, 0.02)  # 0.5%–2% del equity
    atr_sl    = random.uniform(2.0, 4.0)
    atr_tp    = random.uniform(3.5, 7.0)
    hold_min  = random.randint(12, 48)
    pos_cap   = random.uniform(0.10, 0.25)

    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_window": rsi_window,
        "rsi_lower": rsi_lower,
        "rsi_upper": rsi_upper,
        "donch_window": donch_window,
        "adx_thr": adx_thr,
        "atr_q": atr_q,
        "risk_frac": risk_frac,
        "atr_sl": atr_sl,
        "atr_tp": atr_tp,
        "hold_min": hold_min,
        "pos_cap": pos_cap,
    }


# ==========================
# Random Search por Calmar
# ==========================
def random_search(train_df: pd.DataFrame, n_trials: int = 40, folds: int = 3) -> Dict:
    best_score = -1e9
    best_params = None

    for t in range(n_trials):
        p = _sample_params()
        score = walk_forward_score(train_df, p, folds=folds)
        if score > best_score:
            best_score, best_params = score, p
        print(f"[{t+1}/{n_trials}] WF-Calmar: {round(score, 4)} | Best: {round(best_score, 4)}")

    print("\nMejores parámetros (según Walk-Forward en TRAIN):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    return best_params


# ===============================================
# Evaluación con parámetros fijos en cualquier set
# ===============================================
def evaluate_with_params(df: pd.DataFrame, p: Dict) -> Tuple[pd.DataFrame, Dict]:
    bt, met = _run_with_params(df, p)
    return bt, met
