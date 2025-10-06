# main.py — Orquestación 60/20/20 + Optimización (objetivo: Calmar)
# Incluye silenciamiento de warnings, switches para pruebas rápidas
# y un chequeo opcional de conteo de señales.

import warnings
import random
import numpy as np
import pandas as pd

from optimization import (
    split_60_20_20,
    random_search,
    evaluate_with_params,
    _build_with_params,   # usado sólo para el chequeo de señales
)

# =========================
# CONFIGURACIÓN PRINCIPAL
# =========================
CSV_PATH = "Binance_BTCUSDT_1h.csv"

# Silenciar ruido en consola
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Reproducibilidad
random.seed(42)
np.random.seed(42)

# Ajustes para depuración rápida (modifíquelos a conveniencia)
N_TRIALS = 20           # aumente (40–80) para el corrido final
FOLDS = 3               # 3 usual; 2 si desea ir más rápido
FAST_DEBUG_LAST = None  # ej. 20000 para usar sólo las últimas ~20k velas durante pruebas


# =========================
# CARGA Y NORMALIZACIÓN
# =========================
def load_prices(path: str) -> pd.DataFrame:
    """Carga CSV (formato Binance), normaliza columnas y fechas."""
    try:
        df = pd.read_csv(path, skiprows=1).copy()
    except Exception:
        df = pd.read_csv(path).copy()

    # Normalización de nombres
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"Date": "dt", "tradecount": "Trades"})

    # Fecha
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    elif "Unix" in df.columns:
        df["dt"] = pd.to_datetime(df["Unix"], unit="ms", utc=True, errors="coerce")

    # OHLC a numérico
    for c in ("Open", "High", "Low", "Close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Limpieza/orden
    df = (
        df.dropna(subset=["dt", "Open", "High", "Low", "Close"])
          .sort_values("dt")
          .reset_index(drop=True)
    )

    # Recorte opcional para pruebas rápidas
    if FAST_DEBUG_LAST:
        df = df.tail(int(FAST_DEBUG_LAST)).reset_index(drop=True)

    print(f"Rango temporal: {df['dt'].iloc[0]} → {df['dt'].iloc[-1]} | Filas: {len(df):,}")
    return df


# =========================
# PROGRAMA PRINCIPAL
# =========================
def main(n_trials: int = N_TRIALS, folds: int = FOLDS) -> None:
    # 1) Datos
    data = load_prices(CSV_PATH)
    train, test, valid = split_60_20_20(data)
    print(f"Split -> TRAIN: {len(train):,} | TEST: {len(test):,} | VALID: {len(valid):,}")

    # 2) Optimización por Calmar con walk-forward
    best_params = random_search(train, n_trials=n_trials, folds=folds)

    # --- Chequeo opcional de señales para evitar 0 trades por filtros demasiado duros ---
    chk_tr = _build_with_params(train, best_params)
    chk_te = _build_with_params(test, best_params)
    chk_va = _build_with_params(valid, best_params)
    print(
        f"\nConteo de señales  ->  "
        f"TRAIN (L:{int(chk_tr['long_signal'].sum())}, S:{int(chk_tr['short_signal'].sum())}) | "
        f"TEST (L:{int(chk_te['long_signal'].sum())}, S:{int(chk_te['short_signal'].sum())}) | "
        f"VALID (L:{int(chk_va['long_signal'].sum())}, S:{int(chk_va['short_signal'].sum())})\n"
    )

    # 3) Evaluación con parámetros fijos en cada segmento
    rows = []
    for name, df in (("TRAIN", train), ("TEST", test), ("VALID", valid)):
        bt, met = evaluate_with_params(df, best_params)
        rows.append({
            "Set": name,
            "Calmar": met["Calmar"],
            "Sharpe": met["Sharpe"],
            "Sortino": met["Sortino"],
            "MaxDD": met["MaxDD"],
            "WinRate": met["WinRate"],
            "Trades": met["Trades"],
            "Equity0": met["Equity0"],
            "EquityF": met["EquityF"],
        })

    res = pd.DataFrame(
        rows,
        columns=["Set", "Calmar", "Sharpe", "Sortino", "MaxDD", "WinRate", "Trades", "Equity0", "EquityF"]
    )

    print("\n=== Resultados por segmento (parámetros fijos encontrados en TRAIN) ===")
    print(res.to_string(index=False))

    # 4) Guardado
    res.to_csv("results_60_20_20.csv", index=False)
    pd.Series(best_params).to_json("best_params.json", indent=2)
    print("\nGuardado: results_60_20_20.csv y best_params.json")


if __name__ == "__main__":
    main()




