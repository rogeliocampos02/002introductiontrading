# main.py
# ------------------------------------------------------------
# Driver limpio: carga datos, split 60/20/20, random search en TRAIN,
# imprime tabla final, guarda resultados y genera gráficas.
# ------------------------------------------------------------
from __future__ import annotations
import json
import logging
import os
import warnings  # para silenciar FutureWarning
import pandas as pd

from optlib import split_60_20_20, random_search, evaluate_with_params
from plots import plot_equity, plot_drawdown, save_monthly_returns_csv
from calmar import total_return_pct  # % de rendimiento total

# === CONFIG ===
CSV_PATH   = "Binance_BTCUSDT_1h.csv"
N_TRIALS   = 50
SEED       = 42
SHOW_PLOTS = True            
OUT_DIR    = "figs"

SHOW_WINDOWS    = True       # abre ventanas de Matplotlib
OPEN_AFTER_SAVE = False      # abre los PNG con el visor del SO

# Silenciar todo lo que no sea error
logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(message)s")
# Silenciar FutureWarning de pandas (por cambios futuros de downcasting, etc.)
warnings.filterwarnings("ignore", category=FutureWarning)


def _fmt_table(rows):
    df = pd.DataFrame(rows)
    cols = [
        "Set", "Calmar", "Sharpe", "Sortino", "MaxDD",
        "WinRate", "Trades", "Equity0", "EquityF", "ReturnPct"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    out = []
    head = "  ".join(f"{c:>10}" for c in cols)
    out.append(head)

    for _, r in df.iterrows():
        line_items = []
        for c in cols:
            if c == "Set":
                line_items.append(f"{str(r[c]):<10}")
            elif c in ("Trades",):
                line_items.append(f"{int(r[c]):>10d}")
            elif c in ("Equity0", "EquityF"):
                line_items.append(f"{float(r[c]):>10.1f}")
            elif c == "ReturnPct":
                line_items.append(f"{float(r[c]):>9.2f}%")
            else:
                line_items.append(f"{float(r[c]):>10.4f}")
        out.append("  ".join(line_items))
    return "\n".join(out)


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=1).copy()
    data.columns = [c.strip().replace(" ", "_") for c in data.columns]
    data = data.rename(
        columns={
            "Date": "dt",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume_BTC": "Volume_BTC",
            "Volume_USDT": "Volume_USDT",
            "tradecount": "Trades",
            "tradcount": "Trades",
        }
    )
    data["dt"] = pd.to_datetime(data["dt"], utc=True, errors="coerce")
    data = data.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    req = ["dt", "Open", "High", "Low", "Close"]
    for c in req:
        if c not in data.columns:
            raise ValueError(f"Falta columna {c} en el CSV")
    return data


def main() -> None:
    # 1) Datos
    df = load_data(CSV_PATH)
    train, test, valid = split_60_20_20(df)

    # 2) Optimización en TRAIN
    best_params = random_search(train, n_iter=N_TRIALS, seed=SEED)

    # Guardar mejores parámetros para poder revertir fácilmente
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    # 3) Evaluación con los parámetros fijos
    rows = []
    by_set_bt = {}  # guardamos los backtests para graficar

    for name, part in [("TRAIN", train), ("TEST", test), ("VALID", valid)]:
        bt, metrics = evaluate_with_params(part, best_params, start_capital=10_000)
        metrics["ReturnPct"] = total_return_pct(bt["equity"])  # añadir %
        row = {"Set": name}
        row.update(metrics)
        rows.append(row)
        by_set_bt[name] = bt

    # 4) Resumen 
    print("\n=== Resultados por segmento (parámetros fijos encontrados en TRAIN) ===")
    print(_fmt_table(rows))

    # Guardar resultados a CSV 
    cols_csv = [
        "Set", "Calmar", "Sharpe", "Sortino", "MaxDD",
        "WinRate", "Trades", "Equity0", "EquityF", "ReturnPct"
    ]
    pd.DataFrame(rows)[cols_csv].to_csv("results_60_20_20.csv", index=False)

    # 5) Gráficas
    if SHOW_PLOTS:
        os.makedirs(OUT_DIR, exist_ok=True)
        for name, bt in by_set_bt.items():
            eq_png = plot_equity(
                bt, title=name, outdir=OUT_DIR,
                show=SHOW_WINDOWS, open_after_save=OPEN_AFTER_SAVE
            )
            dd_png = plot_drawdown(
                bt, title=name, outdir=OUT_DIR,
                show=SHOW_WINDOWS, open_after_save=OPEN_AFTER_SAVE
            )
            mr_csv = save_monthly_returns_csv(bt, title=name, outdir=OUT_DIR)
            print(f"[Guardado] {name}: {eq_png} | {dd_png} | {mr_csv}")


if __name__ == "__main__":
    main()

