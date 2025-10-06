# main.py
# ------------------------------------------------------------
# Driver: carga datos, split 60/20/20, random search en TRAIN
# e informe de métricas en TRAIN/TEST/VALID con los mejores
# parámetros encontrados en TRAIN.
# ------------------------------------------------------------
from __future__ import annotations
import logging
import pandas as pd

from optlib import split_60_20_20, random_search, evaluate_with_params

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


CSV_PATH = "Binance_BTCUSDT_1h.csv"   # ajusta si está en otra ruta
N_TRIALS = 50
SEED = 42


def _fmt_table(rows):
    # filas: dict con keys iguales
    df = pd.DataFrame(rows)
    cols = ["Set", "Calmar", "Sharpe", "Sortino", "MaxDD", "WinRate",
            "Trades", "Equity0", "EquityF"]
    df = df[cols]
    # Ancho fijo ascii
    out = []
    head = "  ".join(f"{c:>8}" for c in cols)
    out.append(head)
    for _, r in df.iterrows():
        line = "  ".join(
            [
                f"{r['Set']:<8}",
                f"{r['Calmar']:>8.4f}",
                f"{r['Sharpe']:>8.4f}",
                f"{r['Sortino']:>8.4f}",
                f"{r['MaxDD']:>8.4f}",
                f"{r['WinRate']:>8.4f}",
                f"{int(r['Trades']):>8d}",
                f"{r['Equity0']:>8.1f}",
                f"{r['EquityF']:>8.1f}",
            ]
        )
        out.append(line)
    return "\n".join(out)


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=1).copy()
    # Normaliza nombres
    data.columns = [c.strip().replace(" ", "_") for c in data.columns]
    # Renombres esperados
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
    # Columns mínimas
    req = ["dt", "Open", "High", "Low", "Close"]
    for c in req:
        if c not in data.columns:
            raise ValueError(f"Falta columna {c} en el CSV")
    return data


def main():
    logging.info("Cargando datos…")
    df = load_data(CSV_PATH)

    train, test, valid = split_60_20_20(df)

    logging.info("Optimizando (Random Search baseline)…")
    best_params = random_search(train, n_iter=N_TRIALS, seed=SEED)

    logging.info("Mejores parámetros (TRAIN):")
    for k, v in best_params.items():
        logging.info(f"  {k}: {v}")

    rows = []
    for name, part in [("TRAIN", train), ("TEST", test), ("VALID", valid)]:
        _, metrics = evaluate_with_params(part, best_params, start_capital=10_000)
        row = {"Set": name}
        row.update(metrics)
        rows.append(row)

    print("\n=== Resultados por segmento (parámetros fijos encontrados en TRAIN) ===")
    print(_fmt_table(rows))


if __name__ == "__main__":
    main()
