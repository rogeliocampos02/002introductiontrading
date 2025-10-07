# plots.py
from __future__ import annotations
import os
import sys
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _series_with_dt_index(bt: pd.DataFrame, col: str) -> pd.Series:
    """
    Devuelve una Serie con índice DatetimeIndex usando bt['dt'] si existe.
    Si no hay 'dt', lanza un error claro (resample necesita DateTimeIndex).
    """
    s = pd.Series(bt[col]).astype(float)
    if "dt" in bt.columns:
        idx = pd.to_datetime(bt["dt"], utc=True, errors="coerce")
        s.index = idx
        s = s[~s.index.isna()]
        return s
    raise TypeError(
        "Para graficar o calcular retornos mensuales se requiere bt['dt'].\n"
        "Verifica que el backtest devuelva la columna 'dt' en el DataFrame."
    )


def _maybe_open(path: str, open_after_save: bool) -> None:
    if not open_after_save:
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def plot_equity(
    bt: pd.DataFrame,
    title: str = "",
    outdir: str = "figs",
    show: bool = True,
    open_after_save: bool = False,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    eq = _series_with_dt_index(bt, "equity")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq.values, lw=1.2)
    ax.set_title(f"Equity – {title}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(outdir, f"equity_{title}.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    _maybe_open(path, open_after_save)
    return path


def plot_drawdown(
    bt: pd.DataFrame,
    title: str = "",
    outdir: str = "figs",
    show: bool = True,
    open_after_save: bool = False,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    # si no tienes drawdown precalculado, lo calculamos desde equity
    eq = _series_with_dt_index(bt, "equity")
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(dd.index, dd.values, 0.0, color="tab:red", alpha=0.35, step="pre")
    ax.set_title(f"Drawdown – {title}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("DD")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(outdir, f"drawdown_{title}.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    _maybe_open(path, open_after_save)
    return path


def save_monthly_returns_csv(
    bt: pd.DataFrame,
    title: str = "",
    outdir: str = "figs",
) -> str:
    """
    Guarda un CSV con retornos mensuales (%). Requiere bt['dt'].
    """
    os.makedirs(outdir, exist_ok=True)

    eq = _series_with_dt_index(bt, "equity")
    ret = eq.pct_change().fillna(0.0) + 1.0  # factor diario/horario
    # retorno mensual compuesto
    monthly = ret.resample("M").apply(lambda x: np.prod(x) - 1.0)

    # tabla año x mes (en %)
    dfm = monthly.to_frame("ret")
    dfm["Year"] = dfm.index.year
    dfm["Month"] = dfm.index.strftime("%b")
    pivot = dfm.pivot(index="Year", columns="Month", values="ret").sort_index()
    pivot = (pivot * 100.0).round(2)  # a porcentaje, 2 decimales

    path = os.path.join(outdir, f"monthly_returns_{title}.csv")
    pivot.to_csv(path, float_format="%.2f")
    return path
