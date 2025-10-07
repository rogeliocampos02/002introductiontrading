# backtesting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class BTConfig:
    fee: float = 0.00125          # 0.125% por operación
    pos_frac: float = 0.25        # fracción del equity por trade (0..1)
    atr_sl: float = 3.0           # stop = entry +/- atr * atr_sl
    atr_tp: float = 5.0           # tp   = entry +/- atr * atr_tp
    hold_min: int = 12            # barras mínimas antes de permitir salida/flip
    start_capital: float = 10_000 # capital inicial
    qty_step: float = 0.001       # tamaño mínimo de unidad (p. ej. 0.001 BTC)


_REQ_COLS = ["dt", "Open", "High", "Low", "Close", "long_signal", "short_signal"]


def _ensure_atr(df: pd.DataFrame, colname: str = "atr", window: int = 14) -> pd.Series:
    """Devuelve df[colname] si existe; si no, calcula ATR tipo Wilder."""
    if colname in df.columns:
        return df[colname].astype(float)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / float(window), adjust=False).mean()
    return atr


def backtest(df: pd.DataFrame, cfg: BTConfig) -> pd.DataFrame:
    """
    Backtest long/short sin apalancamiento con:
      - tamaños fraccionarios (qty_step)
      - comisiones (fee)
      - SL/TP por ATR (atr_sl, atr_tp)
      - hold mínimo (hold_min)
    Requisitos: dt, Open, High, Low, Close, long_signal, short_signal
    Si no existe 'atr', se calcula.

    Devuelve df_out con columnas 'equity' y 'ret'.
    En df_out.attrs['trade_pnls'] queda la lista de PnLs realizados por trade.
    """
    for c in _REQ_COLS:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: '{c}'")

    d = df.reset_index(drop=True).copy()
    d["long_signal"] = d["long_signal"].astype(bool)
    d["short_signal"] = d["short_signal"].astype(bool)

    atr = _ensure_atr(d, "atr")
    price = d["Close"].astype(float)

    # Estado
    cash: float = float(cfg.start_capital)
    position: float = 0.0
    entry_price: float | None = None
    bars_held: int = 0

    equity_path: List[float] = []
    trade_pnls: List[float] = []

    def _equity_mark_to_market(px: float) -> float:
        return float(cash + position * px)

    def _units_affordable(px: float, equ: float) -> float:
        if equ <= 0 or px <= 0:
            return 0.0
        size_val = equ * float(cfg.pos_frac)
        raw_units = size_val / px
        step = max(float(getattr(cfg, "qty_step", 0.001)), 1e-9)
        units = np.floor(raw_units / step) * step
        min_cost = units * px * (1.0 + cfg.fee)
        if units <= 0 or min_cost > equ:
            return 0.0
        return float(units)

    for i, row in d.iterrows():
        px = float(price.iloc[i])
        bar_atr = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0

        # --- Salidas / Flip ---
        if position != 0.0:
            bars_held += 1
            can_exit = bars_held >= int(cfg.hold_min)

            if entry_price is not None:
                if position > 0:  # LONG
                    sl = entry_price - bar_atr * cfg.atr_sl
                    tp = entry_price + bar_atr * cfg.atr_tp
                    hit_sl = px <= sl
                    hit_tp = px >= tp
                else:            # SHORT
                    sl = entry_price + bar_atr * cfg.atr_sl
                    tp = entry_price - bar_atr * cfg.atr_tp
                    hit_sl = px >= sl
                    hit_tp = px <= tp
            else:
                hit_sl = hit_tp = False

            flip_long  = bool(row["long_signal"])  and position < 0
            flip_short = bool(row["short_signal"]) and position > 0

            if can_exit and (hit_sl or hit_tp or flip_long or flip_short):
                if position > 0:
                    gross = position * px
                    fee_paid = gross * cfg.fee
                    cash += (gross - fee_paid)
                    pnl = (px - float(entry_price)) * position
                else:
                    qty = abs(position)
                    cost = qty * px
                    fee_paid = cost * cfg.fee
                    cash -= (cost + fee_paid)
                    pnl = (float(entry_price) - px) * qty

                trade_pnls.append(float(pnl))
                position = 0.0
                entry_price = None
                bars_held = 0

                # Flip de posición si aplica
                if flip_long or flip_short:
                    equ = _equity_mark_to_market(px)
                    units = _units_affordable(px, equ)
                    if units > 0.0:
                        if flip_long:
                            entry_cost = units * px
                            fee_in = entry_cost * cfg.fee
                            total = entry_cost + fee_in
                            if total <= cash:
                                cash -= total
                                position += units
                                entry_price = px
                                bars_held = 0
                        else:  # flip_short
                            proceeds = units * px
                            fee_in = proceeds * cfg.fee
                            cash += (proceeds - fee_in)
                            position -= units
                            entry_price = px
                            bars_held = 0

        # --- Entradas ---
        elif position == 0.0:
            if bool(row["long_signal"]) or bool(row["short_signal"]):
                equ = _equity_mark_to_market(px)
                units = _units_affordable(px, equ)
                if units > 0.0:
                    if bool(row["long_signal"]) and not bool(row["short_signal"]):
                        entry_cost = units * px
                        fee_in = entry_cost * cfg.fee
                        total = entry_cost + fee_in
                        if total <= cash:
                            cash -= total
                            position += units
                            entry_price = px
                            bars_held = 0
                    elif bool(row["short_signal"]) and not bool(row["long_signal"]):
                        proceeds = units * px
                        fee_in = proceeds * cfg.fee
                        cash += (proceeds - fee_in)
                        position -= units
                        entry_price = px
                        bars_held = 0

        equity_path.append(_equity_mark_to_market(px))

    out = d.copy()
    out["equity"] = pd.Series(equity_path, index=out.index)
    out["ret"] = out["equity"].pct_change().fillna(0.0)
    out.attrs["trade_pnls"] = trade_pnls
    return out


if __name__ == "__main__":
    # Protegemos ejecución directa para no romper con variables inexistentes.
    print(
        "backtesting.py es un módulo. Úsalo importándolo:\n"
        "  from backtesting import BTConfig, backtest\n"
        "y llama backtest(df, cfg)."
    )
