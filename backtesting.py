# backtesting.py
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class BTConfig:
    fee: float = 0.00125          # 0.125%
    capital0: float = 10_000.0
    risk_frac: float = 0.01       # % del equity en riesgo por trade (via SL)
    atr_sl: float = 2.5           # distancia SL = atr_sl * ATR
    atr_tp: float = 4.0           # distancia TP = atr_tp * ATR
    hold_min: int = 12            # barras mínimas en posición
    pos_cap: float = 0.20         # máximo % del equity comprometido por trade

def backtest(df: pd.DataFrame, cfg: BTConfig) -> pd.DataFrame:
    """
    Backtest long/short sin apalancamiento:
      - Largos: compra con fee; cierra con fee. Equity = cash + units*price
      - Cortos: reserva margen (value) + fee; cierra comprando y liberando margen.
        Equity = cash + short_margin + units*price  (units<0)
      - Sizing: shares = min( riesgo sobre SL, tope por pos_cap )
    Requiere columnas: dt, Open, High, Low, Close, atr, long_signal, short_signal
    """
    req = ["dt","Open","High","Low","Close","atr","long_signal","short_signal"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")

    d = df.reset_index(drop=True).copy()

    cash = float(cfg.capital0)
    position = 0               # +n (long), -n (short), 0 (flat)
    entry_price = None
    short_margin = 0.0         # margen bloqueado en cortos
    bars_held = 0

    equity_hist = []
    trade_pnls = []

    def mark_to_market(price: float) -> float:
        # equity = cash + valor MTM de posición + margen (si short)
        if position > 0:
            return cash + position * price
        elif position < 0:
            return cash + short_margin + position * price  # position<0
        else:
            return cash

    for _, row in d.iterrows():
        price = float(row["Close"])
        atr   = float(row["atr"])
        long_sig  = bool(row["long_signal"])
        short_sig = bool(row["short_signal"])

        # ======= SALIDAS =======
        if position != 0 and entry_price is not None:
            hit_sl = False
            hit_tp = False
            flip   = False

            if position > 0:
                sl = entry_price - cfg.atr_sl * atr
                tp = entry_price + cfg.atr_tp * atr
                hit_sl = price <= sl
                hit_tp = price >= tp
                flip   = short_sig and (bars_held >= cfg.hold_min)
            else:
                abs_pos = abs(position)
                sl = entry_price + cfg.atr_sl * atr
                tp = entry_price - cfg.atr_tp * atr
                hit_sl = price >= sl
                hit_tp = price <= tp
                flip   = long_sig and (bars_held >= cfg.hold_min)

            if hit_sl or hit_tp or flip:
                if position > 0:
                    # Cierre long
                    cash += position * price * (1.0 - cfg.fee)
                    pnl = position * (price - entry_price) - cfg.fee * position * (price + entry_price)
                    trade_pnls.append(pnl)
                    position = 0
                    entry_price = None
                    bars_held = 0
                else:
                    # Cierre short
                    abs_pos = abs(position)
                    buy_cost = abs_pos * price * (1.0 + cfg.fee)
                    cash += short_margin - buy_cost
                    pnl = abs_pos * (entry_price - price) - cfg.fee * abs_pos * (price + entry_price)
                    trade_pnls.append(pnl)
                    position = 0
                    entry_price = None
                    short_margin = 0.0
                    bars_held = 0

        # ======= ENTRADAS =======
        if position == 0:
            # Equity actual (flat => equity = cash)
            equity_now = cash

            # Sizing por riesgo (riesgo aprox. = SL distance)
            sl_dist = max(cfg.atr_sl * atr, 1e-8)
            shares_by_risk = int((equity_now * cfg.risk_frac) / sl_dist)
            if shares_by_risk < 1:
                shares_by_risk = 0

            # Tope por tamaño de posición
            cap_notional = equity_now * cfg.pos_cap
            shares_cap_long  = int(cap_notional / (price * (1.0 + cfg.fee)))
            shares_cap_short = int(cap_notional / price)

            if long_sig:
                shares = max(0, min(shares_by_risk, shares_cap_long))
                cost = shares * price * (1.0 + cfg.fee)
                if shares >= 1 and cost <= cash:
                    cash -= cost
                    position = shares
                    entry_price = price
                    bars_held = 0

            elif short_sig:
                shares = max(0, min(shares_by_risk, shares_cap_short))
                margin = shares * price
                fee_open = shares * price * cfg.fee
                if shares >= 1 and (margin + fee_open) <= cash:
                    cash -= (margin + fee_open)
                    position = -shares
                    entry_price = price
                    short_margin = margin
                    bars_held = 0

        # ======= MTM + registro =======
        equity_hist.append(mark_to_market(price))
        if position != 0:
            bars_held += 1

    out = d.copy()
    out["equity"] = pd.Series(equity_hist, index=out.index)
    out["ret"] = out["equity"].pct_change().fillna(0.0)
    out.attrs["trade_pnls"] = trade_pnls
    return out


