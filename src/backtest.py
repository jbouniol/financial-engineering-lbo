"""Backtesting engine for the Bond ETF Rotation strategy.

Signals : V1 (régime sur niveau du spread 2s10s),
          V4 (V1 + filtre trend 3 mois pour bypass duration en marché baissier).

Engine  : vectorisé, rebalancing mensuel, weights laggés d'un jour pour
          exclure le look-ahead bias. Transaction costs et slippage sont
          appliqués au turnover one-way.

Metrics : equity curve, CAGR, Sharpe, Sortino, max DD, vol, win rate,
          turnover annualisé, vs buy & hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE = ["TLT", "IEF", "SHY"]
ANN = 252


# -----------------------------
# Signals
# -----------------------------

def signal_v1(yields: pd.DataFrame, rebal: str = "ME") -> pd.DataFrame:
    """Bucket sur le niveau du spread 2s10s, échantillonné en fin de mois.

    spread > 1.0   → TLT (long duration)
    0 < spread ≤ 1 → IEF (intermediate)
    spread ≤ 0     → SHY (short duration, refuge)
    """
    spread = (yields["DGS10"] - yields["DGS2"]).resample(rebal).last().dropna()
    w = pd.DataFrame(0.0, index=spread.index, columns=UNIVERSE)
    w.loc[spread > 1.0, "TLT"] = 1.0
    w.loc[(spread > 0) & (spread <= 1.0), "IEF"] = 1.0
    w.loc[spread <= 0, "SHY"] = 1.0
    return w


def signal_v4(yields: pd.DataFrame, prices: pd.DataFrame, rebal: str = "ME",
              trend_lookback_months: int = 3) -> pd.DataFrame:
    """V1 + filtre trend : si l'ETF sélectionné a un return négatif sur
    `trend_lookback_months`, on bascule sur SHY (cash-like).

    Confirmation prix au-dessus du seul signal macro pour éviter de tenir
    une duration en plein trend baissier (ex. 2022)."""
    base = signal_v1(yields, rebal=rebal)
    monthly_px = prices.resample(rebal).last().reindex(base.index).ffill()
    trend = monthly_px.pct_change(trend_lookback_months)

    selected = base.idxmax(axis=1)
    sel_trend = pd.Series(
        [trend.loc[d, etf] if pd.notna(trend.loc[d, etf]) else 0.0
         for d, etf in selected.items()],
        index=base.index,
    )

    out = base.copy()
    bad = sel_trend < 0
    out.loc[bad] = 0.0
    out.loc[bad, "SHY"] = 1.0
    return out


# -----------------------------
# Engine
# -----------------------------

def monthly_to_daily_weights(monthly_w: pd.DataFrame, daily_index: pd.DatetimeIndex,
                             execution_lag: int = 1) -> pd.DataFrame:
    """Décale les poids mensuels d'un jour ouvré (signal à t → exécution close t+1)
    puis reindex sur le calendrier daily avec forward-fill."""
    effective = monthly_w.copy()
    effective.index = effective.index + pd.tseries.offsets.BusinessDay(execution_lag)
    daily = effective.reindex(daily_index, method="ffill")
    return daily.fillna(0.0)


def run_backtest(prices: pd.DataFrame, weights_monthly: pd.DataFrame,
                 tc_bps: float = 0.0, slip_bps: float = 0.0) -> dict:
    """Backtest vectorisé.

    tc_bps   : coût de transaction (one-way, en bps de notional traded).
    slip_bps : slippage (one-way, en bps).

    Returns un dict avec equity, returns, weights, turnover, costs.
    """
    prices = prices[UNIVERSE].copy()
    rets = prices.pct_change().fillna(0.0)
    w = monthly_to_daily_weights(weights_monthly, prices.index)

    gross_ret = (w * rets).sum(axis=1)

    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    cost_rate = (tc_bps + slip_bps) / 10000.0
    cost = turnover * cost_rate

    net_ret = gross_ret - cost
    equity = (1.0 + net_ret).cumprod()

    return {
        "weights": w,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "turnover": turnover,
        "cost": cost,
        "equity": equity,
    }


# -----------------------------
# Metrics
# -----------------------------

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min()


def perf_metrics(net_ret: pd.Series, equity: pd.Series,
                 turnover: pd.Series | None = None,
                 rf: float = 0.0) -> dict:
    n = len(net_ret)
    if n == 0:
        return {}
    years = n / ANN
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = net_ret.std() * np.sqrt(ANN)
    sharpe = (net_ret.mean() * ANN - rf) / vol if vol > 0 else np.nan
    downside = net_ret[net_ret < 0]
    sortino = (net_ret.mean() * ANN - rf) / (downside.std() * np.sqrt(ANN)) if len(downside) > 1 else np.nan
    mdd = max_drawdown(equity)
    win = (net_ret > 0).mean()
    out = {
        "CAGR_%":     cagr * 100,
        "Vol_%":      vol * 100,
        "Sharpe":     sharpe,
        "Sortino":    sortino,
        "MaxDD_%":    mdd * 100,
        "WinRate_%":  win * 100,
    }
    if turnover is not None:
        out["Turnover_ann"] = turnover.sum() / (n / ANN)
    return out


def buy_and_hold(prices: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Helper : buy-and-hold avec poids fixes (sera utilisé comme benchmark)."""
    w_df = pd.DataFrame([weights], index=[prices.index[0]])[UNIVERSE]
    return run_backtest(prices, w_df, tc_bps=0, slip_bps=0)
