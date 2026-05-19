"""Backtesting engine for the Bond ETF Rotation strategy.

Signals : V1 (régime sur niveau du spread 2s10s),
          V4 (V1 + filtre trend 3 mois pour bypass duration en marché baissier).

Engine  : vectorisé, rebalancing mensuel, weights laggés d'un jour pour
          exclure le look-ahead bias.

Garanties anti-look-ahead :
- Les yields FRED sont shiftés d'un jour ouvré avant la construction du signal
  (DGS2/DGS10 du jour t ne sont publiés qu'en t+1).
- Le rebalancing se fait au dernier jour de TRADING ETF du mois (pas le dernier
  jour calendaire — robuste aux weekends et US holidays).
- L'exécution est décalée en jours de trading réels (pas BusinessDay civil) via
  le calendrier `prices.index`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE = ["TLT", "IEF", "SHY"]
ANN = 252


# -----------------------------
# Calendar helpers
# -----------------------------

def _month_end_trading_days(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dernier jour de trading de chaque mois dans `daily_index`."""
    s = pd.Series(daily_index, index=daily_index)
    last = s.groupby(daily_index.to_period("M")).last()
    return pd.DatetimeIndex(last.values)


def _spread_at_rebal(yields: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Spread 2s10s mesuré au dernier jour de trading du mois, en utilisant la
    valeur publiée au jour précédent (shift de 1 pour absorber le délai FRED)."""
    spread = (yields["DGS10"] - yields["DGS2"]).shift(1)
    spread_on_etf = spread.reindex(prices.index).ffill()
    return spread_on_etf.loc[_month_end_trading_days(prices.index)].dropna()


# -----------------------------
# Signals
# -----------------------------

def signal_v1(yields: pd.DataFrame, prices: pd.DataFrame,
              threshold_low: float = 0.0, threshold_high: float = 1.0) -> pd.DataFrame:
    """Bucket sur le niveau du spread 2s10s, échantillonné au dernier jour de
    trading du mois (calendrier ETF).

    spread > threshold_high           → TLT (long duration)
    threshold_low < spread ≤ high     → IEF (intermediate)
    spread ≤ threshold_low            → SHY (short duration, refuge)

    Valeurs par défaut (0 et 1%) choisies a priori sur base économique :
    inversion = signal de récession ; prime de terme \"normale\" > 100 bps.
    """
    spread = _spread_at_rebal(yields, prices)
    w = pd.DataFrame(0.0, index=spread.index, columns=UNIVERSE)
    w.loc[spread > threshold_high, "TLT"] = 1.0
    w.loc[(spread > threshold_low) & (spread <= threshold_high), "IEF"] = 1.0
    w.loc[spread <= threshold_low, "SHY"] = 1.0
    return w


def signal_v4(yields: pd.DataFrame, prices: pd.DataFrame,
              trend_lookback_months: int = 3,
              threshold_low: float = 0.0, threshold_high: float = 1.0) -> pd.DataFrame:
    """V1 + filtre trend : si l'ETF sélectionné a un return négatif sur
    `trend_lookback_months`, on bascule sur SHY."""
    base = signal_v1(yields, prices, threshold_low=threshold_low, threshold_high=threshold_high)
    if base.empty:
        return base

    has_signal = base.sum(axis=1) > 0
    monthly_px = prices.reindex(base.index).ffill()
    trend = monthly_px.pct_change(trend_lookback_months)

    selected = base.where(has_signal).idxmax(axis=1)
    sel_trend = pd.Series(
        [trend.loc[d, etf] if pd.notna(etf) and pd.notna(trend.loc[d, etf]) else 0.0
         for d, etf in selected.items()],
        index=base.index,
    )

    out = base.copy()
    bad = (sel_trend < 0) & has_signal
    out.loc[bad] = 0.0
    out.loc[bad, "SHY"] = 1.0
    return out


# -----------------------------
# Engine
# -----------------------------

def monthly_to_daily_weights(monthly_w: pd.DataFrame, daily_index: pd.DatetimeIndex,
                             execution_lag: int = 1) -> pd.DataFrame:
    """Décale les poids mensuels de `execution_lag` jours de **trading**
    (calendrier daily_index, pas BusinessDay civil) puis forward-fill.

    Signal posé à la clôture t → poids effectifs à la clôture t + lag jours
    de trading. Garantit qu'on n'utilise jamais d'info publiée après t.
    """
    cols = list(monthly_w.columns)
    if len(monthly_w) == 0:
        return pd.DataFrame(0.0, index=daily_index, columns=cols)

    pos = daily_index.get_indexer(monthly_w.index, method="pad")
    eff_pos = pos + execution_lag

    mask = (pos >= 0) & (eff_pos < len(daily_index))
    if not mask.any():
        return pd.DataFrame(0.0, index=daily_index, columns=cols)

    effective = monthly_w.loc[mask].copy()
    effective.index = daily_index[eff_pos[mask]]
    effective = effective[~effective.index.duplicated(keep="last")]

    daily = effective.reindex(daily_index, method="ffill").fillna(0.0)
    return daily


def run_backtest(prices: pd.DataFrame, weights_monthly: pd.DataFrame,
                 tc_bps: float = 0.0, slip_bps: float = 0.0,
                 execution_lag: int = 1) -> dict:
    """Backtest vectorisé.

    tc_bps         : coût de transaction (one-way, en bps de notional traded).
    slip_bps       : slippage (one-way, en bps).
    execution_lag  : décalage signal → exécution, en jours de **trading** ETF.
                     1 = T+1 (backtest standard), 2 = T+2 (mode paper prudent).

    Returns un dict avec equity, returns, weights, turnover, costs.
    """
    prices = prices[UNIVERSE].copy()
    rets = prices.pct_change().fillna(0.0)
    w = monthly_to_daily_weights(weights_monthly, prices.index, execution_lag=execution_lag)

    gross_ret = (w * rets).sum(axis=1)

    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    cost_rate = (tc_bps + slip_bps) / 10000.0
    cost = turnover * cost_rate

    net_ret = gross_ret - cost

    # CAGR robuste : on ne compte que la période effective où on est investi.
    active_mask = w.sum(axis=1) > 0
    if active_mask.any():
        start = active_mask.idxmax()
        eq = pd.Series(1.0, index=prices.index)
        eq.loc[start:] = (1.0 + net_ret.loc[start:]).cumprod()
    else:
        eq = pd.Series(1.0, index=prices.index)

    return {
        "weights": w,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "turnover": turnover,
        "cost": cost,
        "equity": eq,
        "first_active": active_mask.idxmax() if active_mask.any() else None,
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
                 first_active: pd.Timestamp | None = None,
                 rf: float = 0.0) -> dict:
    """Métriques calculées sur la période effective (à partir de `first_active`
    si fourni). rf est annualisé (taux sans risque). Pour rendre rf comparable,
    on retire rf annualisé du return annualisé du portefeuille."""
    if first_active is not None:
        net_ret = net_ret.loc[first_active:]
        equity = equity.loc[first_active:]
        if turnover is not None:
            turnover = turnover.loc[first_active:]

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


def trade_log(prices: pd.DataFrame, weights_monthly: pd.DataFrame,
              execution_lag: int = 1) -> pd.DataFrame:
    """Liste les rotations effectives avec leur prix théorique (close du jour
    du signal) et prix exécuté (close après `execution_lag` jours de trading).
    `slip_bps` mesure le drift de marché entre les deux (≠ slippage modélisé)."""
    sel = weights_monthly.idxmax(axis=1)
    rotations_mask = sel != sel.shift(1)
    rotations_mask.iloc[0] = True
    rotations = sel[rotations_mask]

    rows = []
    for sig_date, etf in rotations.items():
        pos = prices.index.get_indexer([sig_date], method="pad")[0]
        if pos < 0:
            continue
        exec_pos = pos + execution_lag
        if exec_pos >= len(prices.index):
            continue
        sig_date_eff = prices.index[pos]
        exec_date = prices.index[exec_pos]
        sig_px = prices.loc[sig_date_eff, etf]
        exec_px = prices.loc[exec_date, etf]
        rows.append({
            "signal_date": sig_date_eff.date(),
            "exec_date":   exec_date.date(),
            "etf":         etf,
            "signal_px":   sig_px,
            "exec_px":     exec_px,
            "slip_bps":    (exec_px / sig_px - 1) * 10000,
        })
    return pd.DataFrame(rows)


def buy_and_hold(prices: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Helper : buy-and-hold avec poids fixes (sera utilisé comme benchmark)."""
    w_df = pd.DataFrame([weights], index=[prices.index[0]])[UNIVERSE]
    return run_backtest(prices, w_df, tc_bps=0, slip_bps=0)
