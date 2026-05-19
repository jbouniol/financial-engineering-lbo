"""Tests de robustesse et significance statistique pour la stratégie.

Quatre outils :
1. `sensitivity_grid`  : heatmap Sharpe en fonction des seuils (threshold_low, threshold_high).
2. `sensitivity_lookback` : Sharpe en fonction du lookback du filtre trend V4.
3. `block_bootstrap_sharpe` : IC du Sharpe par block bootstrap (blocs de 21 jours).
4. `probabilistic_sharpe`   : PSR (Bailey & López de Prado 2012) — probabilité que le
                              Sharpe vrai dépasse un benchmark, en tenant compte de la
                              non-normalité des returns.
5. `newey_west_alpha`       : t-stat HAC de l'excess return vs benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew as sp_skew, kurtosis as sp_kurt

from .backtest import (
    signal_v1, signal_v4, run_backtest, perf_metrics, UNIVERSE, ANN,
)


# -----------------------------
# Sensitivity tests
# -----------------------------

def sensitivity_grid(yields: pd.DataFrame, prices: pd.DataFrame,
                     lows: list[float], highs: list[float],
                     tc_bps: float = 2.0, slip_bps: float = 2.0,
                     use_v4: bool = False, trend_lookback_months: int = 3) -> pd.DataFrame:
    """Calcule le Sharpe pour chaque combinaison (low, high) du signal.

    `use_v4=True` rajoute le filtre trend par-dessus. Renvoie un DataFrame
    indexé par `lows`, colonnes = `highs`. NaN quand high <= low (combinaison
    illégale)."""
    result = pd.DataFrame(index=lows, columns=highs, dtype=float)
    for low in lows:
        for high in highs:
            if high <= low:
                result.loc[low, high] = np.nan
                continue
            if use_v4:
                w = signal_v4(yields, prices, trend_lookback_months=trend_lookback_months,
                              threshold_low=low, threshold_high=high)
            else:
                w = signal_v1(yields, prices, threshold_low=low, threshold_high=high)
            bt = run_backtest(prices, w, tc_bps=tc_bps, slip_bps=slip_bps)
            m = perf_metrics(bt["net_ret"], bt["equity"], bt["turnover"], bt["first_active"])
            result.loc[low, high] = m.get("Sharpe", np.nan)
    return result


def sensitivity_lookback(yields: pd.DataFrame, prices: pd.DataFrame,
                         lookbacks_months: list[int],
                         tc_bps: float = 2.0, slip_bps: float = 2.0) -> pd.DataFrame:
    """Sharpe / CAGR / MaxDD de V4 en fonction du lookback du filtre trend."""
    rows = []
    for lb in lookbacks_months:
        w = signal_v4(yields, prices, trend_lookback_months=lb)
        bt = run_backtest(prices, w, tc_bps=tc_bps, slip_bps=slip_bps)
        m = perf_metrics(bt["net_ret"], bt["equity"], bt["turnover"], bt["first_active"])
        rows.append({"lookback_months": lb, **m})
    return pd.DataFrame(rows).set_index("lookback_months")


# -----------------------------
# Bootstrap & statistical inference
# -----------------------------

def block_bootstrap_sharpe(returns: pd.Series, n_iter: int = 1000,
                           block_size: int = 21, seed: int = 42) -> dict:
    """IC du Sharpe par circular block bootstrap.

    Préserve l'autocorrélation locale en rééchantillonnant des blocs contigus
    de `block_size` jours. Retourne mean, median, IC 95%, p-value vs zéro."""
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)
    if n < block_size * 2:
        return {"error": "série trop courte"}

    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        mu = sampled.mean()
        sd = sampled.std(ddof=1)
        sharpes[i] = (mu / sd) * np.sqrt(ANN) if sd > 0 else 0.0

    obs_sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(ANN) if r.std(ddof=1) > 0 else 0.0
    return {
        "observed_sharpe": float(obs_sharpe),
        "bootstrap_mean":  float(sharpes.mean()),
        "bootstrap_median": float(np.median(sharpes)),
        "ci_2.5":  float(np.percentile(sharpes, 2.5)),
        "ci_97.5": float(np.percentile(sharpes, 97.5)),
        "p_value_vs_zero": float((sharpes <= 0).mean()),
    }


def probabilistic_sharpe(returns: pd.Series, sr_benchmark_ann: float = 0.0) -> dict:
    """PSR (Bailey & López de Prado 2012).

    Renvoie la probabilité que le Sharpe vrai > `sr_benchmark_ann` étant
    donné le Sharpe observé et la non-normalité empirique des returns."""
    r = returns.dropna().values
    n = len(r)
    if n < 3:
        return {"error": "série trop courte"}

    sd = r.std(ddof=1)
    if sd == 0:
        return {"error": "vol nulle"}
    sr_daily = r.mean() / sd
    bench_daily = sr_benchmark_ann / np.sqrt(ANN)
    gamma3 = sp_skew(r, bias=False)
    gamma4 = sp_kurt(r, fisher=False, bias=False)  # raw kurt (normal=3)

    denom = np.sqrt(1 - gamma3 * sr_daily + (gamma4 - 1) / 4 * sr_daily ** 2)
    if denom <= 0:
        return {"error": "denom non positif"}
    z = (sr_daily - bench_daily) * np.sqrt(n - 1) / denom
    psr = norm.cdf(z)
    return {
        "observed_sharpe_ann": float(sr_daily * np.sqrt(ANN)),
        "benchmark_sharpe_ann": float(sr_benchmark_ann),
        "skew": float(gamma3),
        "kurt": float(gamma4),
        "psr": float(psr),
        "z_stat": float(z),
    }


def newey_west_alpha(strat_ret: pd.Series, bench_ret: pd.Series,
                     maxlags: int = 21) -> dict:
    """t-stat HAC (Newey-West) de l'excess return moyen sur benchmark.

    Régresse (strat - bench) sur une constante avec covariance HAC ; teste
    H0: alpha = 0. Annualise l'alpha en multipliant par 252."""
    import statsmodels.api as sm

    s = strat_ret.dropna()
    b = bench_ret.dropna()
    common = s.index.intersection(b.index)
    excess = (s.loc[common] - b.loc[common])
    if len(excess) < maxlags * 2:
        return {"error": "série trop courte"}

    X = np.ones((len(excess), 1))
    model = sm.OLS(excess.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    alpha_daily = float(model.params[0])
    return {
        "alpha_daily":   alpha_daily,
        "alpha_ann_%":   alpha_daily * ANN * 100,
        "t_stat":        float(model.tvalues[0]),
        "p_value":       float(model.pvalues[0]),
        "n_obs":         int(len(excess)),
    }
