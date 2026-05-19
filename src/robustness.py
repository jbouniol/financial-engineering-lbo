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
# Performance decomposition
# -----------------------------

def contribution_by_etf(weights: pd.DataFrame, etf_returns: pd.DataFrame,
                        first_active: pd.Timestamp | None = None) -> pd.DataFrame:
    """Contribution cumulée au PnL par ETF.

    Pour chaque ETF i : Σ_t w_{i,t} * r_{i,t}. Renvoie la contribution totale
    et le poids moyen en % du temps tenu."""
    if first_active is not None:
        weights = weights.loc[first_active:]
        etf_returns = etf_returns.loc[first_active:]

    contrib = (weights * etf_returns).sum(axis=0)
    time_held = (weights > 0).mean(axis=0) * 100
    return pd.DataFrame({
        "Contribution cumulée (%)": contrib * 100,
        "% du temps tenu":         time_held,
    }).round(2)


def perf_by_regime(net_ret: pd.Series, regime: pd.Series,
                   first_active: pd.Timestamp | None = None) -> pd.DataFrame:
    """Returns annualisé, vol, Sharpe par état du `regime` (Series alignée)."""
    if first_active is not None:
        net_ret = net_ret.loc[first_active:]
        regime = regime.loc[first_active:]

    rows = []
    for label, sub in net_ret.groupby(regime):
        if len(sub) < 2:
            continue
        mean_ann = sub.mean() * ANN * 100
        vol_ann  = sub.std() * np.sqrt(ANN) * 100
        sharpe   = (sub.mean() / sub.std()) * np.sqrt(ANN) if sub.std() > 0 else np.nan
        rows.append({
            "regime":       label,
            "n_days":       len(sub),
            "mean_ann_%":   mean_ann,
            "vol_ann_%":    vol_ann,
            "Sharpe":       sharpe,
            "win_rate_%":   (sub > 0).mean() * 100,
        })
    return pd.DataFrame(rows).set_index("regime").round(2)


def perf_by_period(net_ret: pd.Series,
                   periods: dict[str, tuple[str, str]],
                   turnover: pd.Series | None = None) -> pd.DataFrame:
    """Métriques sur des sous-périodes nommées. `periods` est un dict
    {label: (start, end)} avec dates parsables par pandas.

    Pour chaque sous-période on recalcule l'equity comme `(1+ret).cumprod()`,
    indépendamment des autres périodes."""
    rows = []
    for label, (start, end) in periods.items():
        sl = net_ret.loc[start:end].dropna()
        if len(sl) < 5:
            continue
        eq = (1 + sl).cumprod()
        tn = turnover.loc[start:end] if turnover is not None else None
        m = perf_metrics(sl, eq, turnover=tn)
        rows.append({"period": label, **m})
    return pd.DataFrame(rows).set_index("period")


def yearly_returns(net_ret: pd.Series,
                   first_active: pd.Timestamp | None = None) -> pd.Series:
    """Returns annuels en %, composés depuis le 1er janvier de chaque année."""
    if first_active is not None:
        net_ret = net_ret.loc[first_active:]
    yr = net_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    yr.index = yr.index.year
    return yr.round(2)


# -----------------------------
# CAPM-style factor regression
# -----------------------------

def capm_regression(strat_ret: pd.Series, factor_rets: pd.DataFrame,
                    rf_daily: pd.Series | float = 0.0) -> dict:
    """Régression OLS de l'excess return de la stratégie sur les excess returns
    des facteurs (Newey-West HAC standard errors)."""
    import statsmodels.api as sm

    s = strat_ret.dropna()
    f = factor_rets.dropna(how="all").reindex(s.index).dropna()
    common = s.index.intersection(f.index)
    s = s.loc[common]
    f = f.loc[common]

    if isinstance(rf_daily, pd.Series):
        rf = rf_daily.reindex(common).ffill().fillna(0.0)
    else:
        rf = pd.Series(rf_daily, index=common)

    y = (s - rf).values
    X = sm.add_constant(f.subtract(rf, axis=0).values)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 21})

    names = ["alpha"] + list(f.columns)
    coefs = dict(zip(names, [float(c) for c in model.params]))
    tstats = dict(zip(names, [float(t) for t in model.tvalues]))
    pvals = dict(zip(names, [float(p) for p in model.pvalues]))
    return {
        "coefs":        coefs,
        "tstats":       tstats,
        "pvalues":      pvals,
        "alpha_ann_%":  coefs["alpha"] * ANN * 100,
        "r_squared":    float(model.rsquared),
        "n_obs":        int(len(common)),
    }


def rolling_beta(strat_ret: pd.Series, factor_ret: pd.Series,
                 window_days: int = 252) -> pd.Series:
    """Beta roulant de strat_ret sur factor_ret, fenêtre `window_days`."""
    common = strat_ret.dropna().index.intersection(factor_ret.dropna().index)
    s = strat_ret.loc[common]
    f = factor_ret.loc[common]
    cov = s.rolling(window_days).cov(f)
    var = f.rolling(window_days).var()
    return (cov / var).rename("beta")


def tracking_metrics(strat_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """IR, tracking error, beta, alpha vs un benchmark unique.

    - IR  = excess_return_ann / tracking_error
    - TE  = std(strat - bench) × √252
    - Beta et alpha via OLS naïf (sans HAC ici — pour la version HAC, voir
      `newey_west_alpha` et `capm_regression`)."""
    common = strat_ret.dropna().index.intersection(bench_ret.dropna().index)
    if len(common) < 30:
        return {"error": "trop peu d'observations communes"}
    s = strat_ret.loc[common]
    b = bench_ret.loc[common]
    excess = s - b
    te = excess.std() * np.sqrt(ANN)
    mean_excess = excess.mean() * ANN
    ir = mean_excess / te if te > 0 else np.nan

    cov = np.cov(s.values, b.values, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
    alpha_ann = (s.mean() - beta * b.mean()) * ANN if pd.notna(beta) else np.nan
    return {
        "beta":                float(beta),
        "alpha_ann_%":         float(alpha_ann * 100),
        "tracking_error_%":    float(te * 100),
        "information_ratio":   float(ir),
        "excess_return_ann_%": float(mean_excess * 100),
        "n_obs":               int(len(common)),
    }


# -----------------------------
# 2D sensitivity (V4)
# -----------------------------

def sensitivity_high_lookback(yields: pd.DataFrame, prices: pd.DataFrame,
                              highs: list[float], lookbacks: list[int],
                              threshold_low: float = 0.0,
                              tc_bps: float = 2.0, slip_bps: float = 2.0) -> pd.DataFrame:
    """V4 Sharpe sur grille 2D (threshold_high × lookback_months), threshold_low fixé.

    Permet de visualiser dans un seul tableau la sensibilité aux deux
    paramètres les plus importants de V4."""
    result = pd.DataFrame(index=highs, columns=lookbacks, dtype=float)
    for high in highs:
        for lb in lookbacks:
            w = signal_v4(yields, prices, trend_lookback_months=lb,
                          threshold_low=threshold_low, threshold_high=high)
            bt = run_backtest(prices, w, tc_bps=tc_bps, slip_bps=slip_bps)
            m = perf_metrics(bt["net_ret"], bt["equity"], bt["turnover"], bt["first_active"])
            result.loc[high, lb] = m.get("Sharpe", np.nan)
    return result.astype(float)


# -----------------------------
# Walk-forward V4
# -----------------------------

def walk_forward_v4(yields: pd.DataFrame, prices: pd.DataFrame,
                    train_years: int = 5, test_years: int = 3,
                    threshold_low_grid: list[float] | None = None,
                    threshold_high_grid: list[float] | None = None,
                    lookback_grid: list[int] | None = None,
                    tc_bps: float = 2.0, slip_bps: float = 2.0) -> dict:
    """Walk-forward V4 : refit (low, high, lookback) sur chaque fenêtre train
    glissante (`train_years`), évalue sur la fenêtre test suivante
    (`test_years`), concatène les returns de test. Non-overlapping test windows.

    Renvoie net_ret / equity sur la période OOS concaténée, plus le DataFrame
    des paramètres choisis à chaque pas (utile pour repérer un overfit qui
    sauterait d'un point optimal à l'autre)."""
    if threshold_low_grid is None:
        threshold_low_grid = [-0.5, -0.25, 0.0, 0.25]
    if threshold_high_grid is None:
        threshold_high_grid = [0.5, 0.75, 1.0, 1.25, 1.5]
    if lookback_grid is None:
        lookback_grid = [2, 3, 6, 9, 12]

    start = prices.index[0]
    end   = prices.index[-1]

    windows = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_start >= end:
            break
        if test_end > end:
            test_end = end
        windows.append((train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(years=test_years)

    if not windows:
        return {"error": "historique trop court pour les paramètres train/test"}

    test_ret_parts = []
    param_choices  = []

    for train_start, train_end, test_start, test_end in windows:
        best = None
        for low in threshold_low_grid:
            for high in threshold_high_grid:
                if high <= low:
                    continue
                for lb in lookback_grid:
                    w = signal_v4(yields, prices, trend_lookback_months=lb,
                                  threshold_low=low, threshold_high=high)
                    bt = run_backtest(prices, w, tc_bps=tc_bps, slip_bps=slip_bps)
                    r = bt["net_ret"].loc[train_start:train_end]
                    if len(r) < 60 or r.std() <= 0:
                        continue
                    sh = (r.mean() / r.std()) * np.sqrt(ANN)
                    if best is None or sh > best[3]:
                        best = (low, high, lb, sh)

        if best is None:
            continue
        low, high, lb, train_sh = best

        w_best = signal_v4(yields, prices, trend_lookback_months=lb,
                           threshold_low=low, threshold_high=high)
        bt = run_backtest(prices, w_best, tc_bps=tc_bps, slip_bps=slip_bps)
        test_ret = bt["net_ret"].loc[test_start:test_end]
        test_ret_parts.append(test_ret)

        param_choices.append({
            "train":              f"{train_start.date()}→{train_end.date()}",
            "test":               f"{test_start.date()}→{test_end.date()}",
            "threshold_low":      low,
            "threshold_high":     high,
            "lookback_months":    lb,
            "train_sharpe":       round(float(train_sh), 3),
        })

    if not test_ret_parts:
        return {"error": "aucun test window n'a produit de returns"}

    wf_ret = pd.concat(test_ret_parts).sort_index()
    wf_ret = wf_ret[~wf_ret.index.duplicated(keep="first")]
    wf_equity = (1 + wf_ret).cumprod()

    return {
        "net_ret":   wf_ret,
        "equity":    wf_equity,
        "params":    pd.DataFrame(param_choices),
        "windows":   windows,
    }


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


def jobson_korkie_test(ret_a: pd.Series, ret_b: pd.Series) -> dict:
    """Test de Jobson-Korkie (1981) corrigé par Memmel (2003) pour comparer
    deux Sharpe ratios sur le même échantillon.

    H0 : SR_a = SR_b. Statistique z asymptotiquement normale. Retourne le
    z-stat et le p-value bilatéral."""
    common = ret_a.dropna().index.intersection(ret_b.dropna().index)
    if len(common) < 30:
        return {"error": "trop peu d'observations communes"}
    a = ret_a.loc[common].values
    b = ret_b.loc[common].values

    mu_a, sd_a = a.mean(), a.std(ddof=1)
    mu_b, sd_b = b.mean(), b.std(ddof=1)
    if sd_a <= 0 or sd_b <= 0:
        return {"error": "variance nulle"}

    sr_a_daily = mu_a / sd_a
    sr_b_daily = mu_b / sd_b
    sr_a_ann = sr_a_daily * np.sqrt(ANN)
    sr_b_ann = sr_b_daily * np.sqrt(ANN)

    rho = np.corrcoef(a, b)[0, 1]
    n = len(a)
    theta = 2 * (1 - rho) + 0.5 * (sr_a_daily ** 2 + sr_b_daily ** 2 - 2 * sr_a_daily * sr_b_daily * rho ** 2)
    if theta <= 0:
        return {"error": "theta non positif"}
    z = (sr_a_daily - sr_b_daily) * np.sqrt(n / theta)
    p_two_sided = 2 * (1 - norm.cdf(abs(z)))

    return {
        "SR_a_ann":     float(sr_a_ann),
        "SR_b_ann":     float(sr_b_ann),
        "SR_diff_ann":  float(sr_a_ann - sr_b_ann),
        "rho":          float(rho),
        "z_stat":       float(z),
        "p_value":      float(p_two_sided),
        "n_obs":        int(n),
    }


def adf_test(series: pd.Series, max_lag: int | None = None) -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    H0 : la série a une racine unitaire (non stationnaire).
    Si p < 0.05 → on rejette H0 → la série est probablement stationnaire."""
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    if len(s) < 20:
        return {"error": "trop peu d'observations"}
    result = adfuller(s.values, maxlag=max_lag, autolag="AIC")
    return {
        "adf_stat":       float(result[0]),
        "p_value":        float(result[1]),
        "used_lag":       int(result[2]),
        "n_obs":          int(result[3]),
        "crit_5%":        float(result[4]["5%"]),
        "is_stationary":  bool(result[1] < 0.05),
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
