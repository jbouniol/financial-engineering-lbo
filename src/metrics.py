import pandas as pd
import numpy as np
from scipy.stats import norm, skew as sp_skew, kurtosis as sp_kurt

ANN = 252

def max_drawdown(equity):
    peak = equity.cummax()
    return (equity / peak - 1.0).min()

def time_to_recovery_days(equity):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    if dd.min() >= 0:
        return 0
    trough_idx = dd.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    peak_val = equity.loc[peak_idx]
    after = equity.loc[trough_idx:]
    recovered = after[after >= peak_val]
    if recovered.empty:
        return None
    return int(equity.index.get_loc(recovered.index[0]) - equity.index.get_loc(peak_idx))

def perf_metrics(net_ret, equity, turnover=None, first_active=None, rf_daily=0.0):
    if first_active is not None:
        net_ret = net_ret.loc[first_active:]
        equity  = equity.loc[first_active:]
        if turnover is not None:
            turnover = turnover.loc[first_active:]
    n = len(net_ret)
    if n == 0:
        return {}
    if isinstance(rf_daily, pd.Series):
        rf_aligned = rf_daily.reindex(net_ret.index).ffill().fillna(0.0)
    else:
        rf_aligned = pd.Series(rf_daily, index=net_ret.index)
    excess = net_ret - rf_aligned
    years  = n / ANN
    cagr   = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol    = net_ret.std() * np.sqrt(ANN)
    sharpe = (excess.mean() * ANN) / (excess.std() * np.sqrt(ANN)) if excess.std() > 0 else np.nan
    downside_rms = np.sqrt(np.mean(np.minimum(0.0, excess) ** 2))
    sortino = (excess.mean() * ANN) / (downside_rms * np.sqrt(ANN)) if downside_rms > 0 else np.nan
    mdd    = max_drawdown(equity)
    win    = (net_ret > 0).mean()
    calmar = (cagr / abs(mdd)) if mdd < 0 and not np.isnan(cagr) else np.nan
    ttr    = time_to_recovery_days(equity)
    out = {
        "CAGR_%": cagr * 100, "Vol_%": vol * 100, "Sharpe": sharpe,
        "Sortino": sortino, "Calmar": calmar, "MaxDD_%": mdd * 100,
        "TTR_days": ttr, "WinRate_%": win * 100,
    }
    if turnover is not None:
        out["Turnover_ann"] = turnover.sum() / (n / ANN)
    return out

def metrics_table(backtests_dict, rf_daily=0.0):
    metrics = {}
    for name, bt in backtests_dict.items():
        metrics[name] = perf_metrics(
            bt["net_ret"], bt["equity"], 
            bt.get("turnover"), bt.get("first_active"), 
            rf_daily=rf_daily
        )
    return pd.DataFrame(metrics).T.round(2)

def jobson_korkie_test(ret_a, ret_b):
    common = ret_a.dropna().index.intersection(ret_b.dropna().index)
    if len(common) < 30:
        return {"error": "too few common observations"}
    a = ret_a.loc[common].values
    b = ret_b.loc[common].values
    mu_a, sd_a = a.mean(), a.std(ddof=1)
    mu_b, sd_b = b.mean(), b.std(ddof=1)
    if sd_a <= 0 or sd_b <= 0:
        return {"error": "zero variance"}
    sr_a_daily, sr_b_daily = mu_a / sd_a, mu_b / sd_b
    rho = np.corrcoef(a, b)[0, 1]
    n   = len(a)
    theta = 2 * (1 - rho) + 0.5 * (sr_a_daily**2 + sr_b_daily**2 - 2*sr_a_daily*sr_b_daily*rho**2)
    if theta <= 0:
        return {"error": "non-positive theta"}
    z = (sr_a_daily - sr_b_daily) * np.sqrt(n / theta)
    return {"SR_a_ann": float(sr_a_daily * np.sqrt(ANN)),
            "SR_b_ann": float(sr_b_daily * np.sqrt(ANN)),
            "SR_diff_ann": float((sr_a_daily - sr_b_daily) * np.sqrt(ANN)),
            "rho": float(rho), "z_stat": float(z),
            "p_value": float(2 * (1 - norm.cdf(abs(z)))), "n_obs": int(n)}

def adf_test(series, max_lag=None):
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    if len(s) < 20:
        return {"error": "too few observations"}
    result = adfuller(s.values, maxlag=max_lag, autolag="AIC")
    return {"adf_stat": float(result[0]), "p_value": float(result[1]),
            "used_lag": int(result[2]), "n_obs": int(result[3]),
            "crit_5%": float(result[4]["5%"]), "is_stationary": bool(result[1] < 0.05)}

def bootstrap_sharpe_ci(returns, n_iter=1000, block_size=21, seed=42):
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)
    if n < block_size * 2:
        return {"error": "series too short"}
    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        starts  = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        mu, sd  = sampled.mean(), sampled.std(ddof=1)
        sharpes[i] = (mu / sd) * np.sqrt(ANN) if sd > 0 else 0.0
    obs = (r.mean() / r.std(ddof=1)) * np.sqrt(ANN) if r.std(ddof=1) > 0 else 0.0
    return {"observed_sharpe": float(obs), "bootstrap_mean": float(sharpes.mean()),
            "bootstrap_median": float(np.median(sharpes)),
            "ci_2.5": float(np.percentile(sharpes, 2.5)),
            "ci_97.5": float(np.percentile(sharpes, 97.5)),
            "p_value_vs_zero": float((sharpes <= 0).mean())}

