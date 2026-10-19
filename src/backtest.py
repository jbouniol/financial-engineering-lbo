import pandas as pd
import numpy as np

UNIVERSE = ["TLT", "IEF", "SHY"]
ANN = 252

def monthly_to_daily_weights(monthly_w, daily_index, execution_lag=1):
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
    return effective.reindex(daily_index, method="ffill").fillna(0.0)

def run_backtest(prices, weights_monthly, tc_bps=0.0, slip_bps=0.0, execution_lag=1):
    prices = prices[UNIVERSE].copy()
    rets = prices.pct_change().fillna(0.0)
    w = monthly_to_daily_weights(weights_monthly, prices.index, execution_lag=execution_lag)
    
    w_lagged = w.shift(1).bfill()
    gross_ret = (w_lagged * rets).sum(axis=1)
    
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    cost_rate = (tc_bps + slip_bps) / 10000.0
    cost = turnover * cost_rate
    net_ret = gross_ret - cost
    
    active_mask = w.sum(axis=1) > 0
    if active_mask.any():
        start = active_mask.idxmax()
        eq = pd.Series(1.0, index=prices.index)
        eq.loc[start:] = (1.0 + net_ret.loc[start:]).cumprod()
    else:
        eq = pd.Series(1.0, index=prices.index)
        
    return {
        "weights": w, "gross_ret": gross_ret, "net_ret": net_ret,
        "turnover": turnover, "cost": cost, "equity": eq,
        "first_active": active_mask.idxmax() if active_mask.any() else None,
    }
