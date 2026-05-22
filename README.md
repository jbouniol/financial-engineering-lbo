# Bond ETF Rotation via Yield Curve Regimes

Group project for the *Financial Engineering & Intro to Trading* course.

Team: Jonathan Bouniol, Guillaume RABEAU, Sacha NARDOUX, Florent Negaf, Enzo Natali.

## What the strategy does

Each month we place 100% of capital in a single US Treasury ETF among three: TLT (long bonds, 20+ years), IEF (intermediate, 7-10 years), SHY (short, 1-3 years). The choice depends on the shape of the US yield curve, measured by the spread between the 10-year and 2-year Treasury rates (DGS10 minus DGS2).

When the spread is widely positive (> 1%), the curve is normal and steep: we hold TLT to capture the carry on long duration. When the spread tightens (between 0 and 1%), we move to IEF, less exposed. When the curve inverts (negative spread) — historically a leading recession signal — we move to SHY to avoid holding duration while the Fed hikes short rates.

A second layer looks at the 3-month performance of the ETF picked by the rule. If it's negative, we rotate to SHY until the price stabilizes. This is what rescues the strategy in 2022, where the curve stayed positive while TLT was collapsing.

## How to run the project

Python 3.12 recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pip check
```

Then select the venv as the Jupyter / VSCode kernel and open a notebook.

**The data pipeline is fully autonomous — no manual download required.**

On the first run, the notebook automatically fetches ETF prices from Yahoo Finance and yields from FRED (public CSV endpoint, no API key needed), then caches everything as Parquet files under `data/raw/`. Subsequent runs read from the cache and work fully offline. The cache files are committed to the repository, so cloning the repo is sufficient to run the notebook without any network access.

## Repository layout

```
README.md
requirements.txt
Docs/
  group_project_guidelines.md   full grading rubric and project brief
data/raw/                       parquet data cache (auto-generated on first run)
  etf_prices.parquet
  benchmark_prices.parquet
  fred_yields.parquet
  fred_dgs3mo.parquet
notebooks/
  FINAL_notebook.ipynb          delivered notebook, 12 sections (top-to-bottom reproducible)
  01_EDA.ipynb                  (à créer) — data exploration, ADF test on the spread
  02_backtest_and_strategy.ipynb (à créer) — strategy + V1-V4 iterations, robustness, walk-forward
  03_paper_trading_analysis.ipynb (à créer) — paper trading 2024-2026 vs BH IEF
```

`FINAL_notebook.ipynb` is the primary deliverable — self-contained, no external modules. The three numbered notebooks are work-in-progress scratch notebooks used during development.

## Data sources

| Source | Series | Description |
|---|---|---|
| Yahoo Finance | TLT, IEF, SHY | Main universe (iShares Treasury ETFs) |
| Yahoo Finance | AGG, SPY | Wider benchmarks (Aggregate Bond + S&P 500) |
| FRED | DGS2, DGS10 | US 2Y and 10Y constant maturity yields, signal base |
| FRED | DGS3MO | 3-month T-Bill, risk-free proxy for the adjusted Sharpe |

Coverage: 2003-01 to today. Daily frequency. Yahoo Finance prices adjusted for splits and dividends. FRED via its public CSV endpoint — no API key required.

## Backtest parameters

| Item | Value |
|---|---|
| Universe | TLT, IEF, SHY |
| Long-only | Yes, weights sum to 1 |
| Rebalancing | Monthly, last **trading** day of the month (ETF calendar, not civil) |
| Execution lag | 1 trading day (real trading day, not civil BusinessDay) |
| FRED publication lag | yields shifted by 1 day before signal construction |
| Transaction cost | 2 bps per side |
| Backtest slippage | 2 bps |
| Paper slippage | 5 bps |

## Methodology and tests

The project includes the standard methodological controls expected from a serious backtest:

| Control | Implementation |
|---|---|
| Look-ahead bias | FRED shift + ETF calendar + lag in real trading days |
| Signal stationarity | ADF test on the 2s10s spread (non-stationary, exploited as regimes) |
| Parameter sensitivity | 1D heatmap (V1 thresholds) and 2D heatmap (threshold_high × lookback V4) |
| Walk-forward | Refit `(low, high, lookback)` every 3 years on a rolling 5-year train window |
| Sharpe significance | Block bootstrap + PSR + Newey-West + Jobson-Korkie |
| Multiple benchmarks | BH IEF, BH AGG, 60/40 SPY/IEF, CAPM alpha on (AGG, SPY) |
| Sharpe rf | Computed with rf=0 and with rf=DGS3MO |

## Results

Backtest 2003 to 2026, costs included:

| Version | CAGR | Vol | Sharpe | Calmar | Max DD | Turnover/year |
|---|---:|---:|---:|---:|---:|---:|
| V1 no costs | 5.07% | 11.0% | 0.50 | 0.19 | -26.6% | 0.92 |
| V2 + transaction costs | 5.05% | 11.0% | 0.50 | 0.19 | -26.6% | 0.92 |
| V3 + slippage | 5.03% | 11.0% | 0.50 | 0.19 | -26.6% | 0.92 |
| V4 + trend filter | 4.08% | 8.9% | 0.50 | 0.19 | -21.2% | 3.07 |
| Buy & Hold IEF | 3.38% | 6.8% | 0.52 | 0.14 | -23.9% | — |
| Buy & Hold AGG | 3.06% | 5.2% | 0.61 | 0.17 | -18.4% | — |
| 60/40 SPY/IEF | 8.72% | 10.7% | 0.83 | 0.28 | -31.4% | — |

Paper trading 2024-2026 with V4 and hardened execution (T+2 lag, 5 bps slippage):

| Window | CAGR | Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| In-sample 2003-2023 | 4.39% | 9.3% | 0.51 | -21.2% |
| OOS backtest 2024-2026 | 1.30% | 3.9% | 0.35 | -3.95% |
| OOS paper 2024-2026 | 1.09% | 3.9% | 0.30 | -4.68% |
| BH IEF OOS 2024-2026 | 2.32% | 6.0% | 0.41 | -6.89% |

Significance tests V4 vs BH IEF (full history):

| Test | Result |
|---|---|
| Jobson-Korkie (Sharpe difference) | z = −0.13, p = 0.90 |
| Newey-West (alpha vs IEF) | α = +0.87%/year, t = 0.66, p = 0.51 |
| Probabilistic Sharpe Ratio vs IEF | 45% |
| CAPM regression on (AGG, SPY), rf=DGS3MO | α = +3.51%/year, t = 2.31, **p = 0.021** |
| Walk-forward (refit every 3 years) vs a priori parameters | Sharpe 0.48 vs 0.47 |
