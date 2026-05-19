"""Data layer for the Bond ETF Rotation project.

Fetches :
- TLT / IEF / SHY adjusted prices from Yahoo Finance (univers de la stratégie).
- AGG / SPY adjusted prices from Yahoo Finance (benchmarks élargis).
- US Treasury yields DGS2 / DGS10 from FRED (signal).
- DGS3MO 3-month T-Bill yield from FRED (taux sans risque proxy pour Sharpe).

Tout est cache parquet sous `data/raw/` pour reproductibilité offline.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ETF_TICKERS       = ["TLT", "IEF", "SHY"]
BENCHMARK_TICKERS = ["AGG", "SPY"]
FRED_SERIES       = ["DGS2", "DGS10"]
FRED_RF_SERIES    = "DGS3MO"
DEFAULT_START     = "2003-01-01"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def _cache_path(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{name}.parquet"


def _yf_close(tickers: list[str], start: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column")
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices = prices[tickers].sort_index()
    return prices


def load_etf_prices(start: str = DEFAULT_START, refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices for TLT, IEF, SHY (univers de la stratégie)."""
    cache = _cache_path("etf_prices")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    prices = _yf_close(ETF_TICKERS, start)
    prices.to_parquet(cache)
    return prices


def load_benchmark_prices(start: str = DEFAULT_START, refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices for AGG (US Aggregate Bond) et SPY (S&P 500).
    AGG démarre en 2003-09 ; SPY existe depuis 1993."""
    cache = _cache_path("benchmark_prices")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    prices = _yf_close(BENCHMARK_TICKERS, start)
    prices.to_parquet(cache)
    return prices


def _fetch_fred(series: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", series]
    df["date"] = pd.to_datetime(df["date"])
    df[series] = pd.to_numeric(df[series], errors="coerce")
    return df.set_index("date")[series]


def load_yields(start: str = DEFAULT_START, refresh: bool = False) -> pd.DataFrame:
    """Daily Treasury yields DGS2, DGS10 from FRED (en %)."""
    cache = _cache_path("fred_yields")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    parts = [_fetch_fred(s, start) for s in FRED_SERIES]
    yields = pd.concat(parts, axis=1).sort_index()
    yields.index.name = "date"
    yields.to_parquet(cache)
    return yields


def load_rf(start: str = DEFAULT_START, refresh: bool = False) -> pd.Series:
    """3-month T-Bill yield (DGS3MO) en %, proxy du taux sans risque journalier.

    Converti en taux daily (en décimal) pour usage direct dans `perf_metrics` :
    rf_daily = (DGS3MO / 100) / 252."""
    cache = _cache_path("fred_dgs3mo")
    if cache.exists() and not refresh:
        s = pd.read_parquet(cache).iloc[:, 0]
    else:
        s = _fetch_fred(FRED_RF_SERIES, start)
        s.to_frame(FRED_RF_SERIES).to_parquet(cache)
    return s


def load_all(start: str = DEFAULT_START, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Convenience : fetch ETFs + yields + RF + benchmarks, aligne sur le
    calendrier ETF (forward-fill pour les valeurs FRED publiées en t)."""
    prices     = load_etf_prices(start=start, refresh=refresh)
    yields     = load_yields(start=start, refresh=refresh)
    rf_pct     = load_rf(start=start, refresh=refresh)
    benchmarks = load_benchmark_prices(start=start, refresh=refresh)

    yields_on_etf = yields.reindex(prices.index).ffill()
    rf_on_etf     = rf_pct.reindex(prices.index).ffill()
    rf_daily      = (rf_on_etf / 100.0) / 252.0
    bench_on_etf  = benchmarks.reindex(prices.index).ffill()

    return {
        "prices":          prices,
        "yields":          yields,
        "yields_aligned":  yields_on_etf,
        "rf_pct":          rf_on_etf,
        "rf_daily":        rf_daily,
        "benchmarks":      bench_on_etf,
    }
