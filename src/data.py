"""Data layer for the Bond ETF Rotation project.

Fetches:
- TLT / IEF / SHY adjusted prices from Yahoo Finance.
- US Treasury yields DGS2 / DGS10 from FRED (public CSV endpoint, no API key).

Both sources are cached as parquet files under `data/raw/` so the notebooks
stay reproducible when offline.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ETF_TICKERS = ["TLT", "IEF", "SHY"]
FRED_SERIES = ["DGS2", "DGS10"]
DEFAULT_START = "2003-01-01"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def _cache_path(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{name}.parquet"


def load_etf_prices(start: str = DEFAULT_START, refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices for TLT, IEF, SHY. Returns wide DataFrame (date x ticker)."""
    cache = _cache_path("etf_prices")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    raw = yf.download(
        ETF_TICKERS, start=start, auto_adjust=True, progress=False, group_by="column"
    )
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices = prices[ETF_TICKERS].sort_index()
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
    """Daily Treasury yields DGS2, DGS10 from FRED. Returns wide DataFrame in %."""
    cache = _cache_path("fred_yields")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    parts = [_fetch_fred(s, start) for s in FRED_SERIES]
    yields = pd.concat(parts, axis=1).sort_index()
    yields.index.name = "date"
    yields.to_parquet(cache)
    return yields


def load_all(start: str = DEFAULT_START, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Convenience: fetch ETFs + yields and align on the ETF trading calendar."""
    prices = load_etf_prices(start=start, refresh=refresh)
    yields = load_yields(start=start, refresh=refresh)
    yields_on_etf = yields.reindex(prices.index).ffill()
    return {"prices": prices, "yields": yields, "yields_aligned": yields_on_etf}
