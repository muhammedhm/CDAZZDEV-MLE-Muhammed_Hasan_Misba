"""
Task 1A - Financial Data Pipeline
CDAZZDEV Senior MLE Assessment

Fetches >=2 years of daily OHLCV data via yfinance, computes technical
indicators from first principles (no TA-Lib), retrieves news headlines,
and builds a clean summary dict for downstream LLM reasoning (Task 1B).

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'scaffold a robust
# yfinance data pipeline with technical indicators computed from first
# principles and graceful handling of missing data', Date: 2026-09-10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_pipeline")

# ---- Named indicator parameters (no magic numbers in the body below) ----
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULT = 2
MIN_HISTORY_YEARS = 2
MIN_NEWS_HEADLINES = 10


class DataFetchError(Exception):
    """Raised when OHLCV data cannot be retrieved for a ticker."""


@dataclass
class EquitySummary:
    ticker: str
    current_price: Optional[float]
    week52_high: Optional[float]
    week52_low: Optional[float]
    pe_ratio: Optional[float]
    ytd_return_pct: Optional[float]
    momentum_signal: str
    as_of: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return self.__dict__


def fetch_ohlcv(ticker: str, years: int = MIN_HISTORY_YEARS) -> pd.DataFrame:
    """Fetch >= `years` of daily OHLCV data using a relative period string
    (never a hardcoded calendar date) so the window always stays current.
    Pads by 1 extra year so 200-day SMA has warm-up data."""
    period = f"{max(years, MIN_HISTORY_YEARS) + 1}y"
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception as exc:
        raise DataFetchError(f"Failed to fetch OHLCV for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise DataFetchError(f"No OHLCV data returned for ticker '{ticker}'.")

    df = df.rename(columns=str.lower)
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df


# ---------------------------- Indicators -----------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI, computed from first principles (Wilder smoothing via
    an EWM with alpha = 1/period, which is mathematically equivalent)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    rsi_val = rsi_val.fillna(100)  # avg_loss == 0 -> maximum strength (100)
    return rsi_val


def macd(series: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW,
          signal: int = MACD_SIGNAL) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def bollinger_bands(series: pd.Series, window: int = BOLLINGER_WINDOW,
                      num_std: int = BOLLINGER_STD_MULT) -> pd.DataFrame:
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma_50"] = sma(out["close"], SMA_SHORT_WINDOW)
    out["sma_200"] = sma(out["close"], SMA_LONG_WINDOW)
    out["rsi_14"] = rsi(out["close"], RSI_PERIOD)
    out = out.join(macd(out["close"]))
    out = out.join(bollinger_bands(out["close"]))
    return out


def momentum_signal_from_indicators(latest: pd.Series) -> str:
    """Cheap, explainable momentum heuristic used only as a pipeline
    *feature*. The reasoned Buy/Hold/Sell call is produced by the LLM in
    Task 1B, not here."""
    score = 0
    if pd.notna(latest.get("sma_50")) and pd.notna(latest.get("sma_200")):
        score += 1 if latest["sma_50"] > latest["sma_200"] else -1
    if pd.notna(latest.get("rsi_14")):
        if latest["rsi_14"] > 70:
            score -= 1
        elif latest["rsi_14"] < 30:
            score += 1
    if pd.notna(latest.get("macd")) and pd.notna(latest.get("signal")):
        score += 1 if latest["macd"] > latest["signal"] else -1

    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"


# ----------------------------- News -----------------------------

def fetch_news(ticker: str, n: int = MIN_NEWS_HEADLINES) -> list[dict]:
    """Retrieve recent headlines via the yfinance news endpoint. Returns an
    empty, well-formed list on failure rather than raising, so downstream
    stages degrade gracefully."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", ticker, exc)
        return []

    headlines = []
    for item in raw[: max(n, MIN_NEWS_HEADLINES)]:
        # yfinance's news payload schema has shifted between versions;
        # handle both the newer nested "content" shape and the older flat one.
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if not title:
            continue
        provider = content.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else item.get("publisher", "unknown")
        canonical = content.get("canonicalUrl")
        link = canonical.get("url") if isinstance(canonical, dict) else item.get("link", "")
        headlines.append({
            "headline": title,
            "publisher": publisher or "unknown",
            "published": str(content.get("pubDate") or item.get("providerPublishTime", "")),
            "link": link or "",
        })

    if len(headlines) < MIN_NEWS_HEADLINES:
        logger.warning("Only %d headlines retrieved for %s (target %d).",
                        len(headlines), ticker, MIN_NEWS_HEADLINES)
    return headlines


# ----------------------------- Summary -----------------------------

def build_summary(ticker: str, df: pd.DataFrame) -> EquitySummary:
    latest = df.iloc[-1]
    one_year_ago = df.index[-1] - pd.Timedelta(days=365)
    window_1y = df[df.index >= one_year_ago]
    week52_high = window_1y["high"].max() if not window_1y.empty else np.nan
    week52_low = window_1y["low"].min() if not window_1y.empty else np.nan

    try:
        info = yf.Ticker(ticker).info
        pe_ratio = info.get("trailingPE")
    except Exception as exc:
        logger.warning("Could not retrieve P/E for %s: %s", ticker, exc)
        pe_ratio = None

    ytd_rows = df[df.index.year == df.index[-1].year]
    if not ytd_rows.empty:
        ytd_return = (latest["close"] / ytd_rows.iloc[0]["close"] - 1) * 100
    else:
        ytd_return = None

    return EquitySummary(
        ticker=ticker.upper(),
        current_price=round(float(latest["close"]), 2) if pd.notna(latest["close"]) else None,
        week52_high=round(float(week52_high), 2) if pd.notna(week52_high) else None,
        week52_low=round(float(week52_low), 2) if pd.notna(week52_low) else None,
        pe_ratio=round(float(pe_ratio), 2) if pe_ratio else None,
        ytd_return_pct=round(float(ytd_return), 2) if ytd_return is not None and pd.notna(ytd_return) else None,
        momentum_signal=momentum_signal_from_indicators(latest),
    )


def run_pipeline(ticker: str) -> dict:
    """End-to-end Task 1A entry point."""
    logger.info("Fetching OHLCV for %s", ticker)
    df = fetch_ohlcv(ticker)
    df = add_all_indicators(df)
    summary = build_summary(ticker, df)
    news = fetch_news(ticker)
    return {
        "ticker": ticker.upper(),
        "ohlcv_with_indicators": df,
        "summary": summary.to_dict(),
        "news": news,
    }


if __name__ == "__main__":
    result = run_pipeline("AAPL")
    print(result["summary"])
    print(f"Retrieved {len(result['news'])} headlines")
    print(result["ohlcv_with_indicators"].tail())
