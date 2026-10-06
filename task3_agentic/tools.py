"""
Task 3A - Agent tool implementations shared by the single-agent (3A) and
multi-agent (3B) systems. Also implements the tool-call logging required
by Task 3C (agent_trace.jsonl).

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'implement 5 LangChain-
# compatible financial research tools with graceful error handling and a
# jsonl tool-call logging decorator', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from functools import wraps
from typing import Callable

import numpy as np
from duckduckgo_search import DDGS
from groq import Groq
from langchain_core.tools import tool

# reuse Task 1A's battle-tested fetch/indicator logic instead of duplicating it
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "task1_financial"))
from data_pipeline import fetch_ohlcv, add_all_indicators, fetch_news  # noqa: E402

logger = logging.getLogger("tools")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

TRACE_LOG_PATH = os.environ.get(
    "AGENT_TRACE_PATH",
    os.path.join(os.path.dirname(__file__), "logs", "agent_trace.jsonl"),
)
os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)


def traced(fn: Callable) -> Callable:
    """Decorator: logs every tool call (name, inputs, truncated output,
    wall-clock duration) to agent_trace.jsonl for Task 3C observability."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        error = None
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration_ms = round((time.time() - start) * 1000, 1)
            output_str = json.dumps(result, default=str) if error is None else f"ERROR: {error}"
            record = {
                "tool": fn.__name__,
                "inputs": {"args": [str(a) for a in args], "kwargs": kwargs},
                "output_preview": output_str[:200],
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
            with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    return wrapper


@tool
@traced
def get_price_data(ticker: str, period: str = "1y") -> dict:
    """Fetch OHLCV data with technical indicators for `ticker`. `period` is
    accepted for interface compatibility; internally >=2y is always pulled
    so indicators have proper warm-up. Returns the latest indicator
    snapshot. Returns an 'error' key on failure instead of raising."""
    try:
        df = fetch_ohlcv(ticker, years=2)
        df = add_all_indicators(df)
        latest = df.iloc[-1]

        def safe(v):
            return round(float(v), 4) if v is not None and np.isfinite(v) else None

        return {
            "ticker": ticker.upper(),
            "latest_close": safe(latest.get("close")),
            "sma_50": safe(latest.get("sma_50")),
            "sma_200": safe(latest.get("sma_200")),
            "rsi_14": safe(latest.get("rsi_14")),
            "macd": safe(latest.get("macd")),
            "macd_signal": safe(latest.get("signal")),
        }
    except Exception as exc:
        logger.warning("get_price_data failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


@tool
@traced
def get_news(ticker: str, n: int = 10) -> dict:
    """Retrieve up to `n` recent news headlines for `ticker`."""
    try:
        headlines = fetch_news(ticker, n=n)
        return {"ticker": ticker.upper(), "headlines": headlines}
    except Exception as exc:
        logger.warning("get_news failed for %s: %s", ticker, exc)
        return {"error": str(exc), "headlines": []}


@tool
@traced
def calculate_volatility(ticker: str, window: int = 30) -> dict:
    """Compute annualised historical volatility of daily returns over the
    trailing `window` trading days."""
    try:
        df = fetch_ohlcv(ticker, years=1)
        returns = df["close"].pct_change().dropna().tail(window)
        if len(returns) < 2:
            raise ValueError("Not enough return observations to compute volatility.")
        annualised = float(returns.std() * np.sqrt(252))
        return {"ticker": ticker.upper(), "window": window,
                 "annualised_volatility_pct": round(annualised * 100, 2)}
    except Exception as exc:
        logger.warning("calculate_volatility failed for %s: %s", ticker, exc)
        return {"error": str(exc)}


@tool
@traced
def llm_sentiment(headlines: list[str]) -> dict:
    """Score a list of headlines for aggregate sentiment using Groq."""
    if not headlines:
        return {"overall_sentiment": "neutral", "score": 0.0, "n": 0}
    joined = "\n".join(f"- {h}" for h in headlines[:15])
    system = (
        "You are a financial sentiment analyst. Given a list of headlines, respond ONLY "
        'with JSON: {"overall_sentiment": "positive"|"negative"|"neutral", '
        '"score": <float -1 to 1>, "summary": "<one sentence>"}'
    )
    try:
        resp = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": joined}],
        )
        data = json.loads(resp.choices[0].message.content)
        data["n"] = len(headlines)
        return data
    except Exception as exc:
        logger.warning("llm_sentiment failed: %s", exc)
        return {"error": str(exc), "overall_sentiment": "neutral", "score": 0.0, "n": len(headlines)}


@tool
@traced
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web (DuckDuckGo) for analyst commentary or general context."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return {"query": query, "results": [
            {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")} for r in results
        ]}
    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return {"error": str(exc), "results": []}


ALL_TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
