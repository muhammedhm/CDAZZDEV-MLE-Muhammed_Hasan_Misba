"""
Task 1B - LLM Sentiment and Signal Reasoning
CDAZZDEV Senior MLE Assessment

Uses Groq's free inference API to:
  1. Score each news headline for sentiment (structured, Pydantic-validated).
  2. Produce a reasoned Buy/Hold/Sell signal from the technical indicators,
     reasoning over their COMBINATION rather than restating each value.

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'wrap Groq chat
# completions with Pydantic-validated structured outputs, retry logic,
# and clean prompt/business-logic separation', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Literal, Optional

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("llm_reasoning")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_RETRIES = 3

client = Groq(api_key=os.environ["GROQ_API_KEY"])  # set via env var, never hardcoded


# ----------------- Prompt templates (kept separate from business logic) -----------------

SENTIMENT_SYSTEM_PROMPT = """You are a financial news sentiment classifier. \
Given a single news headline, respond with ONLY a JSON object matching this schema:
{"sentiment": "positive" | "negative" | "neutral", "confidence": <float 0-1>, "brief_reason": "<one sentence>"}
Do not include any text outside the JSON object."""

SIGNAL_SYSTEM_PROMPT = """You are a senior equity analyst. You will be given a set of \
technical indicators for a stock. Reason over how the indicators interact with EACH OTHER \
(trend from the moving average relationship, momentum from RSI, the MACD/signal-line \
crossover state, and price position within the Bollinger Bands) to produce a single \
trading signal. Do not just restate each indicator's value in isolation -- explain what \
their COMBINATION implies about the stock's likely near-term direction.
Respond with ONLY a JSON object matching this schema:
{"signal": "Buy" | "Hold" | "Sell", "justification": "<3 to 5 sentences>"}"""


# ----------------------------- Schemas -----------------------------

class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str


class TradingSignal(BaseModel):
    signal: Literal["Buy", "Hold", "Sell"]
    justification: str


# ----------------------------- Core helpers -----------------------------

def _call_groq_json(system_prompt: str, user_prompt: str) -> dict:
    """Call Groq chat completions and parse a JSON object from the
    response, retrying on transient errors or malformed JSON."""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:  # network error OR json.JSONDecodeError
            last_error = exc
            logger.warning("Groq call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Groq call failed after {MAX_RETRIES} attempts: {last_error}")


def score_headline(headline: str) -> Optional[HeadlineSentiment]:
    try:
        data = _call_groq_json(SENTIMENT_SYSTEM_PROMPT, f"Headline: {headline}")
        data["headline"] = headline
        return HeadlineSentiment(**data)
    except (ValidationError, RuntimeError) as exc:
        logger.error("Validation/schema failure for headline %r: %s", headline, exc)
        return None


def score_all_headlines(headlines: list[dict]) -> dict:
    """Score every headline; aggregate into an overall sentiment score in
    [-1, 1], confidence-weighted."""
    scored: list[HeadlineSentiment] = []
    for item in headlines:
        result = score_headline(item["headline"])
        if result is not None:
            scored.append(result)

    if not scored:
        return {"per_headline": [], "overall_sentiment_score": 0.0,
                 "n_scored": 0, "n_failed": len(headlines)}

    polarity = {"positive": 1, "neutral": 0, "negative": -1}
    weighted_sum = sum(polarity[s.sentiment] * s.confidence for s in scored)
    total_conf = sum(s.confidence for s in scored) or 1.0
    overall = weighted_sum / total_conf

    return {
        "per_headline": [s.model_dump() for s in scored],
        "overall_sentiment_score": round(overall, 4),
        "n_scored": len(scored),
        "n_failed": len(headlines) - len(scored),
    }


def generate_signal(indicators: dict) -> Optional[TradingSignal]:
    """`indicators` is the latest row of technical indicators, e.g.
    {"close": .., "sma_50": .., "sma_200": .., "rsi_14": .., "macd": ..,
     "macd_signal": .., "bb_upper": .., "bb_lower": .., "bb_mid": ..}"""
    user_prompt = "Latest technical indicators:\n" + json.dumps(indicators, default=str, indent=2)
    try:
        data = _call_groq_json(SIGNAL_SYSTEM_PROMPT, user_prompt)
        return TradingSignal(**data)
    except (ValidationError, RuntimeError) as exc:
        logger.error("Validation/schema failure for signal generation: %s", exc)
        return None


def run_reasoning(pipeline_output: dict) -> dict:
    """Task 1B end-to-end entry point. Takes the dict returned by
    data_pipeline.run_pipeline() and returns sentiment + signal results."""
    sentiment_result = score_all_headlines(pipeline_output["news"])

    latest_row = pipeline_output["ohlcv_with_indicators"].iloc[-1]
    indicator_snapshot = {
        "close": latest_row.get("close"),
        "sma_50": latest_row.get("sma_50"),
        "sma_200": latest_row.get("sma_200"),
        "rsi_14": latest_row.get("rsi_14"),
        "macd": latest_row.get("macd"),
        "macd_signal": latest_row.get("signal"),
        "bb_upper": latest_row.get("bb_upper"),
        "bb_mid": latest_row.get("bb_mid"),
        "bb_lower": latest_row.get("bb_lower"),
    }
    signal_result = generate_signal(indicator_snapshot)

    return {
        "sentiment": sentiment_result,
        "signal": signal_result.model_dump() if signal_result else None,
    }


if __name__ == "__main__":
    from data_pipeline import run_pipeline

    pipeline_out = run_pipeline("AAPL")
    reasoning_out = run_reasoning(pipeline_out)
    print(json.dumps(reasoning_out, indent=2, default=str))
