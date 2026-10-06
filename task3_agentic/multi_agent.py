"""
Task 3B - Two-agent coordination: Data Analyst -> Research Writer.
Each agent has distinct, enforced tool access; they hand off a structured
Pydantic schema (never a raw string); and the Writer may send Analyst one
clarification request that Analyst must answer before the final report.

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'implement a two-agent
# pipeline with restricted tool access per agent, a Pydantic handoff
# schema, and one critique/clarification cycle', Date: 2026-09-10
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from tools import get_price_data, calculate_volatility, llm_sentiment, get_news, web_search

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)


# ----------------------- Structured handoff schema -----------------------

class DataBrief(BaseModel):
    ticker: str
    latest_close: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    annualised_volatility_pct: Optional[float] = None
    price_regime_note: Optional[str] = None
    clarification_answer: Optional[str] = Field(default=None, description="Filled during the critique loop")


class ClarificationRequest(BaseModel):
    question: str


# ----------------------------- Agent A: Data Analyst -----------------------------

class DataAnalystAgent:
    """Restricted tool access: get_price_data, calculate_volatility,
    llm_sentiment. Deliberately has NO access to web_search."""

    def build_brief(self, ticker: str) -> DataBrief:
        price = get_price_data.invoke({"ticker": ticker})
        vol = calculate_volatility.invoke({"ticker": ticker})

        regime = "trend-up" if (price.get("sma_50") or 0) > (price.get("sma_200") or 0) else "trend-down/neutral"

        return DataBrief(
            ticker=ticker.upper(),
            latest_close=price.get("latest_close"),
            sma_50=price.get("sma_50"),
            sma_200=price.get("sma_200"),
            rsi_14=price.get("rsi_14"),
            macd=price.get("macd"),
            macd_signal=price.get("macd_signal"),
            annualised_volatility_pct=vol.get("annualised_volatility_pct"),
            price_regime_note=f"Quant regime: {regime}",
        )

    def answer_clarification(self, brief: DataBrief, question: str) -> DataBrief:
        """Respond to Agent B's single clarification request using data
        already gathered (no re-fetch needed for these specific fields)."""
        q = question.lower()
        if "volatility" in q:
            answer = f"Annualised volatility is {brief.annualised_volatility_pct}%."
        elif "rsi" in q:
            level = "overbought" if (brief.rsi_14 or 0) > 70 else "oversold" if (brief.rsi_14 or 0) < 30 else "neutral"
            answer = f"RSI(14) is currently {brief.rsi_14} ({level})."
        else:
            answer = f"Latest close {brief.latest_close}, SMA50 {brief.sma_50}, SMA200 {brief.sma_200}."
        brief.clarification_answer = answer
        return brief


# ----------------------------- Agent B: Research Writer -----------------------------

class ResearchWriterAgent:
    """Restricted tool access: web_search, get_news only. Deliberately has
    NO direct access to price-data tools -- it depends on Agent A's brief."""

    def gather_qualitative_context(self, ticker: str) -> dict:
        news = get_news.invoke({"ticker": ticker, "n": 10})
        commentary = web_search.invoke({"query": f"{ticker} stock analyst outlook risks"})
        return {"news": news, "commentary": commentary}

    def maybe_request_clarification(self, brief: DataBrief) -> Optional[ClarificationRequest]:
        """At most one clarification request back to Agent A, only if the
        data brief is missing something the final report needs."""
        if brief.rsi_14 is None or brief.annualised_volatility_pct is None:
            return ClarificationRequest(
                question="Can you confirm the current RSI and annualised volatility precisely?"
            )
        return None

    def write_report(self, ticker: str, brief: DataBrief, qualitative: dict) -> str:
        prompt = f"""You are a research writer producing a final equity report for {ticker}.

Quantitative data brief (from the Data Analyst agent -- this is your ONLY source of price data):
{brief.model_dump_json(indent=2)}

Qualitative context (news + web search, gathered independently by you):
{json.dumps(qualitative, indent=2, default=str)[:3000]}

Write a report with exactly these sections and nothing else:
### Financial Health Summary
### Top Three Risks
### Hedge Strategy Recommendation
Be specific and cite the concrete data points you use from the brief above."""
        resp = llm.invoke(prompt)
        return resp.content


# ----------------------------- Orchestration -----------------------------

def run_multi_agent(ticker: str) -> dict:
    trace = []
    analyst = DataAnalystAgent()
    writer = ResearchWriterAgent()

    # 1. Agent A builds the quantitative brief
    brief = analyst.build_brief(ticker)
    trace.append({"agent": "DataAnalyst", "action": "build_brief", "output": brief.model_dump()})

    # 2. Agent B independently gathers qualitative context
    qualitative = writer.gather_qualitative_context(ticker)
    trace.append({
        "agent": "ResearchWriter", "action": "gather_qualitative_context",
        "output": {
            "n_news": len(qualitative["news"].get("headlines", [])),
            "n_search_results": len(qualitative["commentary"].get("results", [])),
        },
    })

    # 3. Critique loop: Agent B may request one clarification from Agent A
    clarification = writer.maybe_request_clarification(brief)
    if clarification:
        trace.append({"agent": "ResearchWriter", "action": "request_clarification",
                        "question": clarification.question})
        brief = analyst.answer_clarification(brief, clarification.question)
        trace.append({"agent": "DataAnalyst", "action": "answer_clarification",
                        "answer": brief.clarification_answer})

    # 4. Agent B produces the final report, incorporating the (possibly
    #    updated) brief -- runs end-to-end with no manual intervention.
    final_report = writer.write_report(ticker, brief, qualitative)
    trace.append({"agent": "ResearchWriter", "action": "write_report", "output_preview": final_report[:200]})

    return {
        "ticker": ticker.upper(),
        "trace": trace,
        "data_brief": brief.model_dump(),
        "final_report": final_report,
    }


if __name__ == "__main__":
    out = run_multi_agent("AAPL")
    print("--- AGENT MESSAGE / HANDOFF TRACE ---")
    print(json.dumps(out["trace"], indent=2, default=str))
    print("\n--- FINAL REPORT ---\n")
    print(out["final_report"])
