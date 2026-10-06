"""
Task 3C - Short-term memory, persistent cache, and observability.

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'add in-session memory,
# a date/ticker-keyed persistent disk cache, and jsonl tool-call log
# reporting on top of the multi-agent pipeline', Date: 2026-09-10
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Optional

from multi_agent import run_multi_agent

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRACE_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "agent_trace.jsonl")


class ResearchSession:
    """Holds short-term memory (in-process dict) across multiple turns in a
    single run, plus a persistent, date/ticker-keyed disk cache that
    survives process restarts."""

    def __init__(self):
        self._memory: dict[str, dict] = {}  # ticker -> last research result, this session

    # ---------------- persistent cache ----------------

    @staticmethod
    def _cache_path(ticker: str, as_of: Optional[str] = None) -> str:
        as_of = as_of or date.today().isoformat()
        return os.path.join(CACHE_DIR, f"{ticker.upper()}_{as_of}.json")

    def _load_from_disk(self, ticker: str) -> Optional[dict]:
        path = self._cache_path(ticker)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_to_disk(self, ticker: str, brief: dict) -> None:
        with open(self._cache_path(ticker), "w", encoding="utf-8") as f:
            json.dump(brief, f, indent=2, default=str, ensure_ascii=False)

    # ---------------- public API ----------------

    def research(self, ticker: str, force_refresh: bool = False) -> dict:
        ticker = ticker.upper()

        # 1. Short-term memory: answer from context if already researched
        #    earlier THIS session, with zero tool calls.
        if not force_refresh and ticker in self._memory:
            return {**self._memory[ticker], "source": "short_term_memory"}

        # 2. Persistent cache: detect today's cached brief on disk (e.g. a
        #    fresh process / new session object) and load it instead of
        #    re-running the full tool pipeline.
        if not force_refresh:
            cached = self._load_from_disk(ticker)
            if cached is not None:
                self._memory[ticker] = cached
                return {**cached, "source": "persistent_cache"}

        # 3. Cold run: execute the full multi-agent pipeline (this is what
        #    populates agent_trace.jsonl with real tool calls).
        result = run_multi_agent(ticker)
        result["source"] = "live_run"
        self._memory[ticker] = result
        self._save_to_disk(ticker, result)
        return result

    def ask_followup(self, ticker: str, question: str) -> str:
        """Answer a follow-up question using the session-cached brief for
        `ticker` WITHOUT re-invoking any tools -- demonstrates short-term
        memory reuse within a session."""
        ticker = ticker.upper()
        if ticker not in self._memory:
            raise ValueError(f"No prior research for {ticker} in this session -- call research() first.")
        brief = self._memory[ticker]["data_brief"]
        return (
            f"(answered from session memory, no tool calls) For {ticker}: "
            f"RSI(14)={brief.get('rsi_14')}, volatility={brief.get('annualised_volatility_pct')}%, "
            f"in response to: '{question}'"
        )


def print_trace_log_summary() -> None:
    if not os.path.exists(TRACE_LOG_PATH):
        print("No trace log yet -- run research() at least once first.")
        return
    with open(TRACE_LOG_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"{len(records)} tool calls logged to {TRACE_LOG_PATH}")
    for rec in records[-10:]:
        print(f"  {rec['tool']:<22} {rec['duration_ms']:>7.1f} ms  {rec['output_preview'][:80]}")


if __name__ == "__main__":
    session = ResearchSession()

    print("=== Run 1: cold (executes tools, populates agent_trace.jsonl) ===")
    r1 = session.research("AAPL")
    print("source:", r1["source"])

    print("\n=== Run 2: same ticker, same session (short-term memory, no tool calls) ===")
    r2 = session.research("AAPL")
    print("source:", r2["source"])

    print("\n=== Follow-up question, answered from memory, no tool calls ===")
    print(session.ask_followup("AAPL", "How volatile has it been?"))

    print("\n=== New session object, same day (should hit persistent disk cache) ===")
    session2 = ResearchSession()
    r3 = session2.research("AAPL")
    print("source:", r3["source"])

    print("\n=== agent_trace.jsonl summary ===")
    print_trace_log_summary()
