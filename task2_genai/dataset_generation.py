"""
Task 2A - Use Case Definition and Dataset Engineering
CDAZZDEV Senior MLE Assessment

Generates synthetic (clause, label) training examples for the SEC-style
risk-factor classification task (see problem_statement.md) using Groq
Llama-3.3-70B as a TEACHER model. The teacher is deliberately a different,
larger model than the STUDENT model fine-tuned in fine_tune.py
(microsoft/Phi-3-mini-4k-instruct), per the assessment's explicit rule
against using the same model for both roles.

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'generate a diverse
# synthetic dataset for a financial risk-clause classification task using
# a Groq teacher model, with diversity metrics and an 80/10/10 JSONL
# split', Date: 2026-10-06
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import Counter
from typing import Literal, Optional

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("dataset_generation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TEACHER_MODEL = os.environ.get("GROQ_TEACHER_MODEL", "openai/gpt-oss-120b")
STUDENT_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # fine-tuned in fine_tune.py -- must differ from TEACHER_MODEL
MAX_RETRIES = 3
MIN_EXAMPLES = 100

client = Groq(api_key=os.environ["GROQ_API_KEY"])

RISK_CATEGORIES = [
    "Market Risk", "Credit Risk", "Liquidity Risk", "Operational Risk",
    "Regulatory/Legal Risk", "Cybersecurity Risk", "Reputational Risk",
    "Macroeconomic Risk",
]
SECTORS = [
    "consumer retail", "commercial banking", "biotechnology", "oil & gas",
    "semiconductor manufacturing", "enterprise SaaS", "telecommunications",
    "commercial real estate", "airline", "insurance", "renewable energy",
    "automotive manufacturing", "e-commerce logistics",
]
STYLES = [
    "formal 10-K legal drafting style, dense and cautious",
    "slightly more direct investor-letter style, still professional",
    "a short, terse disclosure, 2-3 sentences",
    "a longer disclosure with a specific hypothetical scenario",
]

SYSTEM_CLASSIFICATION_PROMPT = """You are a financial compliance assistant. Given a single \
paragraph excerpted from a company's Risk Factors disclosure, classify it and respond with \
ONLY a JSON object matching this exact schema:
{"risk_category": "<one of: Market Risk, Credit Risk, Liquidity Risk, Operational Risk, \
Regulatory/Legal Risk, Cybersecurity Risk, Reputational Risk, Macroeconomic Risk>", \
"severity": "low" | "medium" | "high", "explanation": "<one sentence that references \
specific wording or facts from the clause itself -- never invent a fact not present in \
the clause>"}
Do not include any text outside the JSON object."""

# Full teacher system prompt used for SYNTHETIC DATA GENERATION (distinct from the
# classification prompt above, which is what the fine-tuned student model will learn
# to follow). Included here in full per the assessment's citation requirements.
TEACHER_GENERATION_PROMPT = """You are helping build a training dataset for a financial \
compliance AI assistant. Generate ONE realistic, original paragraph (50-200 words) that \
could plausibly appear in a public company's "Risk Factors" disclosure (10-K/10-Q style), \
about a company in the {sector} sector, describing a risk that clearly falls under the \
category "{risk_category}". Write it in this style: {style}.

Also provide the correct label for the paragraph you just wrote.

Respond with ONLY a JSON object matching this exact schema:
{{"clause": "<the risk-factor paragraph you wrote>", "label": {{"risk_category": "{risk_category}", \
"severity": "low" | "medium" | "high" (pick whichever is implied by the language you used), \
"explanation": "<one sentence citing specific wording from the clause you wrote>"}}}}
Do not include any text outside the JSON object. Do not reuse company names, numbers, or \
phrasing from any previous example -- make this one distinct."""


class RiskLabel(BaseModel):
    risk_category: Literal[
        "Market Risk", "Credit Risk", "Liquidity Risk", "Operational Risk",
        "Regulatory/Legal Risk", "Cybersecurity Risk", "Reputational Risk",
        "Macroeconomic Risk",
    ]
    severity: Literal["low", "medium", "high"]
    explanation: str


class TrainingExample(BaseModel):
    clause: str
    label: RiskLabel


def _call_teacher_json(prompt: str) -> Optional[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=TEACHER_MODEL,
                temperature=0.9,  # high temperature -> lexical/scenario diversity across calls
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            logger.warning("Teacher call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            time.sleep(1.5 * attempt)
    return None


def generate_example(sector: str, risk_category: str, style: str) -> Optional[TrainingExample]:
    prompt = TEACHER_GENERATION_PROMPT.format(sector=sector, risk_category=risk_category, style=style)
    data = _call_teacher_json(prompt)
    if data is None:
        return None
    try:
        return TrainingExample(**data)
    except ValidationError as exc:
        logger.error("Teacher output failed schema validation, discarding: %s", exc)
        return None


def generate_dataset(min_examples: int = MIN_EXAMPLES, seed: int = 42) -> list[TrainingExample]:
    """Sweep (sector x risk_category x style) combinations so every risk
    category is represented across multiple sectors and writing styles --
    this is what keeps the dataset from collapsing into near-duplicates of
    a single scenario (see diversity metrics below)."""
    rng = random.Random(seed)
    combos = [(s, r, st) for s in SECTORS for r in RISK_CATEGORIES for st in STYLES]
    rng.shuffle(combos)

    examples: list[TrainingExample] = []
    for sector, risk_category, style in combos:
        if len(examples) >= min_examples:
            break
        ex = generate_example(sector, risk_category, style)
        if ex is not None:
            examples.append(ex)
        else:
            logger.warning("Skipped a failed generation for (%s, %s)", sector, risk_category)

    logger.info("Generated %d/%d requested examples.", len(examples), min_examples)
    return examples


# ----------------------------- Diversity metrics -----------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "its", "our", "we", "may", "could", "would", "with", "that", "this", "as",
    "be", "has", "have", "which", "from", "by", "at", "such", "will", "if",
}


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def diversity_report(examples: list[TrainingExample]) -> dict:
    lengths = [len(ex.clause.split()) for ex in examples]
    token_sets = [_tokenize(ex.clause) for ex in examples]

    all_words = Counter(w for ts in token_sets for w in ts)
    top_keywords = all_words.most_common(15)

    category_counts = Counter(ex.label.risk_category for ex in examples)

    # Near-duplicate check: Jaccard similarity between every pair's token sets.
    # Flag pairs above 0.6 as near-duplicates -- a diverse dataset should have
    # very few of these relative to the total example count.
    near_dup_pairs = 0
    n = len(token_sets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard > 0.6:
                near_dup_pairs += 1

    return {
        "n_examples": n,
        "prompt_length_words": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
        "top_keywords": top_keywords,
        "risk_category_distribution": dict(category_counts),
        "near_duplicate_pairs": near_dup_pairs,
        "near_duplicate_pair_ratio": round(near_dup_pairs / max(n * (n - 1) / 2, 1), 4),
    }


# ----------------------------- Chat-template JSONL + split -----------------------------

def to_chat_messages(ex: TrainingExample) -> list[dict]:
    """Generic (role, content) message list. The CORRECT model-specific chat
    template (e.g. Phi-3's <|system|>/<|user|>/<|assistant|> tokens) is
    applied dynamically in fine_tune.py via
    `tokenizer.apply_chat_template(...)`, rather than hardcoded here --
    this keeps the dataset portable across base models and avoids baking
    in a template that silently goes stale if the base model changes."""
    return [
        {"role": "system", "content": SYSTEM_CLASSIFICATION_PROMPT},
        {"role": "user", "content": ex.clause},
        {"role": "assistant", "content": ex.label.model_dump_json()},
    ]


def split_and_save(examples: list[TrainingExample], out_dir: str, seed: int = 42) -> dict:
    """80/10/10 train/val/test split, written as JSONL chat-format files."""
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    os.makedirs(out_dir, exist_ok=True)
    sizes = {}
    for name, subset in splits.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps({"messages": to_chat_messages(ex)}, ensure_ascii=False) + "\n")
        sizes[name] = len(subset)
        logger.info("Wrote %d examples to %s", len(subset), path)

    return sizes


if __name__ == "__main__":
    dataset = generate_dataset(min_examples=MIN_EXAMPLES)
    print("Diversity report:")
    print(json.dumps(diversity_report(dataset), indent=2, default=str))

    sizes = split_and_save(dataset, out_dir=os.path.join(os.path.dirname(__file__), "data"))
    print("Split sizes:", sizes)
