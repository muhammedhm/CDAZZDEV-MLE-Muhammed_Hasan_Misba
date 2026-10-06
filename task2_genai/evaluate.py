"""
Task 2C - Evaluation and Baseline Comparison
CDAZZDEV Senior MLE Assessment

Compares the BASE model (Phi-3-mini-4k-instruct with a system prompt, no
fine-tuning) against the FINE-TUNED model on the held-out test set:
  1. ROUGE-L F1, both models, identical test set, presented as a table.
  2. Additional metric: LLM-as-judge (via Groq, structured JSON) rather
     than BERTScore -- see README for why.
  3. Manual hallucination-rate review of >=10 fine-tuned responses.

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'build an evaluation
# script comparing base vs fine-tuned model with ROUGE-L, an LLM-as-judge
# structured scoring pipeline, and a manual hallucination-rate review
# workflow', Date: 2026-10-06
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal, Optional

import pandas as pd
from groq import Groq
from pydantic import BaseModel, ValidationError
from rouge_score import rouge_scorer

logger = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ----------------------------- ROUGE-L comparison -----------------------------

def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def rouge_comparison_table(base_preds: list[str], ft_preds: list[str], references: list[str]) -> pd.DataFrame:
    rows = [
        {"model": "base (Phi-3-mini, prompted, no fine-tuning)", "rouge_l_f1": compute_rouge_l(base_preds, references)},
        {"model": "fine-tuned (Phi-3-mini + QLoRA)", "rouge_l_f1": compute_rouge_l(ft_preds, references)},
    ]
    return pd.DataFrame(rows)


# ----------------------------- LLM-as-judge -----------------------------
# Chosen over BERTScore: BERTScore requires loading a third (BERT-family)
# model into GPU memory alongside the base and fine-tuned models already
# competing for the free T4's 16GB, and token-embedding similarity doesn't
# verify the thing we actually care about here -- whether the output is
# correctly classified AND grounded in the input clause (not just
# semantically similar text). An LLM judge can check both directly.

JUDGE_SYSTEM_PROMPT = """You are grading a financial-risk-clause classifier's output. You will \
be given the input clause, the gold-standard label, and the model's predicted output. Score the \
prediction on a defined rubric and respond with ONLY a JSON object matching this schema:
{"category_correct": true | false, "grounded_in_clause": true | false, "overall_score": <float 0-1>, "notes": "<one sentence>"}
"category_correct" = does the predicted risk_category match the gold label.
"grounded_in_clause" = does the explanation reference something actually present in the clause \
(false if it invents facts/numbers not in the clause, or is generic boilerplate)."""


class JudgeVerdict(BaseModel):
    category_correct: bool
    grounded_in_clause: bool
    overall_score: float
    notes: str


def llm_judge(clause: str, gold_label: dict, prediction: str) -> Optional[JudgeVerdict]:
    user_prompt = (
        f"Clause:\n{clause}\n\nGold label:\n{json.dumps(gold_label)}\n\nModel prediction:\n{prediction}"
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        )
        return JudgeVerdict(**json.loads(resp.choices[0].message.content))
    except (ValidationError, Exception) as exc:
        logger.error("LLM judge failed: %s", exc)
        return None


def judge_all(clauses: list[str], gold_labels: list[dict], predictions: list[str]) -> pd.DataFrame:
    rows = []
    for clause, gold, pred in zip(clauses, gold_labels, predictions):
        verdict = llm_judge(clause, gold, pred)
        rows.append(verdict.model_dump() if verdict else {
            "category_correct": False, "grounded_in_clause": False, "overall_score": 0.0, "notes": "judge call failed",
        })
    df = pd.DataFrame(rows)
    return df


# ----------------------------- Hallucination rate (manual review) -----------------------------

HallucinationLabel = Literal["correct", "partially_correct", "hallucinated"]


def build_manual_review_sheet(clauses: list[str], predictions: list[str], min_n: int = 10) -> pd.DataFrame:
    """Produces a DataFrame for a human reviewer to fill in the `label`
    column by hand (correct / partially_correct / hallucinated), per the
    assessment's explicit requirement that this NOT be another automated
    metric. Print/display this in the notebook and annotate interactively."""
    n = max(min_n, len(clauses))
    rows = [{"clause": c, "prediction": p, "label": ""} for c, p in zip(clauses[:n], predictions[:n])]
    return pd.DataFrame(rows)


def compute_hallucination_rate(reviewed: pd.DataFrame) -> float:
    assert reviewed["label"].isin(["correct", "partially_correct", "hallucinated"]).all(), \
        "Fill in every row's label as correct / partially_correct / hallucinated before calling this."
    n = len(reviewed)
    n_hallucinated = (reviewed["label"] == "hallucinated").sum()
    return round(100 * n_hallucinated / n, 1) if n else 0.0


# ----------------------------- Qualitative analysis -----------------------------

QUALITATIVE_ANALYSIS_TEMPLATE = """\
## Qualitative Analysis (fill in with real examples after running evaluate.py)

**Where fine-tuning improved behaviour:** [Describe 2-3 specific test-set
examples where the base (prompted-only) model either misclassified the
risk category, used a different/incompatible output format, or wrote a
generic, non-grounded explanation -- and the fine-tuned model corrected
this. Quote the clause id/snippet and both outputs side by side.]

**Remaining failure modes:** [Describe where the fine-tuned model still
gets it wrong -- e.g. confusing adjacent categories like Market Risk vs
Macroeconomic Risk, or specific sectors under-represented in the training
sweep. State what additional data (e.g. more examples for the confused
category, adversarial near-boundary clauses) or training strategy (more
epochs, higher LoRA rank, class-balanced sampling) would address each.]
"""


if __name__ == "__main__":
    # Example end-to-end usage once you have a trained model, base model
    # predictions, and the test split -- see task2_notebook.ipynb for the
    # full interactive version with model loading.
    with open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_examples)} test examples. "
          "Run inference with both models in the notebook, then call "
          "rouge_comparison_table(), judge_all(), and "
          "build_manual_review_sheet() with the resulting predictions.")
