"""
Task 2 Bonus - RAG Fallback Layer
CDAZZDEV Senior MLE Assessment

When the fine-tuned model's self-rated confidence on a classification
falls below a threshold, retrieve the most similar labeled training
clauses from a local ChromaDB vector store and re-query with that context
appended, then compare the before/after outputs.

Confidence signal: LLM SELF-RATING rather than perplexity. Perplexity
would require pulling raw per-token logits out of the fine-tuned model's
forward pass and is architecture-specific to wire up; self-rating reuses
the exact same generate() call path already used for inference, which
keeps this fallback simple to reason about and to swap onto a different
base model later.

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'implement a RAG
# fallback that retrieves similar labeled examples from ChromaDB when a
# fine-tuned classifier self-rates low confidence, and re-queries with
# that context', Date: 2026-10-06
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("rag_fallback")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIDENCE_THRESHOLD = 0.6  # below this self-rated confidence, trigger retrieval
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")


def build_vector_store(train_jsonl_path: str, collection_name: str = "risk_clauses"):
    """Builds (or loads) a persistent Chroma collection from the training
    domain documents -- the clauses + labels generated in dataset_generation.py."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    if collection.count() > 0:
        logger.info("Reusing existing collection with %d documents.", collection.count())
        return collection

    docs, metadatas, ids = [], [], []
    with open(train_jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            messages = record["messages"]
            clause = next(m["content"] for m in messages if m["role"] == "user")
            label = next(m["content"] for m in messages if m["role"] == "assistant")
            docs.append(clause)
            metadatas.append({"label": label})
            ids.append(f"train_{i}")

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    logger.info("Indexed %d training clauses into ChromaDB.", len(docs))
    return collection


def retrieve_similar(collection, query_clause: str, k: int = 3) -> list[dict]:
    results = collection.query(query_texts=[query_clause], n_results=k)
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"clause": doc, "label": meta["label"]})
    return retrieved


def self_rate_confidence(model_generate_fn, clause: str, prediction: str) -> float:
    """`model_generate_fn` is the same generate callable used for normal
    inference (e.g. a wrapper around the fine-tuned model's .generate()).
    Asks the model to self-rate how confident it is in its own prediction."""
    prompt = (
        f"Clause: {clause}\nYour classification: {prediction}\n"
        "On a scale of 0 to 1, how confident are you this classification is correct "
        "and fully grounded in the clause? Respond with ONLY the number."
    )
    raw = model_generate_fn(prompt)
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except ValueError:
        logger.warning("Could not parse self-rated confidence from %r, defaulting to 0.5", raw)
        return 0.5


def classify_with_rag_fallback(model_generate_fn, collection, clause: str, system_prompt: str) -> dict:
    """End-to-end: classify, self-rate, and retrieve+re-query if confidence
    is below threshold. Returns both the initial and (if triggered) the
    RAG-augmented prediction for a concrete before/after comparison."""
    initial_prediction = model_generate_fn(f"{system_prompt}\n\nClause: {clause}")
    confidence = self_rate_confidence(model_generate_fn, clause, initial_prediction)

    result = {
        "clause": clause,
        "initial_prediction": initial_prediction,
        "confidence": confidence,
        "rag_triggered": confidence < CONFIDENCE_THRESHOLD,
    }

    if result["rag_triggered"]:
        similar = retrieve_similar(collection, clause, k=3)
        context_block = "\n\n".join(
            f"Similar clause: {s['clause']}\nCorrect label: {s['label']}" for s in similar
        )
        augmented_prompt = (
            f"{system_prompt}\n\nHere are similar, correctly labeled examples for reference:\n"
            f"{context_block}\n\nNow classify this clause: {clause}"
        )
        result["retrieved_examples"] = similar
        result["rag_prediction"] = model_generate_fn(augmented_prompt)

    return result


if __name__ == "__main__":
    # Minimal smoke test with a stub generate function (no GPU/model needed
    # to exercise the retrieval plumbing). Swap `fake_generate` for a real
    # call to your fine-tuned model in the notebook.
    def fake_generate(prompt: str) -> str:
        if "confident" in prompt.lower():
            return "0.4"  # force the fallback branch for this demo
        return '{"risk_category": "Operational Risk", "severity": "medium", "explanation": "generic"}'

    train_path = os.path.join(os.path.dirname(__file__), "data", "train.jsonl")
    if os.path.exists(train_path):
        coll = build_vector_store(train_path)
        out = classify_with_rag_fallback(fake_generate, coll, "Our primary data center experienced an outage.", "Classify this risk clause.")
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"No training data at {train_path} yet -- run dataset_generation.py first.")
