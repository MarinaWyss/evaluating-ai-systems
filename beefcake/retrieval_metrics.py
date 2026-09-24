"""Retrieval metrics (Module 4).

Each function takes the ranked list of retrieved section IDs for one question and
the set of section IDs a person labeled as relevant. Average them over questions.

A section counts once, at its first position: when long sections are split into
smaller chunks, several hits can map to the same section.
"""

from __future__ import annotations

import math


def _unique(retrieved: list[str]) -> list[str]:
    return list(dict.fromkeys(retrieved))


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1 if at least one relevant section is in the top k, else 0."""
    return float(any(r in relevant for r in _unique(retrieved)[:k]))


def recall(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of all the relevant sections that made the top k."""
    if not relevant:
        return 0.0
    return len(set(_unique(retrieved)[:k]) & relevant) / len(relevant)


def precision(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of the top k that are relevant. If fewer than k came back, the missing ones count as not relevant."""
    return sum(r in relevant for r in _unique(retrieved)[:k]) / k if k else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    """1 / rank of the first relevant section (1 for first place, 1/2 for second...). 0 if none."""
    ranked = _unique(retrieved)
    for rank, r in enumerate(ranked[:k] if k else ranked, 1):
        if r in relevant:
            return 1 / rank
    return 0.0


def ndcg(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """Order-aware, with graded relevance, e.g. {"policies#returns": 2, "faq#...": 1}."""
    dcg = sum(relevance.get(r, 0) / math.log2(i + 2) for i, r in enumerate(_unique(retrieved)[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
