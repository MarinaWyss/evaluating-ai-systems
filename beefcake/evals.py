"""Helpers for the eval spreadsheet, rubric agreement, and simple statistics.

Every function here is short on purpose. Read them: there's no magic.
"""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd

PASS, FAIL = "PASS", "FAIL"

FAILURE_MODES = [
    "Invents policy",
    "Doesn't collect info",
    "Assumes device",
    "Made-up steps",
    "Leaks RAG setup",
    "Out of scope",
]


# ---------------------------------------------------------------------------
# The eval spreadsheet: one PASS/FAIL column per failure mode
# ---------------------------------------------------------------------------

def wilson_interval(fails: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a rate like 8 failures out of 25."""
    if n == 0:
        return (0.0, 0.0)
    p = fails / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def failure_rates(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """The COUNTIF row: how many traces FAIL each column, as a count, a rate, and an interval."""
    columns = columns or [c for c in FAILURE_MODES if c in df.columns]
    rows = []
    for col in columns:
        labels = df[col].astype(str).str.strip().str.upper()
        n = int(labels.isin([PASS, FAIL]).sum())
        fails = int((labels == FAIL).sum())
        low, high = wilson_interval(fails, n)
        rows.append({
            "Failure mode": col,
            "Fails": fails,
            "Out of": n,
            "Rate": f"{fails / n:.0%}" if n else "-",
            "95% interval": f"{low:.0%} to {high:.0%}" if n else "-",
        })
    return pd.DataFrame(rows)


def mcnemar_exact(before: pd.Series, after: pd.Series) -> dict:
    """Compare two runs on the SAME questions. Only the questions that flipped count.

    fixed = FAIL before, PASS after.   broke = PASS before, FAIL after.
    Returns the two-sided exact p-value.
    """
    before = before.astype(str).str.upper().reset_index(drop=True)
    after = after.astype(str).str.upper().reset_index(drop=True)
    fixed = int(((before == FAIL) & (after == PASS)).sum())
    broke = int(((before == PASS) & (after == FAIL)).sum())
    n = fixed + broke
    if n == 0:
        p = 1.0
    else:
        k = min(fixed, broke)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        p = min(1.0, 2 * tail)
    return {"fixed": fixed, "broke": broke, "p_value": round(p, 4)}


# ---------------------------------------------------------------------------
# Testing the rubric: do two reviewers agree?
# ---------------------------------------------------------------------------

def raw_agreement(a: pd.Series, b: pd.Series) -> float:
    a, b = _clean(a), _clean(b)
    return float((a == b).mean())


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement corrected for chance. 0 = no better than chance, 1 = perfect."""
    a, b = _clean(a), _clean(b)
    observed = float((a == b).mean())
    labels = set(a) | set(b)
    expected = sum((a == label).mean() * (b == label).mean() for label in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def disagreements(df: pd.DataFrame, col_a: str, col_b: str, show: list[str] | None = None) -> pd.DataFrame:
    """The rows where two reviewers gave different labels. Read every one."""
    show = show or ["Trace ID", "User Query", "AI Response"]
    mask = _clean(df[col_a]) != _clean(df[col_b])
    return df.loc[mask.values, show + [col_a, col_b]]


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Code checks: run a set of functions over every trace
# ---------------------------------------------------------------------------

def run_checks(df: pd.DataFrame, checks: dict[str, Callable[[pd.Series], bool]]) -> pd.DataFrame:
    """Apply each check to each trace. A check returns True if the trace passes.

    Returns a copy of df with one PASS/FAIL column per check.
    """
    out = df.copy()
    for name, check in checks.items():
        out[name] = [PASS if check(row) else FAIL for _, row in df.iterrows()]
    return out


def compare_to_labels(df: pd.DataFrame, check_col: str, label_col: str) -> pd.DataFrame:
    """Where does an automated check disagree with the human label?"""
    return disagreements(df, check_col, label_col)
