"""Reusable code checks, and the CI gate logic (Modules 3, 4, and 6).

A check looks at one trace and returns True if it passes.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .agents import AgentTrace
from .tools import APPROVED_TOOLS, schema_errors

# --- Checks on the answer text (v1 traces) ---------------------------------

def no_rag_leak(answer: str) -> bool:
    return "provided context" not in answer.lower()


WARRANTY_YEARS = {"row": 2, "rower": 2, "bell": 2, "kettlebell": 2,
                  "pulse": 1, "chest strap": 1, "heart rate strap": 1, "heart-rate strap": 1}
YEARS = {"1": 1, "one": 1, "2": 2, "two": 2}


def warranty_matches_policy(answer: str) -> bool:
    """Every 'N-year warranty' must match the policy for the product named closest before it in the same
    sentence (Row and Bell 2 years, Pulse 1). With no product named, 1 or 2 years passes. (Still crude.)"""
    text = answer.lower()
    for mention in re.finditer(r"\b(\d+|one|two|three)[- ]year warranty", text):
        sentence = re.split(r"[.!?]\s", text[:mention.start()])[-1]
        products = re.findall(r"\b(" + "|".join(WARRANTY_YEARS) + r")\b", sentence)
        allowed = {WARRANTY_YEARS[products[-1]]} if products else {1, 2}
        if YEARS.get(mention.group(1)) not in allowed:
            return False
    return True


# --- Checks on tool calls (v2 and v3 traces) --------------------------------

def only_approved_tools(trace: AgentTrace) -> bool:
    return all(s["name"] in APPROVED_TOOLS for s in trace.tool_calls)


def arguments_match_schema(trace: AgentTrace) -> bool:
    return all(not schema_errors(s["name"], s["input"]) for s in trace.tool_calls if s["name"] in APPROVED_TOOLS)


def first_action_tool(trace: AgentTrace) -> str:
    """The first tool call that isn't a doc search, or 'none'."""
    for s in trace.tool_calls:
        if s["name"] != "search_docs":
            return s["name"]
    return "none"


def end_state_matches(trace: AgentTrace, expected_returns: list[dict], expected_tickets: int) -> bool:
    return (sorted(json.dumps(r, sort_keys=True) for r in trace.end_state.get("returns", []))
            == sorted(json.dumps(r, sort_keys=True) for r in expected_returns)
            and trace.end_state.get("tickets", 0) == expected_tickets)


CLAIMS_SUCCESS = re.compile(
    r"\bi(?:'ve| have)?\s+(?:just\s+)?(?:started|created|initiated|issued|processed|submitted)\b[^.!]*\b(?:return|refund)"
    r"|\b(?:return|refund)\b[^.!]*\b(?:has been|was|is now)\s+(?:started|created|initiated|issued|processed)"
)


def no_false_success(trace: AgentTrace) -> bool:
    """FAIL if the bot says it started a return or issued a refund but no return exists in the store.
    Policy statements ("we refund within 10 business days") don't count as claims. Still a heuristic:
    read the traces it flags."""
    claims = CLAIMS_SUCCESS.search(trace.final_answer.lower().replace("’", "'"))  # curly apostrophes too
    return not (claims and not trace.end_state.get("returns"))


def routed_correctly(trace: AgentTrace, expected_agent: str) -> bool:
    """A handoff assertion (v3): did the router send it to the right agent?"""
    handoffs = [s["name"] for s in trace.spans if s["kind"] == "handoff"]
    return handoffs == [f"transfer_to_{expected_agent}"]


# --- Agent metrics (M6.2) ----------------------------------------------------

def trajectory_efficiency(trace: AgentTrace, shortest_path: int) -> float:
    """Shortest number of tool calls that would have worked, divided by the number the agent made.
    None if it made fewer than that: it skipped a step it needed, so it wasn't efficient, just wrong."""
    taken = len(trace.tool_calls)
    if taken < shortest_path:
        return None
    return round(shortest_path / taken, 2) if taken else 1.0


# --- CI gates (M6.11) --------------------------------------------------------

def gate(results: pd.DataFrame, suite_threshold: float, per_check: dict[str, float]) -> dict:
    """results: one row per (case, check) with a boolean 'passed' column and a 'check' column.

    Returns the suite pass rate, each check's pass rate, and whether each gate passes. A suite-wide
    threshold alone can let one check fail every single time (M6.11), so important checks get their own gate.
    """
    suite_rate = float(results["passed"].mean())
    by_check = results.groupby("check")["passed"].mean()
    check_gates = {c: {"pass rate": round(float(by_check.get(c, 0)), 3), "required": t,
                       "gate": "PASS" if by_check.get(c, 0) >= t else "FAIL"} for c, t in per_check.items()}
    return {
        "suite pass rate": round(suite_rate, 3),
        "suite gate": "PASS" if suite_rate >= suite_threshold else "FAIL",
        "per-check gates": check_gates,
        "ship it": suite_rate >= suite_threshold and all(g["gate"] == "PASS" for g in check_gates.values()),
    }
