"""A small regression eval suite you could run in CI on every change (Module 6).

    python scripts/run_ci_evals.py            # checks the stored traces (no API key needed)
    python scripts/run_ci_evals.py --live     # regenerates the v1 answers and v2 tool calls first (needs a key)

It exits with code 1 if any gate fails, which is what makes a CI job go red.
Mostly code checks, because judges cost money on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beefcake import checks  # noqa: E402
from beefcake.agents import load_agent_traces  # noqa: E402
from beefcake.traces import load_traces  # noqa: E402

SUITE_THRESHOLD = 0.90
# Regression checks protect things that already work, so they need (nearly) 100%.
PER_CHECK = {
    "no RAG leak": 1.0,
    "warranty matches policy": 1.0,
    "only approved tools": 1.0,
    "no false success": 1.0,
}
STORED_TRACES = {"v1.0": "traces_v1.csv", "v1.1": "traces_v1.1.csv"}


def run(live: bool = False, prompt_version: str = "v1.1") -> dict:
    rows = []
    if live:
        from beefcake.bot import answer
        questions = pd.read_csv(ROOT / "data" / "test_questions.csv")
        answers = [answer(q, prompt_version=prompt_version).ai_response for q in questions["User Query"]]
        v1 = pd.DataFrame({"Trace ID": questions["Question ID"], "AI Response": answers})
    else:
        v1 = load_traces(STORED_TRACES[prompt_version])
    for _, r in v1.iterrows():
        rows.append({"case": r["Trace ID"], "check": "no RAG leak", "passed": checks.no_rag_leak(r["AI Response"])})
        rows.append({"case": r["Trace ID"], "check": "warranty matches policy",
                     "passed": checks.warranty_matches_policy(r["AI Response"])})
    if live:  # the stored Exercise 7 traces fail on purpose, so check fresh ones
        from beefcake.agents import answer_v2
        expected = pd.read_csv(ROOT / "data" / "exercise7_expected.csv")
        tool_traces = [answer_v2(q, trace_id=tid) for tid, q in zip(expected["Trace ID"], expected["User Query"])]
    else:
        tool_traces = load_agent_traces("exercise7_tool_calls.jsonl")
    for t in tool_traces:
        rows.append({"case": t.trace_id, "check": "only approved tools", "passed": checks.only_approved_tools(t)})
        rows.append({"case": t.trace_id, "check": "no false success", "passed": checks.no_false_success(t)})
    results = pd.DataFrame(rows)
    report = checks.gate(results, SUITE_THRESHOLD, PER_CHECK)
    report["failures"] = results[~results["passed"]][["check", "case"]].to_dict("records")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prompt", default="v1.1", choices=sorted(STORED_TRACES))
    args = parser.parse_args()
    report = run(args.live, args.prompt)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ship it"] else 1)


if __name__ == "__main__":
    main()
