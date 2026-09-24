"""Run the test questions through the live bot and save the traces.

    python scripts/generate_traces.py                       # v1.0 prompt, default model
    python scripts/generate_traces.py --prompt v1.1         # the fixed prompt
    python scripts/generate_traces.py --model anthropic/claude-haiku-4-5 --out data/my_traces.csv

Needs an API key (see README). The output has the same columns as the
pre-generated data/traces_v1.csv, so every notebook can load it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beefcake.bot import answer  # noqa: E402
from beefcake.traces import save_traces  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "data" / "test_questions.csv"))
    parser.add_argument("--prompt", default="v1.0")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    questions = pd.read_csv(args.questions)
    traces = []
    for _, q in questions.iterrows():
        trace = answer(q["User Query"], query_topic=q["Query Topic"], prompt_version=args.prompt,
                       model=args.model, trace_id=q["Question ID"])
        traces.append(trace)
        print(f"{trace.trace_id}  {q['User Query'][:60]}")

    out = args.out or str(ROOT / "data" / f"my_traces_{args.prompt}.csv")
    save_traces(traces, out)
    print(f"\nSaved {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
