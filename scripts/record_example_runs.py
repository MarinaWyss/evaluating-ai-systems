"""Record real judge runs once, so the Module 5 notebook has results to show without an API key.

    python scripts/record_example_runs.py                     # uses BEEFCAKE_JUDGE_MODEL or the default
    python scripts/record_example_runs.py --model openai/gpt-4o

Writes data/example_runs/ (commit these files):
    judge_assumes_device.csv   our "Assumes device" judge on the Exercise 8 dev and test sets
    swap_test.csv              the Exercise 9 swap test on 20 answer pairs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beefcake import llm  # noqa: E402
from beefcake.judge import ASSUMES_DEVICE_JUDGE, judge_all, swap_test, tpr_tnr  # noqa: E402

OUT = ROOT / "data" / "example_runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    model = args.model or llm.get_model("judge")
    OUT.mkdir(exist_ok=True)

    traces = pd.read_csv(ROOT / "data" / "exercise8_labeled_traces.csv")
    run = judge_all(traces[traces["Split"].isin(["dev", "test"])], ASSUMES_DEVICE_JUDGE, model=model)
    if (run["Judge"] == "ERROR").any():
        raise SystemExit("Some judge calls failed, so nothing was saved. Rerun.")
    run["Judge model"] = model
    run.to_csv(OUT / "judge_assumes_device.csv", index=False)
    for split in ["dev", "test"]:
        part = run[run["Split"] == split]
        print(split, json.dumps(tpr_tnr(part["Assumes device"], part["Judge"])))

    pairs = pd.read_csv(ROOT / "data" / "exercise9_pairs.csv")
    swaps = swap_test(pairs, model=model)
    if (swaps[["A first", "B first"]] == "?").to_numpy().any():
        raise SystemExit("Some pairwise verdicts couldn't be read, so swap_test.csv wasn't saved. Rerun.")
    swaps["Judge model"] = model
    swaps.to_csv(OUT / "swap_test.csv", index=False)
    print(f"Swap test: {int((~swaps['Consistent']).sum())} of {len(swaps)} verdicts flipped")


if __name__ == "__main__":
    main()
