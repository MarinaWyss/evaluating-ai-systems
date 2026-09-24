"""Send BeefCake v3 traces to Arize Phoenix, for exploring spans (and the M6.7 screenshot).

    pip install arize-phoenix    # in a separate virtual environment is safest
    phoenix serve                # then open http://localhost:6006
    python scripts/phoenix_demo.py            # the curated v3 traces
    python scripts/phoenix_demo.py --live     # run v3 live on a few questions (needs a key)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beefcake.agents import answer_v3, load_agent_traces  # noqa: E402
from beefcake.phoenix_export import export_traces  # noqa: E402

LIVE_QUESTIONS = [
    "can I still return the bell from order BC-4417? it's too heavy for me",
    "coach charged me twice after I returned the rower",
    "my pulse keeps disconnecting from the rower",
    "order BC-4460 still hasn't arrived",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--endpoint", default="http://localhost:6006/v1/traces")
    args = parser.parse_args()
    traces = [answer_v3(q) for q in LIVE_QUESTIONS] if args.live else load_agent_traces("v3_traces.jsonl")
    export_traces(traces, endpoint=args.endpoint)
    print(f"Sent {len(traces)} traces to Phoenix. Open http://localhost:6006 and pick the beefcake-support-bot project.")


if __name__ == "__main__":
    main()
