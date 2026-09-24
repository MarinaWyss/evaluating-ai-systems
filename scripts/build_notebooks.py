"""Generate the workshop notebooks in notebooks/.

The notebooks are built from this script so they stay consistent and easy to
review in a diff. Edit here, then run:  python scripts/build_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
sys.path.insert(0, str(ROOT))

REPO_URL = "https://github.com/MarinaWyss/evaluating-ai-systems"

SETUP = f'''# Setup: run this cell first. It works in Google Colab and on your own laptop.
import os, sys
REPO_URL = "{REPO_URL}"
if "google.colab" in sys.modules:
    if not os.path.exists("/content/EvalsWorkshop"):
        !git clone -q {{REPO_URL}} /content/EvalsWorkshop
        !pip install -q "litellm>=1.80.5" tenacity
    os.chdir("/content/EvalsWorkshop")
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())

import pandas as pd
from beefcake import llm
llm.load_colab_secrets()
if llm.has_api_key():
    print("API key found. Bot model:", llm.get_model())
else:
    print("No API key found, so this notebook runs in offline mode with pre-generated data.")'''


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def save(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    for i, cell in enumerate(cells):  # stable cell IDs, so rebuilding doesn't create a noisy diff
        cell["id"] = f"{Path(name).stem[:40]}-{i:02d}"
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NB_DIR / name)
    print("wrote", name)


# ---------------------------------------------------------------------------
# 00: setup check (Opening, Exercise 0)
# ---------------------------------------------------------------------------

def nb00():
    save("00_setup_check.ipynb", [
        md("""
# Exercise 0: Setup check

Meet the BeefCake Fitness support bot. BeefCake is a made-up home-gym company with four products:
the **Row** (a smart rowing machine), the **Bell** (an app-connected adjustable kettlebell),
the **Pulse** (a heart-rate chest strap), and **Coach** (a subscription app).

The bot answers customer questions using RAG: it searches our docs for the most relevant chunks,
pastes them into the prompt, and asks the model to answer.

**Your job for the next 5 minutes:** run the cells below, ask the bot three questions, and raise
your hand if anything breaks. If an answer seems a little off, remember it.

**API key:** in Colab, click the key icon on the left, add a secret named `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`, and turn on notebook access. On your laptop, copy
`.env.example` to `.env` and fill in one key. No key? Everything still runs, and you'll see what
the bot retrieves instead of its answer.
"""),
        code(SETUP),
        md("## Ask the bot a question"),
        code('''
from beefcake.bot import answer
from beefcake.retrieval import retrieve

def ask(question):
    if not question.strip():
        print("Type a question between the quotes, then run the cell again.")
        return
    if llm.has_api_key():
        trace = answer(question)
        print(trace.ai_response)
        print("\\n(retrieved:", ", ".join(trace.retrieved_ids) + ")")
    else:
        print("No API key, so here's what retrieval found. The model would answer from these chunks:\\n")
        for chunk in retrieve(question):
            print(chunk.as_context(), "\\n")

ask("how do i pair the heart rate strap with the rower")
'''),
        md("Now try your own. Ask about any of the products, a return, the warranty, or the Coach subscription."),
        code('ask("")  # your first question'),
        code('ask("")  # your second question'),
        md("That's it for setup. Keep this notebook open: you'll use the bot again later today."),
    ])


# ---------------------------------------------------------------------------
# 02: error analysis (Module 2, Exercises 2 and 3)
# ---------------------------------------------------------------------------

def nb02():
    save("02_error_analysis.ipynb", [
        md("""
# Module 2: Error analysis

1. See how a trace is made.
2. **Exercise 2 (open coding):** read traces and write a short note on anything wrong.
3. **Exercise 3 (axial coding):** group your notes into failure modes and count them.
"""),
        code(SETUP),
        md("""
## 1. How traces are made

We start from a set of test questions, written to cover different topics and different kinds of users.
"""),
        code('''
questions = pd.read_csv("data/test_questions.csv")
questions.head(10)
'''),
        md("""
Each question goes through the bot, and we save the **whole trace**: everything the model saw
(the system prompt and the retrieved chunks) plus its answer. Here's what the model sees for one question:
"""),
        code('''
from beefcake.bot import build_messages
from beefcake.retrieval import retrieve

q = "it keeps disconnecting"
for message in build_messages(q, retrieve(q)):
    print(message["role"].upper() + ":")
    print(message["content"], "\\n")
'''),
        md("""
`scripts/generate_traces.py` runs every test question through the bot and writes one row per trace to a CSV.
The traces you'll read today are pre-generated, so nobody waits on API calls and everyone sees the same
failures. If you have a key and want your own, set `RUN_LIVE = True`.
"""),
        code('''
RUN_LIVE = False  # optional: generate your own traces with a live model (a minute or two)

if RUN_LIVE and llm.has_api_key():
    from beefcake.bot import answer
    from beefcake.traces import save_traces
    traces = [answer(r["User Query"], query_topic=r["Query Topic"], trace_id=r["Question ID"])
              for _, r in questions.iterrows()]
    save_traces(traces, "data/my_traces_v1.0.csv")
    print("Saved data/my_traces_v1.0.csv")
'''),
        md("""
## 2. Exercise 2: Open coding (20 min)

**On your own first (12 min).** Read each trace, including the history, and write either `PASS` or a short
note about what's wrong. Put on your product owner hat: the bot should be helpful, stick to our docs and
policies, and never make up a policy. Don't fix anything yet. Just write it down.

**Then compare with your partner (8 min).** Notice where you disagree.

You can work in a spreadsheet instead: open `data/exercise2_traces.csv` in Google Sheets or Excel and use
the "Your open code" column. The docs are in the `docs/` folder if you want to check a policy.
"""),
        code('''
from beefcake.traces import load_traces, show_trace

ex2 = load_traces("exercise2_traces.csv")
print(len(ex2), "traces")
'''),
        code('''
i = 0  # change this number and rerun to move through the traces
show_trace(ex2, i)
'''),
        code('''
# Your open codes: "PASS" or a short note about what's wrong.
open_codes = {
    # "T01": "Assumed 'it' was the Row",
    # "T04": "PASS",
}
'''),
        code('''
ex2["Your open code"] = ex2["Trace ID"].map(open_codes).fillna("")
ex2.to_csv("data/my_open_codes.csv", index=False)
print(f"Saved {(ex2['Your open code'] != '').sum()} open codes to data/my_open_codes.csv")
'''),
        md("""
## 3. Exercise 3: Axial coding (12 min)

In groups of four, put all your open codes together. Group the ones that describe the same problem,
give each group a short name, and count the traces in each. Aim for categories you could actually fix:
one giant "bad answer" group is too broad, and fifteen groups of one is too narrow.
"""),
        code('''
# failure mode name -> the trace IDs in that group
failure_modes = {
    # "Short name for the problem": ["T01"],
}

counts = pd.DataFrame([{"Failure mode": name, "Traces": len(ids)} for name, ids in failure_modes.items()])
counts.sort_values("Traces", ascending=False) if len(counts) else print("Add some groups above")
'''),
        md("### Compare with ours (after the debrief)\n\nThis is how we labeled all 25 traces."),
        code('''
from beefcake.evals import FAILURE_MODES

ours = load_traces("traces_v1_labeled.csv")
pd.DataFrame({"Failure mode": FAILURE_MODES,
              "Traces (of 25)": [int((ours[m] == "FAIL").sum()) for m in FAILURE_MODES]})
'''),
        code('ours[["Trace ID", "User Query", "Open code"]]'),
    ])


# ---------------------------------------------------------------------------
# 03: rubric and agreement (Module 3, Exercise 4a)
# ---------------------------------------------------------------------------

def nb03():
    ids = pd.read_csv(ROOT / "data" / "exercise4a_traces.csv")["Trace ID"].tolist()
    blank = "{\n" + "".join(f'    "{i}": "",\n' for i in ids) + "}"
    save("03a_rubric_agreement.ipynb", [
        md("""
# Exercise 4a: Rubric stress test (25 min)

1. **Write (8 min):** pick one failure mode from Exercise 3 and write its rubric entry from the template.
2. **Label (6 min):** two people label the same 12 traces **on their own**. No peeking, no discussing.
3. **Compare (8 min):** compute raw agreement and Cohen's kappa, then go through every disagreement and
   fix the rubric.
4. **Share (3 min):** one rubric change and what caused it.
"""),
        code(SETUP),
        md("## 1. Write your rubric\n\nOpen `rubric/rubric_template.md`, copy it, and fill it in. The worked example at the bottom shows a finished entry."),
        code('print(open("rubric/rubric_template.md").read())'),
        md("## 2. Label the 12 traces on your own\n\nThese are new traces you haven't seen. Label each one PASS or FAIL **for your failure mode only**."),
        code('''
from beefcake.traces import load_traces, show_trace

traces = load_traces("exercise4a_traces.csv")
for i in range(len(traces)):
    show_trace(traces, i, show_history=False)
'''),
        md("**Reviewer A** fills in this cell, and **Reviewer B** fills in the next one, each on their own laptop or without looking."),
        code("reviewer_a = " + blank),
        code("reviewer_b = " + blank),
        md("## 3. How much do you agree?"),
        code('''
from beefcake.evals import cohens_kappa, disagreements, raw_agreement

labels = traces[["Trace ID", "User Query", "AI Response"]].copy()
labels["A"] = labels["Trace ID"].map(reviewer_a).str.strip().str.upper()
labels["B"] = labels["Trace ID"].map(reviewer_b).str.strip().str.upper()
both = labels[labels["A"].isin(["PASS", "FAIL"]) & labels["B"].isin(["PASS", "FAIL"])]

if len(both) == 0:
    print("Fill in both reviewers' labels first, or run the demo at the bottom.")
else:
    print(f"Labeled by both: {len(both)} traces")
    print(f"Raw agreement:   {raw_agreement(both['A'], both['B']):.0%}")
    print(f"Cohen's kappa:   {cohens_kappa(both['A'], both['B']):.2f}")
    display(disagreements(both, "A", "B"))
'''),
        md("""
**For every disagreement, ask:**

- Which part of the rubric caused it? A vague word in the definition? A missing edge case? No example like it?
- What rule or example would make the next decision obvious? Add it, bump the version, and write one line in the changelog.
- Still stuck? The owner of the rubric makes the call and writes down why.

Rough guide for kappa: above 0.6 is acceptable and above 0.8 is strong. These are conventions, not laws.

If you finish early, relabel the traces with your new version and see whether agreement goes up.
"""),
        md("""
## Demo: what the numbers look like

No labels yet? Here are two example reviewers labeling the 25 traces from Module 2 for **Assumes device**.
They disagree on two borderline traces: Reviewer B passes the battery trace that Reviewer A fails, and fails the
heart-rate trace that Reviewer A passes.
"""),
        code('''
demo = load_traces("traces_v1_labeled.csv")[["Trace ID", "User Query", "AI Response", "Assumes device"]]
demo = demo.rename(columns={"Assumes device": "A"})
demo["B"] = demo["A"]
borderline = demo["User Query"].isin(["the battery dies really fast", "the heart rate numbers are way off"])
demo.loc[borderline, "B"] = demo.loc[borderline, "A"].map({"PASS": "FAIL", "FAIL": "PASS"})

print(f"Raw agreement: {raw_agreement(demo['A'], demo['B']):.0%}")
print(f"Cohen's kappa: {cohens_kappa(demo['A'], demo['B']):.2f}")
disagreements(demo, "A", "B")
'''),
        md("Notice how 92% raw agreement becomes a much lower kappa. Most traces are an easy PASS, so agreeing on those is cheap. Kappa corrects for that."),
    ])


# ---------------------------------------------------------------------------
# 04: eval suite (Module 3, Exercise 4b)
# ---------------------------------------------------------------------------

def nb04():
    save("03b_eval_suite.ipynb", [
        md("""
# Module 3: From rubric to eval suite

1. The eval spreadsheet: one PASS/FAIL column per failure mode.
2. Is an improvement real? Intervals and McNemar's test.
3. **Exercise 4b (20 min):** label 15 traces, write a reference-based eval for the device classifier,
   and write two code checks.
4. Stretch goals: fix a failure in the prompt, and compare a second model.
"""),
        code(SETUP),
        md("## 1. The eval spreadsheet\n\nEach row is a trace. Each failure mode is a column. Each cell is PASS or FAIL."),
        code('''
from beefcake.evals import FAILURE_MODES, failure_rates, mcnemar_exact, run_checks, compare_to_labels
from beefcake.traces import load_traces

before = load_traces("traces_v1_labeled.csv")
before[["Trace ID", "User Query"] + FAILURE_MODES].head(8)
'''),
        md("The bottom row of the spreadsheet is a COUNTIF per column. Here it is, with a 95% interval for each rate:"),
        code("failure_rates(before)"),
        md("## 2. Is the improvement real?\n\nWe added one rule to the prompt (version v1.1):"),
        code('''
from beefcake.bot import PROMPTS
print(PROMPTS["v1.1"][len(PROMPTS["v1.0"]):].strip())
'''),
        code('''
after = load_traces("traces_v1.1_labeled.csv")
failure_rates(after)
'''),
        md("""
Invents policy went from 8 of 25 to 4 of 25. The intervals overlap a lot, so compare question by question:
the only questions that tell you anything are the ones that flipped.
"""),
        code('mcnemar_exact(before["Invents policy"], after["Invents policy"])'),
        md("""
Four questions got fixed and none broke, and it still isn't significant at the usual 0.05 level with only
25 questions. That's why you want around 100.
"""),
        md("""
## 3. Exercise 4b: Build the suite (20 min)

### Step 1: label 15 traces with your rubric

This cell writes a blank eval sheet. Open `data/my_eval_sheet.csv` in a spreadsheet, fill in PASS or FAIL,
save it, and run the next cell. Change `MY_FAILURE_MODES` to the modes your group found. If the sheet already
exists, the cell leaves it alone, so rerunning the notebook won't wipe your labels.
"""),
        code('''
from pathlib import Path

MY_FAILURE_MODES = FAILURE_MODES  # or your own list, for example ["Invents policy", "Assumes device"]

SHEET = Path("data/my_eval_sheet.csv")
if SHEET.exists():
    print(f"Keeping your labels in {SHEET}. Delete the file to start over.")
else:
    sheet = load_traces("traces_v1.csv")[["Trace ID", "User Query", "AI Response"]].head(15).copy()
    for mode in MY_FAILURE_MODES:
        sheet[mode] = ""
    sheet.to_csv(SHEET, index=False)
    print(f"Wrote {SHEET}")
'''),
        code('''
mine = pd.read_csv("data/my_eval_sheet.csv", keep_default_na=False)
failure_rates(mine, MY_FAILURE_MODES)
'''),
        md("""
### Step 2: a reference-based eval for the device classifier

The bot has a small classifier that says which product a question is about. There's exactly one right answer
per question, so this is a normal test. Finish `eval_device_classifier` so it returns `"PASS"` when the
prediction matches the expected label and `"FAIL"` otherwise.

Without an API key, it uses example predictions saved in the golden set.
"""),
        code('''
from beefcake.classifier import classify_device

golden = pd.read_csv("data/classifier_golden.csv")
EXAMPLE_PREDICTIONS = dict(zip(golden["User Query"], golden["Example prediction"]))

def eval_device_classifier(query, expected):
    predicted = classify_device(query) if llm.has_api_key() else EXAMPLE_PREDICTIONS[query]
    # YOUR CODE HERE: return "PASS" or "FAIL"
    ...
'''),
        code('''
golden["Result"] = [eval_device_classifier(q, e) for q, e in zip(golden["User Query"], golden["Expected"])]
print(f"Accuracy: {(golden['Result'] == 'PASS').mean():.0%}")
golden
'''),
        md("""
### Step 3: two code checks

A code check is a plain function that looks at one trace and returns True if it passes. Here's one for
"Leaks RAG setup". Write another for an easy failure mode, then run both over every trace and compare
with the human labels.

Ideas: every "N-year warranty" in an answer must match the policy; the trial must be "14-day"; a return
window must be 30 days. Watch for checks that are too broad and fail good answers.
"""),
        code('''
def no_rag_leak(row):
    """PASS if the answer doesn't talk about 'the provided context'."""
    return "provided context" not in row["AI Response"].lower()

def my_check(row):
    # YOUR CODE HERE: return True if the trace passes
    return True

checked = run_checks(before, {"Check: no RAG leak": no_rag_leak, "Check: mine": my_check})
checked[["Trace ID", "User Query", "Check: no RAG leak", "Leaks RAG setup", "Check: mine"]]
'''),
        md("Where does the code check disagree with the human label? An empty table means they agree on every trace."),
        code('compare_to_labels(checked, "Check: no RAG leak", "Leaks RAG setup")'),
        md("""
## 4. Stretch goals

### Fix one failure in the prompt

Add a rule to the prompt, rerun the questions that failed, and read the new answers. For a fair comparison,
regenerate the "before" answers with the same live model too, because the pre-generated traces weren't written
by your model.
"""),
        code('''
from beefcake.bot import answer

PROMPTS["my_fix"] = PROMPTS["v1.0"] + " YOUR RULE HERE"

failing = before[before["Invents policy"] == "FAIL"]
if llm.has_api_key():
    for _, row in failing.iterrows():
        new = answer(row["User Query"], prompt_version="my_fix")
        print(row["Trace ID"], row["User Query"], "\\n  ->", new.ai_response, "\\n")
else:
    print("Needs an API key.")
'''),
        md("### Compare a second model on quality, cost, and latency"),
        code('''
if llm.has_api_key():
    OTHER_MODEL = llm.get_model("judge")  # a stronger model from your provider, or any LiteLLM model you have a key for
    rows = []
    for q in load_traces("traces_v1.csv")["User Query"].head(5):
        for model in [llm.get_model(), OTHER_MODEL]:
            t = answer(q, model=model)
            rows.append({"model": model, "question": q, "answer": t.ai_response,
                         "latency_s": llm.LAST_CALL.get("latency_s"), "cost_usd": llm.LAST_CALL.get("cost_usd")})
    comparison = pd.DataFrame(rows)
    display(comparison.groupby("model")[["latency_s", "cost_usd"]].mean())
    display(comparison)
else:
    print("Needs an API key.")
'''),
        md("""
---
## Solutions (try it yourself first)
"""),
        code('''
def eval_device_classifier_solution(query, expected):
    predicted = classify_device(query) if llm.has_api_key() else EXAMPLE_PREDICTIONS[query]
    return "PASS" if predicted == expected else "FAIL"

import re

def warranty_matches_policy(row):
    """PASS if every 'N-year warranty' in the answer is 1 or 2 years. (Crude: it doesn't check which product.)"""
    years = re.findall(r"(\\d+|one|two|three)-year warranty", row["AI Response"].lower())
    return all(y in {"1", "2", "one", "two"} for y in years)

golden["Solution"] = [eval_device_classifier_solution(q, e) for q, e in zip(golden["User Query"], golden["Expected"])]
print(f"Classifier accuracy: {(golden['Solution'] == 'PASS').mean():.0%}")
display(golden[golden["Solution"] == "FAIL"])

checked = run_checks(before, {"Check: warranty": warranty_matches_policy})
checked.loc[checked["Check: warranty"] == "FAIL", ["Trace ID", "User Query", "Invents policy"]]
'''),
    ])


# ---------------------------------------------------------------------------
# 04: component evals (Module 4, Exercises 5, 6, and 7)
# ---------------------------------------------------------------------------

def nb_m4():
    questions = pd.read_csv(ROOT / "data" / "exercise6_questions.csv")["Question"].tolist()
    relevant_blank = "relevant = {\n" + "".join(f'    "{q}": [],\n' for q in questions) + "}"
    save("04_component_evals.ipynb", [
        md("""
# Module 4: Component-level evals

1. **Exercise 5 (blame game):** three wrong answers. Which piece broke?
2. Retrieval metrics, and **Exercise 6:** label relevant chunks and measure retrieval.
3. Groundedness: does the answer stick to what was retrieved?
4. **Exercise 7:** checks on logged tool calls.

This afternoon the bot is **version 2**: an agent with tools. It can search the docs, look up orders, start
returns, check warranties, and hand off to a person with `create_ticket`. There's deliberately no refund tool.
"""),
        code(SETUP),
        md("""
## 1. Exercise 5: Blame game (8 min)

Three wrong answers, with full traces. For each one, decide which piece broke: **retrieval** (it fetched the
wrong text), **prompt** (an instruction is missing), **generation** (it had the right text and still got it
wrong), or **tool call** (a tool was called wrong). First table with all three right wins.
"""),
        code('''
from beefcake.agents import load_agent_traces, show_agent_trace

blame = load_agent_traces("exercise5_blame_game.jsonl")
for t in blame:
    show_agent_trace(t, full=True)
    print()
'''),
        code('my_answers = {"B1": "", "B2": "", "B3": ""}  # retrieval, prompt, generation, or tool call'),
        md("""
## 2. Retrieval metrics

| Metric | What it measures |
|---|---|
| Hit rate@k | Share of questions with a relevant chunk in the top k |
| Recall@k | Share of all relevant chunks that made the top k |
| Precision@k | Share of the top k that are relevant |
| MRR | Average of 1 / rank of the first relevant chunk |

### Exercise 6: Retrieval eval (12 min)

1. For each golden question, look at the top 5 chunks and write down the **section IDs** that actually answer it.
   Ask yourself: could the bot answer from this chunk alone? Mentioning the right product isn't enough.
2. Compute hit rate@3 and MRR.
3. Change one thing (`TOP_K`, or `MAX_WORDS` for the chunk size) and rerun.
"""),
        code('''
from beefcake.retrieval import BM25Retriever, load_chunks

golden = pd.read_csv("data/exercise6_questions.csv")["Question"].tolist()
candidates = BM25Retriever()
for q in golden:
    print("QUESTION:", q)
    for rank, chunk in enumerate(candidates.retrieve(q, k=5), 1):
        print(f"  {rank}. {chunk.section_id}\\n     {chunk.text[:140]}...")
    print()
'''),
        md("""
Every golden question has at least one relevant section somewhere in the docs. If none of the top 5 answer it,
find the right one in this list of all sections, so a miss counts as a miss.
"""),
        code('''
for c in load_chunks():
    print(f"{c.section_id:65} {c.text[:60]}...")
'''),
        code(relevant_blank),
        code('''
from beefcake.retrieval_metrics import hit_rate, precision, recall, reciprocal_rank

TOP_K = 3          # try 5
MAX_WORDS = None   # try 35 or 20 to split long sections into smaller chunks

def retrieval_report(relevant, top_k=TOP_K, max_words=MAX_WORDS):
    retriever = BM25Retriever(load_chunks(max_words=max_words))
    rows = []
    for q, rel in relevant.items():
        ranked = []
        for c in retriever.retrieve(q, k=20):
            if c.section_id not in ranked:
                ranked.append(c.section_id)
        rows.append({"question": q, "retrieved": ranked[:top_k],
                     "hit": hit_rate(ranked, set(rel), top_k),
                     "RR": round(reciprocal_rank(ranked, set(rel), top_k), 2),
                     "precision": round(precision(ranked, set(rel), top_k), 2),
                     "recall": round(recall(ranked, set(rel), top_k), 2)})
    table = pd.DataFrame(rows)
    print(f"top_k={top_k}, max_words={max_words}:  hit rate {table['hit'].mean():.0%},  MRR {table['RR'].mean():.2f},"
          f"  precision {table['precision'].mean():.2f},  recall {table['recall'].mean():.2f}")
    return table

missing = [q for q, r in relevant.items() if not r]
if missing:
    print(f"Label every question first ({len(missing)} still empty). Leaving one out would hide a miss.")
else:
    display(retrieval_report(relevant))
'''),
        md("""
Did MRR and hit rate move together when you changed something? If you tried a larger `TOP_K`, what happened to
precision, and to how long the prompt gets? Sections you never labeled count as not relevant, so if a new
setting surfaces one, label it.
"""),
        md("""
## 3. Groundedness

Split the answer into claims and check each claim against the retrieved text. One unsupported claim is enough to
flag the answer. Here's round 2 of the blame game, checked by an LLM. Like any judge, this checker needs
validating against human labels before you rely on it.
"""),
        code('''
from beefcake.judge import groundedness
from beefcake.tools import run_tool

b2 = blame[1]
context = "\\n\\n".join(run_tool("search_docs", {"query": "rower warranty"})["results"])
print("ANSWER:", b2.final_answer, "\\n")
if llm.has_api_key():
    display(groundedness(b2.final_answer, context))
else:
    print("Needs an API key. By hand: the '3-year warranty' claim isn't supported, because the policy says 2 years.")
'''),
        md("""
## 4. Exercise 7: Tool-call eval (8 min)

Here are logged tool calls from version 2, and what we expected for each. Finish the four checks, run them,
and see which traces fail. Things you can use:

- `first_action_tool(trace)` gives the first tool that isn't a doc search (or `"none"`)
- `schema_errors(name, arguments)` lists problems with a call's arguments
- `APPROVED_TOOLS` is the set of real tool names
- `trace.tool_calls` is the list of tool-call spans (each has `name` and `input`), and `trace.end_state` is what
  the store looks like afterwards
"""),
        code('''
import json
from beefcake.checks import first_action_tool
from beefcake.tools import APPROVED_TOOLS, schema_errors

tool_traces = {t.trace_id: t for t in load_agent_traces("exercise7_tool_calls.jsonl")}
expected = pd.read_csv("data/exercise7_expected.csv").set_index("Trace ID")
show_agent_trace(tool_traces["C04"])
expected
'''),
        code('''
def check_expected_tool(trace, exp):
    # YOUR CODE HERE: True if the first action tool matches exp["Expected first tool"]
    return True

def check_schema(trace, exp):
    # YOUR CODE HERE: True if every approved tool call has valid arguments
    return True

def check_no_made_up_tools(trace, exp):
    # YOUR CODE HERE: True if every tool call is in APPROVED_TOOLS
    return True

def check_end_state(trace, exp):
    # YOUR CODE HERE: compare trace.end_state with json.loads(exp["Expected returns"]) and exp["Expected tickets"]
    return True

CHECKS = {"expected tool": check_expected_tool, "schema": check_schema,
          "no made-up tools": check_no_made_up_tools, "end state": check_end_state}

def run_tool_checks(traces, checks_to_run):
    return pd.DataFrame([
        {"Trace ID": tid, **{name: "PASS" if fn(t, expected.loc[tid]) else "FAIL" for name, fn in checks_to_run.items()}}
        for tid, t in traces.items()
    ])

run_tool_checks(tool_traces, CHECKS)
'''),
        md("""
One of these traces takes a different path to the right end state. Did any of your checks fail it? Should it
have failed? Also look for traces that pass every argument check and still leave the store in the wrong state.
"""),
        md("""
### Stretch: run version 2 live

With a key, run the same questions through the live agent and apply your checks. Your model will make
different mistakes from the logged ones.
"""),
        code('''
from beefcake.agents import answer_v2

if llm.has_api_key():
    live = {tid: answer_v2(expected.loc[tid, "User Query"], trace_id=tid) for tid in expected.index[:5]}
    display(run_tool_checks(live, CHECKS))
    show_agent_trace(live["C01"])
else:
    print("Needs an API key.")
'''),
        md("---\n## Solutions (try it yourself first)"),
        code('''
from beefcake import checks

SOLUTION_CHECKS = {
    "expected tool": lambda t, e: checks.first_action_tool(t) == e["Expected first tool"],
    "schema": lambda t, e: checks.arguments_match_schema(t),
    "no made-up tools": lambda t, e: checks.only_approved_tools(t),
    "end state": lambda t, e: checks.end_state_matches(t, json.loads(e["Expected returns"]), int(e["Expected tickets"])),
    "no false success": lambda t, e: checks.no_false_success(t),
}
run_tool_checks(tool_traces, SOLUTION_CHECKS)
'''),
    ])


# ---------------------------------------------------------------------------
# 05: LLM judges (Module 5, Exercises 8 and 9)
# ---------------------------------------------------------------------------

def nb_m5():
    save("05_llm_judge.ipynb", [
        md("""
# Module 5: LLM-as-judge that you can trust

1. Our judge for "Assumes device", built from the rubric.
2. **Exercise 8 (25 min):** build your own judge, align it on the dev set, run it once on the test set.
3. Correcting a pass rate for an imperfect judge.
4. **Exercise 9 (demo):** the swap test for position bias.

Judges need an API key. Without one, you'll see a recorded run if the facilitator has saved one.
"""),
        code(SETUP),
        md("## 1. A real judge\n\nOverview, one failure mode, PASS and FAIL criteria, reasoning before the verdict, examples, then the trace."),
        code('''
from beefcake.judge import ASSUMES_DEVICE_JUDGE, build_judge_prompt, judge_all, run_judge, tpr_tnr
print(ASSUMES_DEVICE_JUDGE)
'''),
        md("""
## 2. Exercise 8: Judge alignment (25 min)

1. Build an **Assumes device** judge with the six-part template (8 min). Start from your Exercise 4a rubric if
   you wrote it for Assumes device, or from the worked example in `rubric/rubric_template.md`.
2. Run it on the **dev** set. Report TPR and TNR (5 min).
3. Read the reasoning on every disagreement. Improve the prompt once. Rerun (8 min).
4. **One** run on the **test** set. Post your numbers (4 min).

Every group builds the same judge, because these traces are labeled for Assumes device only. They're split
about 10% train (for your few-shot examples), 40% dev, and 50% test.
"""),
        code('''
traces = pd.read_csv("data/exercise8_labeled_traces.csv")
train, dev, test = (traces[traces["Split"] == s] for s in ["train", "dev", "test"])
print(len(train), "train,", len(dev), "dev,", len(test), "test")
display(traces.groupby("Split")["Assumes device"].value_counts().unstack())
train[["User Query", "AI Response", "Assumes device", "Notes"]]
'''),
        code('''
# Build your judge from your rubric. Examples come from the TRAIN split only.
examples = [
    {"user_query": r["User Query"], "ai_response": r["AI Response"], "judgment": r["Assumes device"], "reasoning": r["Notes"]}
    for _, r in pd.concat([train[(train["Assumes device"] == "PASS") & ~train["Notes"].str.contains("different failure")].head(2),
                           train[train["Assumes device"] == "FAIL"].head(2)]).iterrows()
]  # pick your own: the most useful examples are the borderline ones

my_judge = build_judge_prompt(
    failure_mode="Assumes device",
    definition="YOUR DEFINITION FROM THE RUBRIC",
    pass_criteria="YOUR PASS CRITERIA",
    fail_criteria="YOUR FAIL CRITERIA",
    examples=examples,
)
print(my_judge)
'''),
        md("### Run it on the dev set"),
        code('''
from pathlib import Path

RECORDED = Path("data/example_runs/judge_assumes_device.csv")

if llm.has_api_key():
    dev_run = judge_all(dev, my_judge)
elif RECORDED.exists():
    print("No API key: showing the recorded run of OUR judge, not yours.")
    dev_run = pd.read_csv(RECORDED)
    dev_run = dev_run[dev_run["Split"] == "dev"]
else:
    dev_run = None
    print("Needs an API key (or a recorded run in data/example_runs/).")

if dev_run is not None:
    print(tpr_tnr(dev_run["Assumes device"], dev_run["Judge"]))
'''),
        md("### Read every disagreement\n\nIs the prompt ambiguous? Is an edge case missing? Or was the human label wrong?"),
        code('''
if dev_run is not None:
    wrong = dev_run[dev_run["Judge"] != dev_run["Assumes device"]]
    for _, r in wrong.iterrows():
        print(f"{r['Trace ID']}  human={r['Assumes device']}  judge={r['Judge']}")
        print(f"  USER:  {r['User Query']}\\n  BOT:   {r['AI Response']}\\n  JUDGE: {r['Judge reasoning']}\\n")
'''),
        md("""
Improve the prompt once and rerun the dev cell. When you're happy, run the test set **once**. Your leaderboard
score is the lower of TPR and TNR, so a judge that says PASS to everything can't win.
"""),
        code('''
RUN_TEST = False  # set to True once, when you're done iterating on dev

if RUN_TEST and llm.has_api_key():
    test_run = judge_all(test, my_judge)
    rates = tpr_tnr(test_run["Assumes device"], test_run["Judge"])
    print(rates, "  leaderboard score:", min(rates["TPR"], rates["TNR"]))
'''),
        md("""
## 3. Correcting for an imperfect judge

If your judge has TPR 89% and TNR 87% and says 80% of production traces pass, the true pass rate is about 88%:
`(observed + TNR - 1) / (TPR + TNR - 1)`. It only works if TPR + TNR is above 1.
"""),
        code('''
from beefcake.judge import corrected_pass_rate, rogan_gladen

print(f"Worked example: {rogan_gladen(0.80, 0.89, 0.87):.1%}")
'''),
        md("""
With a labeled test set, you can also get an interval. The production verdicts below come from a synthetic
production log (made up for this demo). Notice how wide the interval is with a small test set.
"""),
        code('''
production = pd.read_csv("data/production_log_synthetic.csv", keep_default_na=False)
judged = production.loc[production["Judge: Assumes device"] != "", "Judge: Assumes device"]
if "test_run" in globals():
    labeled_run = test_run                     # your judge's one test-set run
elif RECORDED.exists():
    labeled_run = pd.read_csv(RECORDED)
    labeled_run = labeled_run[labeled_run["Split"] == "test"]   # the recorded run of our judge
else:
    labeled_run = None
    print("Needs a judge run on the test set: yours (RUN_TEST above) or a recorded one.")
if labeled_run is not None:
    print(corrected_pass_rate(labeled_run["Assumes device"], labeled_run["Judge"], judged))
'''),
        md("""
## 4. Exercise 9: The swap test (demo)

A pairwise judge sees two replies and picks the better one. Run every pair in both orders: a consistent judge
picks the same reply both times. In each pair here, both replies give the same answer and one adds friendly
filler, and the judge has to answer "1" or "2", with no option for a tie.
"""),
        code('''
from beefcake.judge import swap_test

pairs = pd.read_csv("data/exercise9_pairs.csv")
SWAPS = Path("data/example_runs/swap_test.csv")
if llm.has_api_key():
    swaps = swap_test(pairs)
elif SWAPS.exists():
    swaps = pd.read_csv(SWAPS)
    print("Recorded run with", swaps["Judge model"].iloc[0])
else:
    swaps = None
    print("Needs an API key (or a recorded run in data/example_runs/).")
if swaps is not None:
    print(f"{(~swaps['Consistent'].astype(bool)).sum()} of {len(swaps)} verdicts flipped when the order changed")
    display(swaps)
'''),
    ])


# ---------------------------------------------------------------------------
# 06: agents in production (Module 6, Exercise 10)
# ---------------------------------------------------------------------------

def nb_m6():
    save("06_agents_production.ipynb", [
        md("""
# Module 6: Multi-agent systems in production

1. Version 3 of the bot: a router and two specialist agents.
2. **Exercise 10 (15 min):** multi-agent autopsy.
3. Agent metrics: task success and trajectory efficiency.
4. A production log: latency, rates by topic, what to sample for review, escalations.
5. Evals in CI: a suite threshold versus per-check gates.

Exercise 11 (your production eval plan) is on paper.
"""),
        code(SETUP),
        md("## 1. Version 3: a router, a handoff, and two agents"),
        code('''
from beefcake.agents import answer_v3, load_agent_traces, show_agent_trace

v3 = {t.trace_id: t for t in load_agent_traces("v3_traces.jsonl")}
if llm.has_api_key():
    show_agent_trace(answer_v3("can I still return the bell from order BC-4417? it's too heavy for me"))
else:
    show_agent_trace(v3["M2"])
'''),
        md("""
## 2. Exercise 10: Multi-agent autopsy (15 min)

This trace has a wrong final answer.

1. Read it span by span and write open codes.
2. Which agent and which span went wrong first?
3. Check your codes against MAST's categories: system design, inter-agent misalignment, task verification.
4. Write the eval that would have caught it: a code check, a handoff assertion, or a judge.
"""),
        code('show_agent_trace(v3["M1"], full=True)'),
        code('''
my_autopsy = {
    "open codes": [],
    "agent and span that went wrong first": "",
    "MAST category": "",
    "eval that would catch it": "",
}
'''),
        md("A handoff is recorded like a tool call (`transfer_to_...`), so you can assert on it. Write a routing check:"),
        code('''
from beefcake.checks import routed_correctly

def expected_agent(query):
    # YOUR CODE HERE: return "orders_billing" or "device_support"
    return "device_support"

for tid, t in v3.items():
    print(tid, t.user_query, "->", "PASS" if routed_correctly(t, expected_agent(t.user_query)) else "FAIL")
'''),
        md("With a key, test the live router on a small golden set:"),
        code('''
from beefcake.agents import route

router_golden = pd.read_csv("data/router_golden.csv")
if llm.has_api_key():
    router_golden["Routed to"] = [route(q) for q in router_golden["User Query"]]
    router_golden["Result"] = ["PASS" if a == b else "FAIL"
                               for a, b in zip(router_golden["Routed to"], router_golden["Expected agent"])]
router_golden
'''),
        md("""
## 3. Agent metrics

Task success: did it reach the goal? Check the end state with code. Trajectory efficiency: the shortest path
that would have worked, divided by the path the agent took.
"""),
        code('''
import json
from beefcake import checks

tool_traces = load_agent_traces("exercise7_tool_calls.jsonl")
expected = pd.read_csv("data/exercise7_expected.csv").set_index("Trace ID")
SHORTEST = {"C09": 2}  # look up the order, then hand off. Everything else needs one call at most.

rows = []
for t in tool_traces:
    e = expected.loc[t.trace_id]
    rows.append({"Trace ID": t.trace_id, "User Query": t.user_query,
                 "Task success": checks.end_state_matches(t, json.loads(e["Expected returns"]), int(e["Expected tickets"]))
                                 and checks.no_false_success(t) and checks.arguments_match_schema(t)
                                 and checks.only_approved_tools(t),
                 "Tool calls": len(t.tool_calls),
                 "Efficiency": checks.trajectory_efficiency(t, SHORTEST.get(t.trace_id, 1))})
agent_metrics = pd.DataFrame(rows)
print(f"Task success: {agent_metrics['Task success'].mean():.0%}")
agent_metrics
'''),
        md("""
The end state alone isn't enough for read-only requests. C02 changes nothing in the store, so its end state is
"right", but it told the customer their order doesn't exist. That's why task success here also requires valid
arguments and no made-up tools.
"""),
        code('''
# The least efficient runs
agent_metrics.sort_values("Efficiency").head(3)
'''),
        md("""
## 4. Production

This log is **synthetic** (made up for the demo), with one row per conversation.
"""),
        code('''
log = pd.read_csv("data/production_log_synthetic.csv", keep_default_na=False)
lat = log["Latency (s)"]
print(f"{len(log)} conversations.  Latency p50: {lat.quantile(0.5):.1f}s   p99: {lat.quantile(0.99):.1f}s   mean: {lat.mean():.1f}s")
log.groupby("Query Topic").agg(
    conversations=("Trace ID", "count"),
    judge_fail_rate=("Judge: Assumes device", lambda s: (s[s != ""] == "FAIL").mean() if (s != "").any() else None),
    thumbs_down_rate=("Thumbs", lambda s: (s == "down").mean()),
    p99_latency=("Latency (s)", lambda s: s.quantile(0.99)),
).round(2)
'''),
        md("### What should a person read this week?\n\nJudge-flagged traces, negative feedback, outliers, and always some random ones."),
        code('''
review = pd.concat([
    log[log["Judge: Assumes device"] == "FAIL"].sample(8, random_state=1).assign(Why="judge flagged"),
    log[log["Thumbs"] == "down"].sample(6, random_state=1).assign(Why="thumbs down"),
    log.nlargest(3, "Latency (s)").assign(Why="slowest"),
    log.sample(8, random_state=2).assign(Why="random"),
]).drop_duplicates("Trace ID")
print(len(review), "traces to read. (The Assumes-device judge only runs on device questions.)")
review[["Trace ID", "Query Topic", "Why"]]
'''),
        md("### Evaluate the handoff to a person like a classifier\n\nUse the conversations a person labeled."),
        code('''
labeled = log[log["Should escalate (human label)"] != ""].copy()
labeled["should"] = labeled["Should escalate (human label)"].astype(str) == "True"
labeled["did"] = labeled["Escalated"].astype(str) == "True"
missed = int((labeled["should"] & ~labeled["did"]).sum())
unneeded = int((~labeled["should"] & labeled["did"]).sum())
print(f"Labeled conversations: {len(labeled)}")
print(f"Missed escalations:    {missed} of {int(labeled['should'].sum())} that needed a person")
print(f"Unneeded escalations:  {unneeded} of {int((~labeled['should']).sum())} that didn't")
'''),
        md("""
## 5. Evals in CI

`scripts/run_ci_evals.py` runs code checks on every change. Compare the suite threshold with the per-check gates.
"""),
        code('''
import importlib.util

spec = importlib.util.spec_from_file_location("run_ci_evals", "scripts/run_ci_evals.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
report = ci.run()
print("Suite pass rate:", report["suite pass rate"], "->", report["suite gate"])
display(pd.DataFrame(report["per-check gates"]).T)
print("Ship it?", report["ship it"])
'''),
        md("""
The suite clears its 90% threshold while three checks fail. A whole-suite threshold can hide a test that fails
every time, so anything important gets its own gate.

To explore traces span by span in Arize Phoenix, see `scripts/phoenix_demo.py`.
"""),
    ])


if __name__ == "__main__":
    nb00()
    nb02()
    nb03()
    nb04()
    nb_m4()
    nb_m5()
    nb_m6()
