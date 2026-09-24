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
        !pip install -q litellm
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
    # "Invents policy": ["T03", "T12"],
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
Reviewer B passes the two borderline battery and heart-rate traces that Reviewer A fails.
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
save it, and run the next cell. Change `MY_FAILURE_MODES` to the modes your group found.
"""),
        code('''
MY_FAILURE_MODES = FAILURE_MODES  # or your own list, for example ["Invents policy", "Assumes device"]

sheet = load_traces("traces_v1.csv")[["Trace ID", "User Query", "AI Response"]].head(15).copy()
for mode in MY_FAILURE_MODES:
    sheet[mode] = ""
sheet.to_csv("data/my_eval_sheet.csv", index=False)
print("Wrote data/my_eval_sheet.csv")
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
OTHER_MODEL = "anthropic/claude-haiku-4-5"  # any LiteLLM model name you have a key for

if llm.has_api_key():
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


if __name__ == "__main__":
    nb00()
    nb02()
    nb03()
    nb04()
