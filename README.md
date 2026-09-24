# Evaluating AI Systems: workshop code

This repo holds the running example for the workshop: the support bot for **BeefCake Fitness**, a made-up
home-gym company. You'll find its failures, write a rubric for them, and turn that rubric into an eval suite.

BeefCake sells four products:

| Product | What it is |
|---|---|
| BeefCake Row | Smart rowing machine with a screen |
| BeefCake Bell | App-connected adjustable kettlebell |
| BeefCake Pulse | Heart-rate chest strap |
| BeefCake Coach | Subscription app with classes |

## Getting started

### Option 1: Google Colab (easiest)

Open any notebook in `notebooks/` in Colab. The first cell downloads the repo and installs what it needs.

To add an API key, click the key icon in Colab's left sidebar, add a secret named `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`, and switch on notebook access.

### Option 2: your laptop

You need Python 3.10 or newer.

```bash
git clone https://github.com/MarinaWyss/evaluating-ai-systems.git
cd evaluating-ai-systems
pip install -r requirements.txt        # or: uv pip install -r requirements.txt
cp .env.example .env                   # then paste in one API key
jupyter notebook notebooks/
```

### No API key?

Every exercise works without one. The traces you'll analyze are pre-generated, and cells that need a live
model tell you so and skip themselves. A key only matters for chatting with the bot, generating your own
traces, and the stretch goals.

### Choosing a model

The bot uses [LiteLLM](https://docs.litellm.ai/), so any major provider works. It picks a default model for
whichever key it finds. To choose one yourself, set `BEEFCAKE_MODEL` (and `BEEFCAKE_JUDGE_MODEL` for judges)
in `.env` using LiteLLM's `provider/model` names, for example `openai/gpt-4o-mini`.

## The notebooks

| Notebook | Workshop section | What you do |
|---|---|---|
| `00_setup_check.ipynb` | Opening, Exercise 0 | Chat with the bot and check your setup |
| `02_error_analysis.ipynb` | Module 2, Exercises 2 and 3 | Read traces, write open codes, group them into failure modes |
| `03a_rubric_agreement.ipynb` | Module 3, Exercise 4a | Write a rubric, label traces with a partner, measure agreement |
| `03b_eval_suite.ipynb` | Module 3, Exercise 4b | Build the eval spreadsheet, a reference-based eval, and code checks |
| `04_component_evals.ipynb` | Module 4, Exercises 5, 6, and 7 | Blame game, retrieval metrics, groundedness, tool-call checks |
| `05_llm_judge.ipynb` | Module 5, Exercises 8 and 9 | Build a judge, align it on dev, test it once, correct a pass rate, swap test |
| `06_agents_production.ipynb` | Module 6, Exercise 10 | Multi-agent autopsy, handoff assertions, agent metrics, a production log, CI gates |

Module 1 and Exercise 11 are on paper, so they have no notebook.

## The three versions of the bot

| Version | Used in | What it is |
|---|---|---|
| v1 (`bot.py`) | Opening to Module 3 | A RAG bot: retrieve chunks from the docs, then answer |
| v2 (`agents.answer_v2`) | Module 4 | One agent with tools: `search_docs`, `lookup_order`, `start_return`, `check_warranty`, `create_ticket` (the handoff to a person). There's deliberately no refund tool |
| v3 (`agents.answer_v3`) | Module 6 | A router hands each question to a Device Support agent or an Orders and Billing agent |

The tools act on a small fake store (`store.py`) with five orders and four devices, reset before every run. Its
"today" is fixed at October 5, 2026, so return windows and warranties behave the same every time.

## What's in the repo

```
beefcake/        the bot and the eval helpers
  bot.py           version 1: a RAG bot (retrieve chunks, then answer)
  retrieval.py     BM25 keyword search over the docs
  classifier.py    which product is the customer asking about?
  traces.py        save, load, and print traces
  evals.py         failure rates, Wilson intervals, McNemar's test, Cohen's kappa, code checks
  agents.py        versions 2 and 3: the agent loop, the router, and agent traces made of spans
  tools.py         tool schemas, and running tool calls against the store
  store.py         the fake store: orders, returns, warranties, tickets
  checks.py        reusable code checks, agent metrics, and CI gates
  judge.py         LLM judges, TPR and TNR, the Rogan-Gladen correction, the swap test, groundedness
  retrieval_metrics.py  hit rate, recall, precision, MRR, nDCG
  phoenix_export.py     optional: send traces to Arize Phoenix
  llm.py           one small wrapper around LiteLLM
docs/            the product manuals, policies, and FAQ the bot answers from
data/            test questions, traces, and exercise files
rubric/          the rubric template, with a worked example
scripts/         generate live traces; rebuild the data and notebooks
tests/           checks that the materials still match the workshop script
answer_keys/     facilitator only (see below)
```

## About the traces

The questions and the retrieval are real: the History column in every trace is exactly what the v1 bot
sends to the model. The answers in the pre-generated traces are written by hand, so everyone in the room
sees the same planted failures. Those traces are marked `curated` in the Source column.

To see what your own model does with the same questions:

```bash
python scripts/generate_traces.py                 # saves data/my_traces_v1.0.csv
python scripts/generate_traces.py --prompt v1.1   # the prompt with the policy fix
```

Your live traces will fail in different ways from the curated ones. That's a good exercise in itself: code
them the same way and compare.

The agent traces for Modules 4 and 6 work the same way. The tool calls are planned by hand, but they run through
the real agent loop and the real tools, so every tool result and end state is real. The Exercise 8 traces are
built from templates and labeled with the "Assumes device" rubric. The production log in Module 6 is synthetic.

To explore agent traces span by span in [Arize Phoenix](https://github.com/Arize-ai/phoenix), install it in a
separate virtual environment (it can pull in a newer OpenAI SDK than LiteLLM supports), start it, and send traces to it:

```bash
pip install arize-phoenix && phoenix serve        # in its own environment; then open http://localhost:6006
pip install arize-phoenix-otel                    # in this repo's environment
python scripts/phoenix_demo.py                    # or --live to run v3 on a few questions first
```

## For facilitators

- **Answer keys** live in `answer_keys/`, which is gitignored so it never reaches the public repo. Bring it
  out during the debrief.
- **Before the workshop,** set `REPO_URL` in `scripts/build_notebooks.py` to the public repo, then run
  `python scripts/build_notebooks.py` so the Colab setup cell clones the right place.
- **If you change the docs or the curated answers,** run `python scripts/build_curated_data.py`, then
  `python -m pytest -q`. The tests check that retrieval and the failure counts still match the slides.
  That script is gitignored like `answer_keys/`, because it contains the Exercise 4a labels.
  For Modules 4 to 6, the same goes for `scripts/build_agent_data.py`, which writes the Exercise 5, 6, 7, and
  10 answer keys.
- **Record the judge runs once,** with a key: `python scripts/record_example_runs.py`, then commit
  `data/example_runs/`. The Module 5 notebook shows that run to anyone without a key, and the swap-test result
  on the M5.14 slide comes from it.
- **Evals in CI:** `python scripts/run_ci_evals.py` exits with an error when a gate fails. On the stored traces
  it fails on purpose, to show that a 90% suite threshold can pass while individual checks fail.
