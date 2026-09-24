"""Checks that the workshop materials still behave the way the script says they do.

Run with:  python -m pytest -q
None of these call a model, so they run without an API key.
"""

import pandas as pd
import pytest

from beefcake.bot import PROMPTS, build_messages
from beefcake.evals import (
    FAILURE_MODES, cohens_kappa, failure_rates, mcnemar_exact, raw_agreement, run_checks, wilson_interval,
)
from beefcake.retrieval import load_chunks, retrieve
from beefcake.traces import DATA_DIR, load_traces


# --- Retrieval behaves the way the slides describe -------------------------

def test_disconnecting_pulls_the_pulse_chunk_first():
    # M2.9: "it keeps disconnecting" retrieves the Pulse section, not the Row's Coach-app section
    top = retrieve("it keeps disconnecting")[0]
    assert top.id.startswith("pulse_manual#")


def test_warranty_question_retrieves_the_warranty_policy():
    # Blame game round 2: the right chunk is retrieved, so a wrong answer is a generation failure
    assert retrieve("how long is the warranty on the rower")[0].id == "policies#warranty"


def test_every_chunk_has_text():
    assert all(c.text.strip() for c in load_chunks())


def test_prompt_versions_differ_only_by_the_policy_rule():
    assert PROMPTS["v1.1"].startswith(PROMPTS["v1.0"])


def test_history_contains_the_chunks():
    msgs = build_messages("how do i cancel coach", retrieve("how do i cancel coach"))
    assert "Cancel subscription" in msgs[1]["content"]


# --- The planted counts match the script (M2.12, M3.15, M3.16) ---------------

def test_v1_counts_match_the_script():
    df = load_traces("traces_v1_labeled.csv")
    counts = {m: int((df[m] == "FAIL").sum()) for m in FAILURE_MODES}
    assert counts == {
        "Invents policy": 8, "Doesn't collect info": 4, "Assumes device": 3,
        "Made-up steps": 3, "Leaks RAG setup": 2, "Out of scope": 1,
    }
    assert len(df) == 25


def test_prompt_fix_halves_invented_policy_and_breaks_nothing():
    before = load_traces("traces_v1_labeled.csv")
    after = load_traces("traces_v1.1_labeled.csv")
    result = mcnemar_exact(before["Invents policy"], after["Invents policy"])
    assert result == {"fixed": 4, "broke": 0, "p_value": 0.125}


def test_provided_context_check_catches_both_leaks():
    df = load_traces("traces_v1_labeled.csv")
    checked = run_checks(df, {"check": lambda r: "provided context" not in r["AI Response"].lower()})
    assert (checked["check"] == checked["Leaks RAG setup"]).all()


def test_exercise_files_have_the_right_sizes():
    assert len(load_traces("exercise2_traces.csv")) == 20
    assert len(load_traces("exercise4a_traces.csv")) == 12


def test_answer_key_covers_exercise_4a():
    key = DATA_DIR.parent / "answer_keys" / "exercise4a_key.csv"
    if not key.exists():
        pytest.skip("answer_keys/ is facilitator only, so it isn't in the public repo")
    assert len(pd.read_csv(key)) == 12


# --- Statistics match the numbers on the slides ------------------------------

def test_wilson_intervals_match_m3_16():
    low, high = wilson_interval(8, 25)
    assert (round(low, 3), round(high, 3)) == (0.172, 0.516)
    low, high = wilson_interval(4, 25)
    assert (round(low, 3), round(high, 3)) == (0.064, 0.347)


def test_kappa_chance_example_from_m3_7():
    # Two reviewers who each say PASS 90% of the time, independently: raw agreement is high, kappa is ~0
    a = pd.Series(["PASS"] * 9 + ["FAIL"])
    b = pd.Series(["PASS"] * 8 + ["FAIL", "PASS"])
    assert raw_agreement(a, b) == pytest.approx(0.8)
    assert cohens_kappa(a, b) < 0


def test_kappa_perfect_agreement():
    s = pd.Series(["PASS", "FAIL", "PASS"])
    assert cohens_kappa(s, s) == 1.0


def test_failure_rates_table():
    df = load_traces("traces_v1_labeled.csv")
    table = failure_rates(df)
    row = table[table["Failure mode"] == "Invents policy"].iloc[0]
    assert row["Fails"] == 8 and row["Rate"] == "32%"


# --- The live code path works (LiteLLM's mock_response, so no key is used) ---

def test_llm_complete_with_mock(monkeypatch):
    from beefcake import llm
    text = llm.complete([{"role": "user", "content": "hi"}], model="openai/gpt-4o-mini", mock_response="hello")
    assert text == "hello"
    assert "latency_s" in llm.LAST_CALL


def test_bot_answer_builds_a_full_trace(monkeypatch):
    from beefcake import bot, llm
    monkeypatch.setattr(llm, "complete", lambda messages, model=None: "mocked answer")
    trace = bot.answer("it keeps disconnecting", model="openai/gpt-4o-mini", trace_id="X1")
    assert trace.ai_response == "mocked answer"
    assert trace.retrieved_ids[0].startswith("pulse_manual#")
    assert set(trace.to_row()) == {"Trace ID", "Query Topic", "User Query", "History", "AI Response",
                                   "Retrieved Chunks", "Prompt Version", "Model", "Source"}


def test_classifier_parses_labels(monkeypatch):
    from beefcake import classifier, llm
    monkeypatch.setattr(llm, "complete", lambda messages, model=None: " Beefcake_Bell\n")
    assert classifier.classify_device("dial stuck") == "beefcake_bell"
    monkeypatch.setattr(llm, "complete", lambda messages, model=None: "I think it's the rower")
    assert classifier.classify_device("??") == "unknown"


def test_model_selection_follows_the_key(monkeypatch):
    from beefcake import llm
    for key, _ in llm.DEFAULT_MODELS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("BEEFCAKE_MODEL", raising=False)
    monkeypatch.delenv("BEEFCAKE_JUDGE_MODEL", raising=False)
    assert not llm.has_api_key()
    with pytest.raises(llm.NoAPIKeyError):
        llm.get_model()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.get_model().startswith("anthropic/")
    monkeypatch.setenv("BEEFCAKE_MODEL", "gemini/some-model")
    assert llm.get_model() == "gemini/some-model" and llm.get_model("judge") == "gemini/some-model"
