"""Checks for Modules 4 to 6: the store and tools, the agent loop, judges, and the exercise data.

None of these call a model.
"""

import json

import pandas as pd
import pytest

from beefcake import checks, store
from beefcake.agents import (AgentTrace, ScriptedModel, answer_v2, answer_v3, load_agent_traces, route,
                             scripted_router)
from beefcake.judge import _parse_pick, _parse_verdict, corrected_pass_rate, rogan_gladen, swap_test, tpr_tnr
from beefcake.retrieval import BM25Retriever, load_chunks
from beefcake.retrieval_metrics import hit_rate, ndcg, precision, recall, reciprocal_rank
from beefcake.tools import run_tool, schema_errors
from beefcake.traces import DATA_DIR

# --- The store and tools ---------------------------------------------------

def test_return_window_and_damaged_label():
    st = store.Store()
    assert "error" in store.start_return("BC-4420", "BeefCake Row", "too big", st)      # 52 days ago
    ok = store.start_return("BC-4452", "BeefCake Bell", "cracked", st)
    assert ok["prepaid_label"] is True
    assert st.snapshot() == {"returns": [{"order_id": "BC-4452", "item": "BeefCake Bell"}], "tickets": 0}


def test_warranty_lengths_follow_the_policy():
    assert store.check_warranty("ROW-2025-00318")["in_warranty"] is True
    assert store.check_warranty("ROW-2024-00871")["in_warranty"] is False
    assert store.check_warranty("PULSE-2025-00562")["warranty_years"] == 1


def test_schema_errors():
    assert schema_errors("lookup_order", {"order_id": "BC-4417"}) == []
    assert schema_errors("lookup_order", {"order_id": "sam.k@example.com"})
    assert schema_errors("start_return", {"order_id": "BC-4417", "item": "BeefCake Bell"})  # missing reason
    assert schema_errors("issue_refund", {}) == ["'issue_refund' isn't an approved tool"]


def test_unknown_tool_is_refused():
    assert "error" in run_tool("issue_refund", {"order_id": "BC-4420"})


# --- The agent loop ----------------------------------------------------------

def test_v2_runs_tools_and_records_the_end_state():
    trace = answer_v2("return my bell from BC-4417", chat_fn=ScriptedModel([
        {"tool_calls": [("start_return", {"order_id": "BC-4417", "item": "BeefCake Bell", "reason": "heavy"})]},
        {"content": "Done."},
    ]), st=store.Store())
    assert [s["name"] for s in trace.tool_calls] == ["start_return"]
    assert trace.end_state["returns"] == [{"order_id": "BC-4417", "item": "BeefCake Bell"}]
    assert trace.final_answer == "Done."


def test_max_turns_stops_a_runaway_agent():
    loop = [{"tool_calls": [("search_docs", {"query": "x"})]}] * 20
    trace = answer_v2("?", chat_fn=ScriptedModel(loop), st=store.Store())
    assert any(s["name"] == "MaxTurnsExceeded" for s in trace.spans)
    assert trace.end_state["tickets"] == 1  # the "passed to our support team" reply is true


def test_v3_records_router_and_handoff():
    trace = answer_v3("why did coach charge me twice", chat_fn=ScriptedModel([{"content": "ok"}]),
                      complete_fn=scripted_router("orders_billing"), st=store.Store())
    assert [s["kind"] for s in trace.spans[:2]] == ["router", "handoff"]
    assert checks.routed_correctly(trace, "orders_billing")


def test_specialist_cannot_run_tools_it_was_not_given():
    trace = answer_v3("return my bell", chat_fn=ScriptedModel([
        {"tool_calls": [("start_return", {"order_id": "BC-4417", "item": "BeefCake Bell", "reason": "heavy"})]},
        {"content": "Done."},
    ]), complete_fn=scripted_router("device_support"), st=store.Store())
    assert "error" in trace.tool_calls[0]["output"]
    assert trace.end_state["returns"] == []


def test_router_parses_replies():
    assert route("x", complete_fn=lambda m, model=None: "orders_billing") == "orders_billing"
    assert route("x", complete_fn=lambda m, model=None: " Device_Support\n") == "device_support"


# --- Exercise data matches the script -----------------------------------------

def test_blame_game_rounds():
    b1, b2, b3 = load_agent_traces("exercise5_blame_game.jsonl")
    # No comments naming the broken piece: this file is public, and Exercise 5 is a game.
    assert b1.tool_calls[0]["output"]["chunk_ids"][0].startswith("pulse_manual#")
    assert b2.tool_calls[0]["output"]["chunk_ids"][0] == "policies#warranty"
    assert "3-year" in b2.final_answer
    assert b3.tool_calls[0]["input"] == {"order_id": "sam.k@example.com"}


def test_exercise7_checks():
    traces = {t.trace_id: t for t in load_agent_traces("exercise7_tool_calls.jsonl")}
    expected = pd.read_csv(DATA_DIR / "exercise7_expected.csv").set_index("Trace ID")
    def end_ok(tid):
        e = expected.loc[tid]
        return checks.end_state_matches(traces[tid], json.loads(e["Expected returns"]), int(e["Expected tickets"]))
    assert not checks.arguments_match_schema(traces["C02"])
    assert end_ok("C04") and checks.first_action_tool(traces["C04"]) == "lookup_order"   # valid alternate path
    assert not checks.only_approved_tools(traces["C05"])
    assert not end_ok("C07") and not end_ok("C11")
    assert end_ok("C10") and not checks.no_false_success(traces["C10"])                   # only the claim is wrong
    for tid in ["C01", "C03", "C06", "C08", "C09", "C12"]:
        assert end_ok(tid) and checks.no_false_success(traces[tid]) and checks.only_approved_tools(traces[tid])


def test_false_success_regex():
    def t(text, returns=()):
        return checks.no_false_success(AgentTrace("x", "q", "v2", final_answer=text,
                                                  end_state={"returns": list(returns), "tickets": 0}))
    assert not t("I've issued a full refund for your Row.")
    assert not t("I’ve issued a full refund for your Row.")  # curly apostrophe
    assert not t("Your return has been started.")
    assert t("I haven't started a return yet. Want me to?")
    assert t("We refund your original payment method within 10 business days.")
    assert t("I've started your return (R-001).", returns=[{"order_id": "BC-4417", "item": "BeefCake Bell"}])


def test_warranty_check_reads_each_product():
    assert checks.warranty_matches_policy(
        "The Row and the Bell have a 2-year warranty, and the Pulse has a 1-year warranty.")
    assert not checks.warranty_matches_policy("The BeefCake Row comes with a 3-year warranty.")
    assert not checks.warranty_matches_policy("Your rower has a 1-year warranty.")
    assert checks.warranty_matches_policy("It's covered by our 2-year warranty.")


def test_trajectory_efficiency_never_rewards_a_skipped_step():
    one_call = AgentTrace("x", "q", "v2", spans=[{"kind": "tool", "name": "lookup_order"}])
    assert checks.trajectory_efficiency(one_call, 2) is None
    assert checks.trajectory_efficiency(one_call, 1) == 1.0
    assert checks.trajectory_efficiency(AgentTrace("x", "q", "v2"), 0) == 1.0


def test_exercise10_trace_matches_the_script():
    m1 = load_agent_traces("v3_traces.jsonl")[0]
    assert m1.user_query == "coach charged me twice after I returned the rower"
    assert checks.routed_correctly(m1, "device_support")


def test_exercise8_split_and_labels():
    df = pd.read_csv(DATA_DIR / "exercise8_labeled_traces.csv")
    assert len(df) >= 100
    assert set(df["Split"]) == {"train", "dev", "test"}
    shares = df["Split"].value_counts(normalize=True)
    assert 0.07 < shares["train"] < 0.15 and 0.35 < shares["dev"] < 0.45
    assert 0.25 < (df["Assumes device"] == "FAIL").mean() < 0.5
    # The judge's few-shot examples must not appear in the data
    assert not df["User Query"].isin(["it won't turn on", "how do I reset it?", "the battery dies really fast"]).any()


def test_ci_demo_shows_the_suite_threshold_gotcha():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ci", DATA_DIR.parent / "scripts" / "run_ci_evals.py")
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    report = ci.run()
    assert report["suite gate"] == "PASS" and report["ship it"] is False
    assert ci.run(prompt_version="v1.0")["ship it"] is False  # the stored v1.0 file is traces_v1.csv


# --- Retrieval metrics ----------------------------------------------------------

def test_retrieval_metrics():
    ranked = ["a", "b", "c"]
    assert hit_rate(ranked, {"c"}, 3) == 1 and hit_rate(ranked, {"c"}, 2) == 0
    assert reciprocal_rank(ranked, {"b"}) == 0.5
    assert precision(ranked, {"a", "c"}, 3) == pytest.approx(2 / 3)
    assert recall(ranked, {"a", "z"}, 3) == 0.5
    assert ndcg(["a", "b"], {"a": 2, "b": 1}, 2) == 1.0


def test_retrieval_metrics_count_each_section_once():
    # Split chunks can put the same section in the ranking twice
    assert ndcg(["s", "s", "x"], {"s": 1}, 3) == 1.0
    assert precision(["s", "s", "x"], {"s"}, 3) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "x", "s"], {"s"}) == 0.5


def test_smaller_chunks_keep_their_section():
    small = load_chunks(max_words=20)
    assert len(small) > len(load_chunks())
    assert all(c.section_id in {x.id for x in load_chunks()} for c in small)
    assert BM25Retriever(small).retrieve("warranty on the pulse")[0].section_id == "policies#warranty"


# --- Judges ------------------------------------------------------------------------

def test_rogan_gladen_worked_example():
    assert rogan_gladen(0.80, 0.89, 0.87) == pytest.approx(0.8816, abs=1e-3)
    with pytest.raises(ValueError):
        rogan_gladen(0.8, 0.5, 0.5)


def test_tpr_tnr_always_pass_judge():
    human = pd.Series(["PASS", "PASS", "FAIL", "FAIL"])
    rates = tpr_tnr(human, pd.Series(["PASS"] * 4))
    assert rates["TPR"] == 1.0 and rates["TNR"] == 0.0


def test_verdict_parsing():
    assert _parse_verdict('{"reasoning": "asked first", "judgment": "PASS"}')["judgment"] == "PASS"
    assert _parse_verdict("Reasoning... so the answer is FAIL")["judgment"] == "FAIL"
    assert _parse_verdict("no idea")["judgment"] == "ERROR"
    assert _parse_verdict('{"reasoning": "ok", "judgment": "FAIL"}\nNote: {n/a}')["judgment"] == "FAIL"


def test_judge_errors_are_reported():
    rates = tpr_tnr(pd.Series(["PASS", "FAIL"]), pd.Series(["PASS", "ERROR"]))
    assert rates["TNR"] == 0.0 and rates["judge errors"] == 1
    assert "judge errors" not in tpr_tnr(pd.Series(["PASS"]), pd.Series(["PASS"]))


def test_pairwise_pick_parsing():
    assert _parse_pick("1\n\nReply 1 is better because it mentions the 2-hour charge.") == "1"
    assert _parse_pick("**2**") == "2"
    assert _parse_pick("Reply 2 is better: the 1-year warranty is right.") == "2"
    assert _parse_pick("Both give the 2-hour charge time, but the second is kinder.") == "?"


def test_groundedness_survives_messy_replies(monkeypatch):
    from beefcake import judge, llm
    monkeypatch.setattr(llm, "complete", lambda messages, model=None:
                        '{"claims": [{"claim": "2-year warranty", "supported": true}]}\nNote: {see above}')
    assert judge.groundedness("answer", "docs", model="x")["score"] == 1.0
    monkeypatch.setattr(llm, "complete", lambda messages, model=None: "No JSON here.")
    assert "error" in judge.groundedness("answer", "docs", model="x")


def test_corrected_pass_rate_interval():
    human = pd.Series(["PASS"] * 40 + ["FAIL"] * 20)
    judge = pd.Series(["PASS"] * 36 + ["FAIL"] * 4 + ["FAIL"] * 17 + ["PASS"] * 3)
    result = corrected_pass_rate(human, judge, pd.Series(["PASS"] * 80 + ["FAIL"] * 20), n_boot=300)
    low, high = result["95% interval"]
    assert low <= result["corrected"] <= high


def test_swap_test_detects_position_bias():
    pairs = pd.DataFrame({"Question": ["q"] * 3, "Answer A": ["a"] * 3, "Answer B": ["b"] * 3})
    always_first = lambda q, x, y: "1"  # noqa: E731
    assert not swap_test(pairs, judge_fn=always_first)["Consistent"].any()
    picks_a = lambda q, x, y: "1" if x == "a" else "2"  # noqa: E731
    assert swap_test(pairs, judge_fn=picks_a)["Consistent"].all()


def test_agent_trace_round_trip(tmp_path):
    from beefcake.agents import save_agent_traces
    t = AgentTrace("X", "hi", "v2", final_answer="hello")
    save_agent_traces([t], tmp_path / "t.jsonl")
    assert load_agent_traces(tmp_path / "t.jsonl")[0].final_answer == "hello"


def test_llm_chat_parses_tool_calls(monkeypatch):
    from types import SimpleNamespace as NS

    import litellm

    from beefcake import llm
    fake = NS(choices=[NS(message=NS(content=None, tool_calls=[
        NS(id="c1", function=NS(name="lookup_order", arguments='{"order_id": "BC-4417"}'))]))], usage=None)
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake)
    reply = llm.chat([{"role": "user", "content": "hi"}], tools=[], model="openai/gpt-4o-mini")
    assert reply == {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "c1", "name": "lookup_order", "arguments": '{"order_id": "BC-4417"}'}]}


def test_live_v2_loop_with_fake_litellm(monkeypatch):
    """The real llm.chat path, driven by a fake LiteLLM that asks for one tool and then answers."""
    from types import SimpleNamespace as NS

    import litellm
    replies = iter([
        NS(choices=[NS(message=NS(content="", tool_calls=[
            NS(id="c1", function=NS(name="lookup_order", arguments='{"order_id": "BC-4417"}'))]))], usage=None),
        NS(choices=[NS(message=NS(content="Delivered on Sept 20.", tool_calls=None))], usage=None),
    ])
    seen = []
    def fake(**kw):
        seen.append(kw["messages"])
        return next(replies)
    monkeypatch.setattr(litellm, "completion", fake)
    trace = answer_v2("where's BC-4417?", model="openai/gpt-4o-mini", st=store.Store())
    assert trace.final_answer == "Delivered on Sept 20."
    assert seen[1][-1]["role"] == "tool" and "BC-4417" in seen[1][-1]["content"]
