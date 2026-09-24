"""Versions 2 and 3 of the BeefCake support bot.

v2 (Module 4): one agent with tools. It searches the docs, looks up orders,
starts returns, checks warranties, and can hand off to a human (create_ticket).

v3 (Module 6): a router sends each question to one of two specialist agents.
    Device Support: setup and troubleshooting. Tools: search_docs, create_ticket.
    Orders and Billing: orders, returns, warranty, Coach billing. All tools.

Every run is recorded as an AgentTrace: a list of spans, one per step (a router
decision, a handoff, a model call, a tool call), plus the final answer and the
store's end state.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import llm, store
from .tools import openai_tools, parse_arguments, run_tool

MAX_TURNS = 10  # like the OpenAI Agents SDK default: stop a runaway agent

COMPANY = (
    "BeefCake Fitness is a home-gym company that sells the BeefCake Row (a smart rowing machine), "
    "the BeefCake Bell (an app-connected adjustable kettlebell), the BeefCake Pulse (a heart-rate "
    "chest strap), and the BeefCake Coach subscription app."
)

V2_PROMPT = (
    f"You are a friendly customer support agent for BeefCake Fitness. {COMPANY} "
    "Use search_docs to look up product help and policies before answering. Use the order, return, "
    "and warranty tools when the customer asks about an order or a device they own. If you can't "
    "resolve the problem, use create_ticket to hand it to a person."
)

DEVICE_SUPPORT_PROMPT = (
    f"You are the Device Support agent for BeefCake Fitness. {COMPANY} "
    "You help with setup and troubleshooting for the Row, the Bell, and the Pulse. Use search_docs "
    "before answering. If you can't resolve the problem, use create_ticket to hand it to a person."
)

ORDERS_BILLING_PROMPT = (
    f"You are the Orders and Billing agent for BeefCake Fitness. {COMPANY} "
    "You help with orders, returns, warranties, and Coach subscription billing. Use search_docs to "
    "check policies, and the order, return, and warranty tools for anything about a specific order "
    "or device. If you can't resolve the problem, use create_ticket to hand it to a person."
)

ROUTER_PROMPT = """You route BeefCake Fitness support questions to one of two agents.

device_support: setting up, using, pairing, or troubleshooting the Row, the Bell, or the Pulse.
orders_billing: orders, deliveries, returns, refunds, warranties, and Coach subscription billing.

Reply with only the agent name."""

AGENTS = {
    "device_support": {"prompt": DEVICE_SUPPORT_PROMPT, "tools": ["search_docs", "create_ticket"]},
    "orders_billing": {"prompt": ORDERS_BILLING_PROMPT,
                       "tools": ["search_docs", "lookup_order", "start_return", "check_warranty", "create_ticket"]},
}


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

@dataclass
class AgentTrace:
    trace_id: str
    user_query: str
    version: str                      # "v2" or "v3"
    spans: list[dict] = field(default_factory=list)
    final_answer: str = ""
    end_state: dict = field(default_factory=dict)
    model: str = ""
    source: str = "live"              # "live" or "curated"

    @property
    def tool_calls(self) -> list[dict]:
        return [s for s in self.spans if s["kind"] == "tool"]

    def add_span(self, kind: str, agent: str, name: str, input, output, latency_s: float = 0.0, **extra) -> dict:
        span = {"span_id": len(self.spans) + 1, "kind": kind, "agent": agent, "name": name,
                "input": input, "output": output, "latency_s": latency_s, **extra}
        self.spans.append(span)
        return span


def save_agent_traces(traces: list[AgentTrace], path: str | Path) -> None:
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(asdict(t)) + "\n")


def load_agent_traces(path: str | Path) -> list[AgentTrace]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent.parent / "data" / path
    with open(path) as f:
        return [AgentTrace(**json.loads(line)) for line in f if line.strip()]


def show_agent_trace(trace: AgentTrace, full: bool = False) -> None:
    """Print a trace span by span. full=True also prints the long tool outputs (like doc search results)."""
    bar = "=" * 80
    print(bar)
    print(f"Trace {trace.trace_id} ({trace.version})")
    print(f"USER: {trace.user_query}")
    print(bar)
    for s in trace.spans:
        head = f"[span {s['span_id']}] {s['kind'].upper():8} agent={s['agent']:15} {s['name']}"
        print(head)
        if s["kind"] == "tool":
            print(f"    arguments: {json.dumps(s['input'])}")
            out = s["output"]
            if s["name"] == "search_docs" and not full and isinstance(out, dict) and "chunk_ids" in out:
                out = {"chunk_ids": out["chunk_ids"], "results": "(hidden, use full=True)"}
            print(f"    result:    {json.dumps(out)}")
        elif s["kind"] in ("router", "handoff"):
            print(f"    {s['output']}")
        elif s["kind"] == "llm" and s["output"]:
            print(f"    text: {s['output']}")
        elif s["kind"] == "error":
            print(f"    {s['output']}")
    print(bar)
    print(f"FINAL ANSWER:\n{trace.final_answer}")
    print(bar)
    print(f"END STATE: {trace.end_state}")


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

ChatFn = Callable[..., dict]


def run_agent(trace: AgentTrace, agent_name: str, system_prompt: str, tool_names: list[str],
              messages: list[dict], chat_fn: ChatFn | None = None, model: str | None = None,
              st: store.Store | None = None, max_turns: int = MAX_TURNS) -> str:
    """Call the model, run any tools it asks for, and repeat until it answers in text."""
    chat_fn = chat_fn or llm.chat
    st = st or store.STORE
    tools = openai_tools(tool_names)
    convo = [{"role": "system", "content": system_prompt}] + messages
    for _ in range(max_turns):
        start = time.perf_counter()
        reply = chat_fn(convo, tools=tools, model=model)
        trace.add_span("llm", agent_name, "model call", None, reply["content"],
                       round(time.perf_counter() - start, 2), tokens=llm.LAST_CALL.get("tokens"))
        if not reply["tool_calls"]:
            return reply["content"]
        convo.append({
            "role": "assistant", "content": reply["content"] or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}} for c in reply["tool_calls"]],
        })
        for call in reply["tool_calls"]:
            args = parse_arguments(call["arguments"])
            start = time.perf_counter()
            result = run_tool(call["name"], args, st)
            trace.add_span("tool", agent_name, call["name"], args, result, round(time.perf_counter() - start, 3))
            convo.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    trace.add_span("error", agent_name, "MaxTurnsExceeded", None, f"Stopped after {max_turns} turns")
    # Hand off for real, so the reply below is true.
    summary = f"Agent stopped after {max_turns} turns: {messages[-1]['content']}"
    ticket = run_tool("create_ticket", {"summary": summary}, st)
    trace.add_span("tool", agent_name, "create_ticket", {"summary": summary}, ticket)
    return "Sorry, I couldn't finish that. I've passed your question to our support team."


def answer_v2(query: str, chat_fn: ChatFn | None = None, model: str | None = None,
              st: store.Store | None = None, trace_id: str | None = None) -> AgentTrace:
    """Version 2: one agent with tools."""
    st = st or store.STORE
    st.reset()
    trace = AgentTrace(trace_id or uuid.uuid4().hex[:8], query, "v2",
                       model=model or ("scripted" if chat_fn else llm.get_model("bot")))
    trace.final_answer = run_agent(trace, "support", V2_PROMPT, list(AGENTS["orders_billing"]["tools"]),
                                   [{"role": "user", "content": query}], chat_fn, model, st)
    trace.end_state = st.snapshot()
    return trace


def route(query: str, complete_fn: Callable[..., str] | None = None, model: str | None = None) -> str:
    complete_fn = complete_fn or llm.complete
    reply = complete_fn([{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": query}],
                        model=model).strip().lower()
    return "orders_billing" if "orders" in reply or "billing" in reply else "device_support"


def answer_v3(query: str, chat_fn: ChatFn | None = None, complete_fn: Callable[..., str] | None = None,
              model: str | None = None, st: store.Store | None = None, trace_id: str | None = None) -> AgentTrace:
    """Version 3: a router, a handoff, then one specialist agent."""
    st = st or store.STORE
    st.reset()
    trace = AgentTrace(trace_id or uuid.uuid4().hex[:8], query, "v3",
                       model=model or ("scripted" if chat_fn else llm.get_model("bot")))
    start = time.perf_counter()
    agent = route(query, complete_fn, model)
    trace.add_span("router", "router", "route", query, agent, round(time.perf_counter() - start, 2))
    # A handoff is recorded like a tool call, the way the OpenAI Agents SDK does it.
    trace.add_span("handoff", "router", f"transfer_to_{agent}", {"user_query": query}, f"handed off to {agent}")
    cfg = AGENTS[agent]
    trace.final_answer = run_agent(trace, agent, cfg["prompt"], cfg["tools"],
                                   [{"role": "user", "content": query}], chat_fn, model, st)
    trace.end_state = st.snapshot()
    return trace


# ---------------------------------------------------------------------------
# A scripted "model", used to build the curated traces and in tests
# ---------------------------------------------------------------------------

class ScriptedModel:
    """Replays planned assistant turns. Each turn is either
    {"tool_calls": [(name, arguments_dict), ...]} or {"content": "final answer"}.
    The tools themselves really run, so results and end states are real.
    """

    def __init__(self, turns: list[dict]):
        self.turns = list(turns)
        self.n = 0

    def __call__(self, messages, tools=None, model=None) -> dict:
        turn = self.turns[self.n]
        self.n += 1
        calls = [{"id": f"call_{self.n}_{i}", "name": name, "arguments": json.dumps(args)}
                 for i, (name, args) in enumerate(turn.get("tool_calls", []))]
        return {"role": "assistant", "content": turn.get("content", ""), "tool_calls": calls}


def scripted_router(agent: str) -> Callable[..., str]:
    return lambda messages, model=None: agent
