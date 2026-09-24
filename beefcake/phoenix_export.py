"""Send AgentTraces to Arize Phoenix so you can click through them span by span (M6.7).

Optional: needs `pip install arize-phoenix-otel` and a running Phoenix server:
    pip install arize-phoenix && phoenix serve          # then open http://localhost:6006

Then:
    from beefcake.agents import answer_v3
    from beefcake.phoenix_export import export_traces
    export_traces([answer_v3("can I still return the bell from order BC-4417?")])

Live traces carry real latencies and token counts. Curated traces were never timed,
so they're shown with nominal durations (0.8 s per model call, 0.05 s per tool call).
"""

from __future__ import annotations

import json
import time

from .agents import AgentTrace

NOMINAL_S = {"llm": 0.8, "router": 0.5, "tool": 0.05, "handoff": 0.01, "error": 0.01}
KIND = {"llm": "LLM", "router": "LLM", "tool": "TOOL", "handoff": "TOOL", "error": "CHAIN"}


def check_phoenix(endpoint: str = "http://localhost:6006/v1/traces") -> None:
    """Raise ConnectionError if nothing answers at the endpoint's server. The exporter itself fails quietly."""
    import urllib.error
    import urllib.request
    from urllib.parse import urlsplit

    base = "{0.scheme}://{0.netloc}/".format(urlsplit(endpoint))
    try:
        urllib.request.urlopen(base, timeout=3)
    except urllib.error.HTTPError:
        pass  # it answered, so it's running
    except OSError as e:
        raise ConnectionError(f"Can't reach Phoenix at {base}. Is `phoenix serve` running?") from e


def _tracer(endpoint: str, project: str):
    from phoenix.otel import register

    provider = register(project_name=project, endpoint=endpoint, batch=False, verbose=False)
    return provider, provider.get_tracer("beefcake")


def export_traces(traces: list[AgentTrace], endpoint: str = "http://localhost:6006/v1/traces",
                  project: str = "beefcake-support-bot") -> None:
    check_phoenix(endpoint)
    provider, tracer = _tracer(endpoint, project)
    from opentelemetry import trace as otel

    for tr in traces:
        clock = time.time_ns()

        def duration(span):
            real = span.get("latency_s") or 0
            return int((real if real and tr.source == "live" else NOMINAL_S[span["kind"]]) * 1e9)

        total = sum(duration(s) for s in tr.spans)
        root = tracer.start_span("support request", start_time=clock, attributes={
            "openinference.span.kind": "CHAIN", "input.value": tr.user_query, "output.value": tr.final_answer,
            "metadata": json.dumps({"trace_id": tr.trace_id, "version": tr.version, "source": tr.source,
                                    "end_state": tr.end_state}),
        })
        root_ctx = otel.set_span_in_context(root)
        agent_span, agent_ctx, current_agent, t = None, root_ctx, None, clock
        for s in tr.spans:
            if s["agent"] != current_agent and s["kind"] not in ("router", "handoff"):
                if agent_span:
                    agent_span.end(end_time=t)
                current_agent = s["agent"]
                agent_span = tracer.start_span(f"agent: {current_agent}", context=root_ctx, start_time=t,
                                               attributes={"openinference.span.kind": "AGENT"})
                agent_ctx = otel.set_span_in_context(agent_span)
            parent = root_ctx if s["kind"] in ("router", "handoff") else agent_ctx
            d = duration(s)
            attrs = {"openinference.span.kind": KIND[s["kind"]],
                     "input.value": json.dumps(s["input"]) if s["input"] is not None else "",
                     "output.value": s["output"] if isinstance(s["output"], str) else json.dumps(s["output"])}
            if s["kind"] in ("tool", "handoff"):
                attrs["tool.name"] = s["name"]
                attrs["tool.parameters"] = json.dumps(s["input"])
            if s["kind"] in ("llm", "router"):
                attrs["llm.model_name"] = tr.model
                if s.get("tokens"):
                    attrs["llm.token_count.total"] = s["tokens"]
            span = tracer.start_span(s["name"], context=parent, start_time=t, attributes=attrs)
            span.end(end_time=t + d)
            t += d
        if agent_span:
            agent_span.end(end_time=t)
        root.end(end_time=clock + total)
    provider.force_flush()
