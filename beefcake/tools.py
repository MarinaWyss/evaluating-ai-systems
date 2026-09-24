"""The tools the v2 and v3 bots can call, with their JSON schemas.

There's deliberately no refund tool. Refunds happen after a return arrives,
so if you ever see a call to something like issue_refund, the model made it up.
"""

from __future__ import annotations

import json
import re

from . import store
from .retrieval import retrieve

TOOL_SCHEMAS = {
    "search_docs": {
        "description": "Search the BeefCake manuals, policies, and FAQ. Returns the most relevant sections.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
    "lookup_order": {
        "description": "Look up an order by its order ID, for example BC-1234.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "pattern": r"^BC-\d{4}$", "description": "Order ID, like BC-1234"}},
            "required": ["order_id"],
        },
    },
    "start_return": {
        "description": "Start a return for one item in an order. Only works within 30 days of delivery.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "pattern": r"^BC-\d{4}$"},
                "item": {"type": "string", "enum": ["BeefCake Row", "BeefCake Bell", "BeefCake Pulse"]},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "item", "reason"],
        },
    },
    "check_warranty": {
        "description": "Check the warranty for a device by its serial number, for example ROW-2025-00318.",
        "parameters": {
            "type": "object",
            "properties": {"serial_number": {"type": "string", "pattern": r"^(ROW|BELL|PULSE)-\d{4}-\d{5}$"}},
            "required": ["serial_number"],
        },
    },
    "create_ticket": {
        "description": "Hand the conversation to a human on the BeefCake Support team. Use when you can't resolve the issue.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "A short summary for the support agent"}},
            "required": ["summary"],
        },
    },
}

APPROVED_TOOLS = set(TOOL_SCHEMAS)


def openai_tools(names: list[str] | None = None) -> list[dict]:
    """The schemas in the format LiteLLM expects."""
    names = names or list(TOOL_SCHEMAS)
    return [{"type": "function", "function": {"name": n, **TOOL_SCHEMAS[n]}} for n in names]


def schema_errors(name: str, arguments: dict) -> list[str]:
    """Check arguments against the tool's schema: required fields, no extra fields, types, patterns, enums."""
    if name not in TOOL_SCHEMAS:
        return [f"{name!r} isn't an approved tool"]
    params = TOOL_SCHEMAS[name]["parameters"]
    props, required = params["properties"], params.get("required", [])
    errors = [f"missing required argument {r!r}" for r in required if r not in arguments]
    for key, value in arguments.items():
        if key not in props:
            errors.append(f"unexpected argument {key!r}")
            continue
        spec = props[key]
        if spec.get("type") == "string" and not isinstance(value, str):
            errors.append(f"{key!r} should be a string")
            continue
        if "pattern" in spec and not re.fullmatch(spec["pattern"].strip("^$"), str(value)):
            errors.append(f"{key!r}={value!r} doesn't match {spec['pattern']}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key!r}={value!r} isn't one of {spec['enum']}")
    return errors


def run_tool(name: str, arguments: dict, st: store.Store | None = None) -> dict:
    """Execute one tool call against the fake store and return its result."""
    st = st or store.STORE
    if name not in APPROVED_TOOLS:
        return {"error": f"Unknown tool {name!r}."}
    missing = [r for r in TOOL_SCHEMAS[name]["parameters"].get("required", []) if r not in arguments]
    if missing:
        return {"error": f"Missing required argument(s): {', '.join(missing)}."}
    known = {k: v for k, v in arguments.items() if k in TOOL_SCHEMAS[name]["parameters"]["properties"]}
    try:
        if name == "search_docs":
            chunks = retrieve(str(known["query"]), k=3)
            return {"chunk_ids": [c.id for c in chunks], "results": [c.as_context() for c in chunks]}
        return getattr(store, name)(**known, store=st)
    except Exception as e:  # bad arguments come back to the model as an error, like a real API would
        return {"error": f"{name} failed: {e}"}


def parse_arguments(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_unparseable": raw}
    return parsed if isinstance(parsed, dict) else {"_unparseable": raw}
