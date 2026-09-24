"""Version 1 of the BeefCake support bot: a plain RAG bot.

It retrieves the top chunks from the docs, pastes them into the prompt, and asks
the model to answer. The v1.0 prompt is deliberately simple, so the bot fails in
the ways real first versions do.
"""

from __future__ import annotations

import uuid

from . import llm
from .retrieval import Chunk, retrieve
from .traces import Trace

PROMPTS = {
    # The starting prompt. Short and friendly, with no rules about policies.
    "v1.0": (
        "You are a friendly customer support assistant for BeefCake Fitness, a home-gym company "
        "that sells the BeefCake Row, the BeefCake Bell, the BeefCake Pulse, and the BeefCake Coach app. "
        "Answer the customer's question using the provided context."
    ),
    # The fix used in Module 3: one added rule about policies.
    "v1.1": (
        "You are a friendly customer support assistant for BeefCake Fitness, a home-gym company "
        "that sells the BeefCake Row, the BeefCake Bell, the BeefCake Pulse, and the BeefCake Coach app. "
        "Answer the customer's question using the provided context. "
        "When a question is about returns, warranties, refunds, or billing, only state policies exactly "
        "as they appear in the context. If the context doesn't cover it, say you're not sure and suggest "
        "contacting BeefCake Support."
    ),
}

DEFAULT_PROMPT_VERSION = "v1.0"


def build_messages(query: str, chunks: list[Chunk], prompt_version: str = DEFAULT_PROMPT_VERSION) -> list[dict]:
    context = "\n\n".join(c.as_context() for c in chunks)
    return [
        {"role": "system", "content": PROMPTS[prompt_version]},
        {"role": "user", "content": f"Context:\n{context}\n\nCustomer question: {query}"},
    ]


def answer(
    query: str,
    query_topic: str = "",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    k: int = 3,
    model: str | None = None,
    trace_id: str | None = None,
) -> Trace:
    """Answer one customer question and return the full trace."""
    chunks = retrieve(query, k=k)
    messages = build_messages(query, chunks, prompt_version)
    model = model or llm.get_model("bot")
    response = llm.complete(messages, model=model)
    return Trace(
        trace_id=trace_id or uuid.uuid4().hex[:8],
        query_topic=query_topic,
        user_query=query,
        history=messages,
        ai_response=response,
        retrieved_ids=[c.id for c in chunks],
        prompt_version=prompt_version,
        model=model,
        source="live",
    )


def chat(query: str, **kwargs) -> str:
    """Convenience for the setup check: ask one question, get the answer text."""
    return answer(query, **kwargs).ai_response
