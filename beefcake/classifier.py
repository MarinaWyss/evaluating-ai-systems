"""A small LLM classifier: which BeefCake product is the customer asking about?

There's exactly one right answer for each question, which makes this a good
candidate for a reference-based eval (Module 3).
"""

from __future__ import annotations

from . import llm

LABELS = ["beefcake_row", "beefcake_bell", "beefcake_pulse", "beefcake_coach", "unknown"]

CLASSIFIER_PROMPT = """Classify which BeefCake product the customer is asking about.

Products:
- beefcake_row: the BeefCake Row, a smart rowing machine with a screen
- beefcake_bell: the BeefCake Bell, an app-connected adjustable kettlebell
- beefcake_pulse: the BeefCake Pulse, a heart-rate chest strap
- beefcake_coach: the BeefCake Coach subscription app, including billing and login
- unknown: the question doesn't make clear which product, or it's about none of them

Reply with only the label."""


def classify_device(query: str, model: str | None = None) -> str:
    messages = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": query},
    ]
    reply = llm.complete(messages, model=model).strip().lower()
    for label in LABELS:
        if label in reply:
            return label
    return "unknown"
