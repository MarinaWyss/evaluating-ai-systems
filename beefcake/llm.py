"""One small wrapper around LiteLLM, so the workshop works with any provider.

Set ONE of these environment variables (or put it in a .env file):
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY

The bot picks a default model for whichever key it finds. To choose a model
yourself, set BEEFCAKE_MODEL (the bot) and BEEFCAKE_JUDGE_MODEL (judges),
using LiteLLM's "provider/model" names, for example "openai/gpt-4o-mini".
"""

from __future__ import annotations

import os
from pathlib import Path

# Default model per provider. Model names change often: if one of these has been
# retired, set BEEFCAKE_MODEL instead of editing this file.
DEFAULT_MODELS = [
    ("OPENAI_API_KEY", "openai/gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic/claude-haiku-4-5"),
    ("GEMINI_API_KEY", "gemini/gemini-3.5-flash-lite"),
]


class NoAPIKeyError(RuntimeError):
    """Raised when a live model call is attempted without any API key."""


def _load_dotenv() -> None:
    """Read KEY=value lines from a .env file in the repo root, if there is one."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def has_api_key() -> bool:
    return any(os.environ.get(key) for key, _ in DEFAULT_MODELS)


def get_model(role: str = "bot") -> str:
    """Return the model name for the bot or the judge."""
    override = os.environ.get("BEEFCAKE_JUDGE_MODEL" if role == "judge" else "BEEFCAKE_MODEL")
    if override:
        return override
    if role == "judge" and os.environ.get("BEEFCAKE_MODEL"):
        return os.environ["BEEFCAKE_MODEL"]
    for key, model in DEFAULT_MODELS:
        if os.environ.get(key):
            return model
    raise NoAPIKeyError(
        "No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY "
        "(see README). Every exercise also has pre-generated data, so you can keep going without one."
    )


def load_colab_secrets() -> None:
    """In Google Colab, copy API keys from the Secrets panel (the key icon) into the environment."""
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return
    for key, _ in DEFAULT_MODELS:
        try:
            value = userdata.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = value


# Latency and cost of the most recent call, for comparing models (Exercise 4b stretch).
LAST_CALL: dict = {}


def complete(messages: list[dict], model: str | None = None, temperature: float = 0.0, **kwargs) -> str:
    """Send chat messages to the model and return the text of its reply."""
    model = model or get_model("bot")
    import time

    import litellm  # imported here so the rest of the package works without it

    # Newer models (GPT-6, Claude Sonnet 5 and Opus 5.5) reject temperature=0. drop_params tells
    # LiteLLM to leave out any setting a model doesn't support instead of raising an error.
    kwargs.setdefault("drop_params", True)
    start = time.perf_counter()
    response = litellm.completion(model=model, messages=messages, temperature=temperature, **kwargs)
    LAST_CALL.clear()
    LAST_CALL["model"] = model
    LAST_CALL["latency_s"] = round(time.perf_counter() - start, 2)
    try:
        LAST_CALL["cost_usd"] = litellm.completion_cost(completion_response=response)
    except Exception:
        LAST_CALL["cost_usd"] = None
    return response.choices[0].message.content or ""
