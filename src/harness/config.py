import os

from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "HARNESS_MODEL",
    "openai:gpt-5.6-luna",
)

SUMMARIZER_MODEL = os.getenv(
    "HARNESS_SUMMARIZER",
    DEFAULT_MODEL,
)

DEFAULT_TEMPERATURE = float(
    os.getenv("HARNESS_TEMPERATURE", "0")
)

DEFAULT_REASONING_EFFORT = os.getenv(
    "HARNESS_REASONING_EFFORT",
    "none",
)

PROMPT_CACHE_ENABLED = (
    os.getenv(
        "HARNESS_PROMPT_CACHE_ENABLED",
        "true",
    ).lower()
    == "true"
)

PROMPT_CACHE_KEY = os.getenv(
    "HARNESS_PROMPT_CACHE_KEY",
    "harness-v1",
)


def _defaults(model_id: str, kwargs: dict) -> dict:
    kwargs.setdefault(
        "temperature",
        DEFAULT_TEMPERATURE,
    )

    kwargs.setdefault(
        "stream_usage",
        True,
    )

    if (
        DEFAULT_REASONING_EFFORT
        and "gpt-5" in model_id
    ):
        kwargs.setdefault(
            "reasoning_effort",
            DEFAULT_REASONING_EFFORT,
        )

    return kwargs


def get_chat_model(
    model: str | None = None,
    **kwargs,
):
    model_id = model or DEFAULT_MODEL

    kwargs = _defaults(
        model_id,
        kwargs,
    )

    return init_chat_model(
        model_id,
        **kwargs,
    )


def get_summarizer_model(**kwargs):
    model_id = SUMMARIZER_MODEL

    kwargs = _defaults(
        model_id,
        kwargs,
    )

    return init_chat_model(
        model_id,
        **kwargs,
    )


def get_prompt_cache_key() -> str | None:
    if not PROMPT_CACHE_ENABLED:
        return None

    return PROMPT_CACHE_KEY


def get_prompt_cache_options(model: str | None = None) -> dict[str, Any]:
    """Return provider-specific prompt-caching kwargs when supported.

    OpenAI prompt caching is automatic for supported models/accounts; it is not
    activated by a custom `prompt_cache_key` in the installed SDK stack here.
    Anthropic-style prompt caching uses a message-level `cache_control` hint.
    """
    if not PROMPT_CACHE_ENABLED:
        return {}

    model_id = (model or DEFAULT_MODEL).lower()

    if "anthropic" in model_id:
        return {"cache_control": {"type": "ephemeral"}}

    if "openai" in model_id:
        return {}

    return {}