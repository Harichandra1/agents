"""
Single source of truth for models and tunables.

Every module reads its model from here rather than hardcoding a string.
Week 1 had four different model ids scattered across five files (including a
malformed one that crashed at runtime); centralising them means Day 9's
routing work only has to change this file.

Override any of these from the environment or .env without touching code:

    HARNESS_MODEL=openai:gpt-5.6-luna
    HARNESS_SUMMARIZER=openai:gpt-5.6-luna
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Called exactly once, here. No other module should call load_dotenv().
load_dotenv()

DEFAULT_MODEL = os.getenv("HARNESS_MODEL", "openai:gpt-5.6-luna")

# Summarisation and triage don't need the strongest model. Day 9 (routing)
# points this at something cheaper and measures the cost/quality delta.
SUMMARIZER_MODEL = os.getenv("HARNESS_SUMMARIZER", DEFAULT_MODEL)

DEFAULT_TEMPERATURE = float(os.getenv("HARNESS_TEMPERATURE", "0"))

# Reasoning models (the gpt-5.x family) reject function tools on
# /v1/chat/completions unless reasoning_effort is "none":
#
#   BadRequestError: Function tools with reasoning_effort are not supported
#   for <model> in /v1/chat/completions. To use function tools, use
#   /v1/responses or set reasoning_effort to 'none'.
#
# Every day from Day 1 on binds tools, so this would break all of them. Set
# it once here. Raise it (e.g. "medium") only for a call that binds no tools,
# or switch that call to the Responses API.
DEFAULT_REASONING_EFFORT = os.getenv("HARNESS_REASONING_EFFORT", "none")


def _defaults(model_id: str, kwargs: dict) -> dict:
    kwargs.setdefault("temperature", DEFAULT_TEMPERATURE)
    # Only the reasoning families take this parameter at all.
    if DEFAULT_REASONING_EFFORT and "gpt-5" in model_id:
        kwargs.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    return kwargs


def get_chat_model(model: str | None = None, **kwargs):
    """Main chat model. Pass `model` to override the configured default."""
    model_id = model or DEFAULT_MODEL
    return init_chat_model(model_id, **_defaults(model_id, kwargs))


def get_summarizer_model(**kwargs):
    """Cheaper model for summarisation/compaction. Falls back to the default."""
    return init_chat_model(SUMMARIZER_MODEL, **_defaults(SUMMARIZER_MODEL, kwargs))
