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


def get_chat_model(model: str | None = None, **kwargs):
    """Main chat model. Pass `model` to override the configured default."""
    kwargs.setdefault("temperature", DEFAULT_TEMPERATURE)
    return init_chat_model(model or DEFAULT_MODEL, **kwargs)


def get_summarizer_model(**kwargs):
    """Cheaper model for summarisation/compaction. Falls back to the default."""
    kwargs.setdefault("temperature", DEFAULT_TEMPERATURE)
    return init_chat_model(SUMMARIZER_MODEL, **kwargs)
