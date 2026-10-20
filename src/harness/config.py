from typing import Any
from langchain.chat_models import init_chat_model
from .settings import settings


def _defaults(model_id: str, kwargs: dict) -> dict:
    # gpt-6-luna rejects temperature=0; only the default value 1 is valid.
    # Tools plus reasoning_effort also require the Responses API, not chat completions.
    if "gpt-6" in model_id:
        kwargs.setdefault("temperature", 1)
        kwargs.setdefault("use_responses_api", True)
    else:
        kwargs.setdefault("temperature", settings.temperature)
    kwargs.setdefault("stream_usage", True)

    if "gpt-5" in model_id or "gpt-6" in model_id:
        kwargs.setdefault("reasoning_effort", settings.reasoning_effort)

    if model_id.startswith("openai:") and settings.openai_api_key:
        kwargs.setdefault("api_key", settings.openai_api_key)

    return kwargs


def _build(model_id: str, **kwargs):
    return init_chat_model(model_id, **_defaults(model_id, kwargs))


def get_chat_model(model: str | None = None, **kwargs):
    return _build(model or settings.model, **kwargs)


def get_summarizer_model(**kwargs):
    return _build(settings.model, **kwargs)


def get_routing_model(**kwargs):
    return _build(settings.routing_model, **kwargs)


def get_prompt_cache_key() -> str | None:
    return settings.prompt_cache_key if settings.prompt_cache_enabled else None


def get_prompt_cache_options(model: str | None = None) -> dict[str, Any]:
    if not settings.prompt_cache_enabled:
        return {}
    if "anthropic" in (model or settings.model).lower():
        return {"cache_control": {"type": "ephemeral"}}
    return {}