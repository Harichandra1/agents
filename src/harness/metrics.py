"""
LLM performance and prompt-cache metrics.
"""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class LLMMetrics:
    start_time: float
    end_time: float
    latency_ms: float

    ttft_ms: float | None = None

    output_tokens: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0

    generation_time_ms: float | None = None
    tps: float | None = None

    cache_hit: bool = False
    cache_creation_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_usage(response) -> dict[str, int]:
    """
    Normalize LangChain/OpenAI usage metadata.

    The exact field names differ across providers and SDK versions, so we accept
    both the canonical cache_* keys and the older OpenAI aliases.
    """
    usage = getattr(response, "usage_metadata", None) or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    input_details = usage.get(
        "input_token_details",
        {},
    ) or {}
    if not input_details:
        input_details = usage.get(
            "input_tokens_details",
            {},
        ) or {}

    cached_tokens = (
        input_details.get("cache_read")
        or input_details.get("cache_read_tokens")
        or input_details.get("cached_tokens")
        or usage.get("cache_read_tokens")
        or usage.get("cached_tokens")
        or 0
    )

    cache_creation = (
        input_details.get("cache_creation")
        or input_details.get("cache_creation_tokens")
        or input_details.get("cache_write_tokens")
        or usage.get("cache_creation_tokens")
        or usage.get("cache_write_tokens")
        or 0
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation,
    }