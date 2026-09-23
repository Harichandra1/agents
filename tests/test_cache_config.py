from harness.config import get_prompt_cache_key, get_prompt_cache_options
from harness.metrics import extract_usage


def test_prompt_cache_options_are_exposed():
    options = get_prompt_cache_options("openai:gpt-5-mini")
    assert options == {}

    anthropic_options = get_prompt_cache_options("anthropic:claude-3-5-sonnet")
    assert anthropic_options == {"cache_control": {"type": "ephemeral"}}


def test_extract_usage_handles_provider_cache_aliases():
    usage = {
        "input_tokens": 128,
        "output_tokens": 32,
        "input_token_details": {
            "cache_read": 64,
            "cache_creation": 12,
        },
    }
    normalized = extract_usage(type("Response", (), {"usage_metadata": usage})())
    assert normalized["cached_tokens"] == 64
    assert normalized["cache_creation_tokens"] == 12
