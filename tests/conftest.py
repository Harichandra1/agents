"""
Shared offline test doubles.

Every test in this suite runs without an API key or network access: the
graphs are driven by a scripted fake chat model instead of a real provider.
That is what lets `pytest` be a fast regression gate in CI (Day 22).
"""
import uuid

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedLLM(BaseChatModel):
    """Returns pre-scripted AIMessages in order, ignoring bind_tools."""

    script: list

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        """Structured-output calls pop the next scripted item verbatim."""
        return _ScriptedStructured(self)

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        idx = getattr(self, "_call_count", 0)
        response = self.script[min(idx, len(self.script) - 1)]
        object.__setattr__(self, "_call_count", idx + 1)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


class _ScriptedStructured:
    """Mimics `.with_structured_output(...)` by replaying raw objects."""

    def __init__(self, parent: ScriptedLLM):
        self._parent = parent

    def invoke(self, _messages, **_kwargs):
        idx = getattr(self._parent, "_call_count", 0)
        item = self._parent.script[min(idx, len(self._parent.script) - 1)]
        object.__setattr__(self._parent, "_call_count", idx + 1)
        return item


def tool_call(name: str, args: dict) -> dict:
    return {
        "name": name,
        "args": args,
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "tool_call",
    }


@pytest.fixture
def scripted():
    """Factory: `scripted([AIMessage(...), ...])` -> a ScriptedLLM."""
    return lambda script: ScriptedLLM(script=script)
