"""
Validates the graph's wiring (lazy tool loading, scratchpad, compaction)
using a scripted fake chat model -- no API key or network call needed.

Run: python3 test_graph.py
"""
import uuid
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from graph import build_graph


class ScriptedLLM(BaseChatModel):
    """Returns pre-scripted AIMessages in order, ignoring bind_tools."""

    script: list

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        idx = getattr(self, "_call_count", 0)
        response = self.script[min(idx, len(self.script) - 1)]
        object.__setattr__(self, "_call_count", idx + 1)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def tool_call(name, args):
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def test_lazy_loading_and_scratchpad():
    script = [
        # Turn 1: discover + load the calculator tool
        AIMessage(content="", tool_calls=[tool_call("load_tools", {"names": ["calculator"]})]),
        # Turn 2: use it
        AIMessage(content="", tool_calls=[tool_call("calculator", {"expression": "6*7"})]),
        # Turn 3: stash a big result in the scratchpad
        AIMessage(content="", tool_calls=[tool_call("write_file", {"path": "notes/answer.txt", "content": "42"})]),
        # Turn 4: final answer, no more tool calls
        AIMessage(content="The answer is 42, and I saved it to notes/answer.txt."),
    ]
    llm = ScriptedLLM(script=script)
    app = build_graph(llm)

    config = {"configurable": {"thread_id": "test-1"}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content="What's 6*7? Save the answer too.")],
            "scratchpad": {},
            "loaded_tools": [],
        },
        config,
    )

    assert "calculator" in result["loaded_tools"], "load_tools should have updated state"
    assert result["scratchpad"]["notes/answer.txt"] == "42", "write_file should have updated scratchpad"
    assert "42" in result["messages"][-1].content
    print("PASS: lazy loading + scratchpad wiring works")
    print("  loaded_tools:", result["loaded_tools"])
    print("  scratchpad:", result["scratchpad"])
    print("  final message:", result["messages"][-1].content)


def test_compaction_triggers_and_preserves_recent():
    # Force compaction to trigger almost immediately for the test.
    import compaction
    compaction.TOKEN_THRESHOLD = 10
    compaction.KEEP_LAST_N_MESSAGES = 2

    script = [
        AIMessage(content="This is a long filler answer. " * 20),
        AIMessage(content="Second filler answer, also fairly long. " * 20),
        AIMessage(content="Final short answer."),
    ]
    llm = ScriptedLLM(script=script)
    summarizer = ScriptedLLM(script=[AIMessage(content="SUMMARY: earlier turns discussed filler content.")])
    app = build_graph(llm, summarizer_llm=summarizer)

    config = {"configurable": {"thread_id": "test-2"}}
    state = {"messages": [HumanMessage(content="hi")], "scratchpad": {}, "loaded_tools": []}

    state = app.invoke(state, config)
    state = app.invoke({"messages": [HumanMessage(content="continue")]}, config)

    msgs = state["messages"]
    has_summary = any(isinstance(m, SystemMessage) and "[Summary" in str(m.content) for m in msgs)
    assert has_summary, "compaction should have inserted a summary message"
    print("PASS: compaction triggers and inserts a summary")
    print(f"  message count after compaction: {len(msgs)}")
    for m in msgs:
        print(f"   [{m.type}] {str(m.content)[:80]}")


if __name__ == "__main__":
    test_lazy_loading_and_scratchpad()
    print()
    test_compaction_triggers_and_preserves_recent()
