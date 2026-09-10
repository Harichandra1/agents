"""
Day 1 — the bare ReAct loop, then the same thing as a LangGraph StateGraph.

The whole point of this day is to feel what LangGraph actually buys you, so
both versions live here side by side and solve the same problem:

    run_bare_loop()    -- raw while-loop + manual tool dispatch, no framework
    build_loop_graph() -- identical behaviour as a StateGraph

The graph version gets you routing, state reduction, checkpointing and
streaming for free. The bare version is ~20 lines and has none of it. Read
them together; the difference is the lesson.
"""
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from harness.config import get_chat_model
from harness.tools.arithmetic import ARITHMETIC_TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def run_bare_loop(messages: list[AnyMessage], llm=None, max_turns: int = 10) -> list[AnyMessage]:
    """
    The no-framework version: bind tools, call, dispatch any tool calls by
    hand, append results, repeat until the model stops asking for tools.

    Returns the full message list including the final answer.
    """
    llm = llm or get_chat_model()
    model_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)
    tools_by_name = {t.name: t for t in ARITHMETIC_TOOLS}

    history = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    for _ in range(max_turns):
        response = model_with_tools.invoke(history)
        history.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return history

        for call in tool_calls:
            tool = tools_by_name[call["name"]]
            observation = tool.invoke(call["args"])
            # content must be a str -- the tools return ints/floats.
            history.append(
                ToolMessage(content=str(observation), tool_call_id=call["id"])
            )

    return history


def build_loop_graph(llm=None):
    """The same loop as a StateGraph: agent node -> tool node -> back."""
    llm = llm or get_chat_model()
    model_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)
    tools_by_name = {t.name: t for t in ARITHMETIC_TOOLS}

    def llm_call(state: LoopState) -> dict:
        """LLM decides whether to call a tool or not."""
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: LoopState) -> dict:
        """Execute every tool the last message asked for."""
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name[call["name"]]
            observation = tool.invoke(call["args"])
            results.append(
                ToolMessage(content=str(observation), tool_call_id=call["id"])
            )
        return {"messages": results}

    def should_continue(state: LoopState) -> Literal["tool_node", "__end__"]:
        last = state["messages"][-1]
        return "tool_node" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(LoopState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile()
