"""
Day 1 — the bare ReAct loop, then the same thing as a LangGraph StateGraph.

The whole point of this day is to feel what LangGraph actually buys you, so
both versions live here side by side and solve the same problem:

    build_loop_graph() -- ReAct loop implemented as a StateGraph

The graph version gets you routing, state reduction, checkpointing and
streaming for free.
"""
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from harness.config import get_chat_model
from harness.tools.arithmetic import ARITHMETIC_TOOLS
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
import operator 
import uuid
from dataclasses import dataclass

SYSTEM_PROMPT = """
You are a helpful assistant tasked with performing arithmetic.

You also have access to long-term memories about the user.

Use those memories when relevant.
Do not assume a memory is relevant if it is unrelated to the current request
"""


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

@dataclass
class Context:
    user_id: str

def build_loop_graph(llm=None):
    """The same loop as a StateGraph: agent node -> tool node -> back."""
    llm = llm or get_chat_model()
    model_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)
    tools_by_name = {t.name: t for t in ARITHMETIC_TOOLS}
    # short-term memory
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    def llm_call(state: LoopState,runtime:Runtime[Context],) -> dict:
        """LLM decides whether to call a tool or not."""

        user_id = runtime.context.user_id
        namespace = ("users",user_id,"memories")
        memories = runtime.store.search(namespace,query=state["messages"][-1].content,limit=5)
        memory_text = "\n".join(f"-{memory.value['text']}"for memory in memories)

        system_prompt = f"""
        {SYSTEM_PROMPT}

        Long-term memories about this user:

        {memory_text or "No relevant memories found."}
        """


        response = model_with_tools.invoke(
            [SystemMessage(content=system_prompt), *state["messages"]]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: LoopState,runtime:Runtime[Context]) -> dict:
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
    def save_memory(state: LoopState,runtime: Runtime[Context],) -> dict:

        user_id = runtime.context.user_id

        namespace = ("users", user_id, "memories")

        last_message = state["messages"][-1]

        text = last_message.content

        # Very simple demo logic.
        # In production, use an LLM structured-output step.
        if "my name is" in text.lower():

            memory = {
                "text": text
            }

            runtime.store.put(
                namespace,
                str(uuid.uuid4()),
                memory,
            )

        return {}

    builder = StateGraph(LoopState,context_schema=Context)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_node("save_memory", save_memory)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("llm_call", "save_memory")

    return builder.compile(checkpointer=checkpointer,store=store)

