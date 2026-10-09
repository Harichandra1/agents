"""This contains sandboxing with docker and obserability"""

import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from opentelemetry import trace

from harness.config import get_chat_model
from harness.tools.sandboxing import execute_python_code_in_docker

tracer = trace.get_tracer(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to answer questions.
For calculations or executable code, use the execute_python_code_in_docker tool.
Pass valid Python source code in its `code` argument, inspect the tool result, and base your
numerical answer on that result. Do not claim to have run code unless the tool was called.
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
    all_tools = [execute_python_code_in_docker]
    model_with_tools = llm.bind_tools(all_tools)
    tools_by_name = {tool.name: tool for tool in all_tools}
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    def llm_call(
        state: LoopState,
        runtime: Runtime[Context],
    ) -> dict:
        """LLM decides whether to call a tool or not."""
        user_id = runtime.context.user_id
        namespace = ("users", user_id, "memories")
        user_message = next(
            (
                message
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        query = (
            user_message.content
            if user_message is not None
            else state["messages"][-1].content
        )

        with tracer.start_as_current_span("memory-search") as span:
            span.set_attribute("langfuse.observation.input", str(query))
            memories = runtime.store.search(namespace, query=str(query), limit=5)
            span.set_attribute("memory.count", len(memories))
            span.set_attribute(
                "langfuse.observation.output",
                str([memory.value["text"] for memory in memories]),
            )

        memory_text = "\n".join(f"-{memory.value['text']}" for memory in memories)

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

    def tool_node(state: LoopState) -> dict:
        """Execute every tool the last message asked for."""
        results = []
        for call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[call["name"]]
            observation = selected_tool.invoke(call["args"])
            results.append(
                ToolMessage(content=str(observation), tool_call_id=call["id"])
            )
        return {"messages": results}

    def should_continue(state: LoopState) -> Literal["tool_node", "save_memory"]:
        last = state["messages"][-1]
        return "tool_node" if getattr(last, "tool_calls", None) else "save_memory"

    def save_memory(state: LoopState, runtime: Runtime[Context]) -> dict:
        user_id = runtime.context.user_id
        namespace = ("users", user_id, "memories")
        user_message = next(
            (
                message
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            None,
        )

        # Very simple demo logic.
        # In production, use an LLM structured-output step.
        if user_message is not None and isinstance(user_message.content, str):
            text = user_message.content
            if "my name is" in text.lower():
                runtime.store.put(
                    namespace,
                    str(uuid.uuid4()),
                    {"text": text},
                )

        return {}

    builder = StateGraph(LoopState, context_schema=Context)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", "save_memory"],
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("save_memory", END)

    return builder.compile(checkpointer=checkpointer, store=store)