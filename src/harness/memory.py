
"""
LangGraph ReAct loop with:

- long-term memory
- provider-level prompt caching
- streaming
- TTFT
- TPS
- latency
- token usage
- cache hit/write metrics
"""

import operator
import time
import uuid

from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

from harness.config import (
    get_chat_model,
    get_prompt_cache_key,
    get_prompt_cache_options,
)

from harness.metrics import (
    LLMMetrics,
    extract_usage,
)

from harness.tools.arithmetic import ARITHMETIC_TOOLS


SYSTEM_PROMPT = """
You are a helpful assistant tasked with performing arithmetic.

You also have access to long-term memories about the user.

Use those memories when relevant.
Do not assume a memory is relevant if it is unrelated to the current request.
""".strip()


class LoopState(TypedDict):
    messages: Annotated[
        list[AnyMessage],
        operator.add,
    ]

    llm_calls: int


@dataclass
class Context:
    user_id: str


def build_loop_graph(llm=None):
    """
    Build the LangGraph ReAct loop.
    """

    llm = llm or get_chat_model()
    print("\n=== MODEL CONFIG ===")
    print(llm)
    print("====================\n") 
    model_with_tools = llm.bind_tools(
        ARITHMETIC_TOOLS
    )

    tools_by_name = {
        tool.name: tool
        for tool in ARITHMETIC_TOOLS
    }

    checkpointer = InMemorySaver()
    store = InMemoryStore()

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def llm_call(
        state: LoopState,
        runtime: Runtime[Context],
    ) -> dict:
        """
        Call the model while measuring:

        - request start
        - request end
        - TTFT
        - generation duration
        - TPS
        - token usage
        - prompt-cache usage
        """

        user_id = runtime.context.user_id

        namespace = (
            "users",
            user_id,
            "memories",
        )

        # --------------------------------------------------------------
        # Retrieve long-term memories
        # --------------------------------------------------------------

        memories = runtime.store.search(
            namespace,
            query=state["messages"][-1].content,
            limit=5,
        )

        memory_text = "\n".join(
            f"- {memory.value['text']}"
            for memory in memories
        )

        # --------------------------------------------------------------
        # Build prompt
        # --------------------------------------------------------------

        system_prompt = f"""
{SYSTEM_PROMPT}

Relevant memories:
{memory_text if memory_text else "- None"}
""".strip()

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            *state["messages"],
        ]

        # --------------------------------------------------------------
        # Timing
        # --------------------------------------------------------------

        start_time = time.perf_counter()

        first_token_time = None
        chunks = []

        # --------------------------------------------------------------
        # Provider-level prompt cache
        # --------------------------------------------------------------

        cache_options = get_prompt_cache_options(
            getattr(llm, "model_name", None) or getattr(llm, "model", None)
        )

        stream_kwargs = {}

        if "cache_control" in cache_options:
            system_message = SystemMessage(
                content=system_prompt,
                additional_kwargs={
                    "cache_control": cache_options["cache_control"],
                },
            )
            messages = [system_message, *state["messages"]]

        # --------------------------------------------------------------
        # Stream response
        # --------------------------------------------------------------

        for chunk in model_with_tools.stream(
            messages,
            **stream_kwargs,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()

            chunks.append(chunk)

        end_time = time.perf_counter()

        if not chunks:
            raise RuntimeError(
                "LLM returned no streaming chunks."
            )

        # --------------------------------------------------------------
        # Combine streamed chunks
        # --------------------------------------------------------------

        response = chunks[0]

        for chunk in chunks[1:]:
            response += chunk

        # --------------------------------------------------------------
        # Usage
        # --------------------------------------------------------------

        usage = extract_usage(response)

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        latency_ms = (
            end_time - start_time
        ) * 1000

        ttft_ms = None

        if first_token_time is not None:
            ttft_ms = (
                first_token_time - start_time
            ) * 1000

        generation_time_ms = None

        if first_token_time is not None:
            generation_time_ms = (
                end_time - first_token_time
            ) * 1000

        tps = None

        if (
            generation_time_ms is not None
            and generation_time_ms > 0
            and usage["output_tokens"] > 0
        ):
            tps = (
                usage["output_tokens"]
                / (generation_time_ms / 1000)
            )

        metrics = LLMMetrics(
            start_time=start_time,
            end_time=end_time,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            output_tokens=usage["output_tokens"],
            input_tokens=usage["input_tokens"],
            cached_tokens=usage["cached_tokens"],
            generation_time_ms=generation_time_ms,
            tps=tps,
            cache_hit=usage["cached_tokens"] > 0,
            cache_creation_tokens=usage[
                "cache_creation_tokens"
            ],
        )

        # --------------------------------------------------------------
        # Metrics output
        # --------------------------------------------------------------

        print("\n=== LLM Metrics ===")

        print(
            f"Start time       : "
            f"{metrics.start_time:.6f}"
        )

        print(
            f"End time         : "
            f"{metrics.end_time:.6f}"
        )

        print(
            f"Latency          : "
            f"{metrics.latency_ms:.2f} ms"
        )

        print(
            f"TTFT             : "
            f"{metrics.ttft_ms:.2f} ms"
            if metrics.ttft_ms is not None
            else "TTFT             : N/A"
        )

        print(
            f"Generation time  : "
            f"{metrics.generation_time_ms:.2f} ms"
            if metrics.generation_time_ms is not None
            else "Generation time  : N/A"
        )

        print(
            f"Input tokens     : "
            f"{metrics.input_tokens}"
        )

        print(
            f"Output tokens    : "
            f"{metrics.output_tokens}"
        )

        print(
            f"Cached tokens    : "
            f"{metrics.cached_tokens}"
        )

        print(
            f"Cache write      : "
            f"{metrics.cache_creation_tokens}"
        )

        print(
            f"Cache hit        : "
            f"{metrics.cache_hit}"
        )

        print(
            f"TPS              : "
            f"{metrics.tps:.2f}"
            if metrics.tps is not None
            else "TPS              : N/A"
        )

        return {
            "messages": [response],
            "llm_calls": (
                state.get("llm_calls", 0) + 1
            ),
        }

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def tool_node(
        state: LoopState,
        runtime: Runtime[Context],
    ) -> dict:
        """
        Execute all tools requested by the model.
        """

        results = []

        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name[call["name"]]

            observation = tool.invoke(
                call["args"]
            )

            results.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=call["id"],
                )
            )

        return {
            "messages": results
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_after_llm(
        state: LoopState,
    ) -> Literal[
        "tool_node",
        "save_memory",
    ]:
        last = state["messages"][-1]

        if getattr(
            last,
            "tool_calls",
            None,
        ):
            return "tool_node"

        return "save_memory"

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def save_memory(
        state: LoopState,
        runtime: Runtime[Context],
    ) -> dict:
        user_id = runtime.context.user_id

        namespace = (
            "users",
            user_id,
            "memories",
        )

        last_message = state["messages"][-1]

        text = last_message.content

        if "my name is" in text.lower():
            runtime.store.put(
                namespace,
                str(uuid.uuid4()),
                {
                    "text": text
                },
            )

        return {}

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    builder = StateGraph(
        LoopState,
        context_schema=Context,
    )

    builder.add_node(
        "llm_call",
        llm_call,
    )

    builder.add_node(
        "tool_node",
        tool_node,
    )

    builder.add_node(
        "save_memory",
        save_memory,
    )

    builder.add_edge(
        START,
        "llm_call",
    )

    builder.add_conditional_edges(
        "llm_call",
        route_after_llm,
        {
            "tool_node": "tool_node",
            "save_memory": "save_memory",
        },
    )

    builder.add_edge(
        "tool_node",
        "llm_call",
    )

    builder.add_edge(
        "save_memory",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
