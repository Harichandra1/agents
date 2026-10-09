
import operator
import time
import uuid

from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

from harness.config import (
    get_chat_model,
)

from harness.metrics import LLMMetrics, extract_usage
from harness.config import get_routing_model
from harness.tools import sandboxing

try:
    from harness.tools import ARITHMETIC_TOOLS
except ImportError:
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @tool
    def subtract(a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    @tool
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    ARITHMETIC_TOOLS = [add, subtract, multiply, divide]

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to answer questions.  
"""
ROUTING_SYSTEM_PROMPT = """
You are a triage assistant choosing the best model for each user request.

This is a strong routing policy. Be decisive.

Routing rules:
- Use "openai:gpt-4o-mini" only for trivial one-shot questions with no calculation, no comparison, and no synthesis.
- Use "openai:gpt-5.6-luna" only for short, easy factual questions or tiny tasks with minimal reasoning.
- Use "openai:gpt-5.6-terra" for most mixed questions, multi-step tasks, fact + calculation requests, summarization, and moderate analysis.
- Use "openai:gpt-5.6-sol" for hard reasoning, deep judgment, multi-part questions, nuanced synthesis, or any request that combines explanation and problem-solving.

Choose exactly one full model id from the list above and set `routing_model` to that exact string.
Set `reasoning_effort` to the matching depth: "none", "low", "medium", or "high".
Do not return just "openai". Do not return a short alias. Return the exact full model id.
"""


class RoutingConfig(TypedDict):
    routing_model: Literal[
        "openai:gpt-4o-mini",
        "openai:gpt-5.6-luna",
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-sol",
    ]
    reasoning_effort: Literal["none", "low", "medium", "high"]


def normalize_routing_model(value: str | None) -> str:
    """Normalize generic provider names to the exact full model ID."""
    if value is None:
        return "openai:gpt-4o-mini"

    cleaned = str(value).strip().lower()
    aliases = {
        "openai": "openai:gpt-4o-mini",
        "gpt-4o-mini": "openai:gpt-4o-mini",
        "gpt-5.6-luna": "openai:gpt-5.6-luna",
        "openai:gpt-5.6-luna": "openai:gpt-5.6-luna",
        "gpt-5.6-terra": "openai:gpt-5.6-terra",
        "openai:gpt-5.6-terra": "openai:gpt-5.6-terra",
        "gpt-5.6-sol": "openai:gpt-5.6-sol",
        "openai:gpt-5.6-sol": "openai:gpt-5.6-sol",
    }

    return aliases.get(cleaned, cleaned if cleaned.startswith("openai:") else f"openai:{cleaned}")

MODEL_INPUT_COST_PER_1K = {
    "openai:gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o-mini": (0.00015, 0.0006),
    "openai:gpt-4o": (0.005, 0.015),
    "gpt-4o": (0.005, 0.015),
    "openai:gpt-5.6-luna": (0.0015, 0.006),
    "gpt-5.6-luna": (0.0015, 0.006),
    "openai:gpt-5.6-terra": (0.0025, 0.01),
    "gpt-5.6-terra": (0.0025, 0.01),
    "openai:gpt-5.6-sol": (0.01, 0.04),
    "gpt-5.6-sol": (0.01, 0.04),
}


def calculate_response_cost(response, model_name: str | None = None) -> float:
    """Return a best-effort cost estimate for a model response in USD."""
    usage = extract_usage(response)
    if not usage or not any(usage.values()):
        return 0.0

    model_key = str(model_name or getattr(response, "model", "") or "").lower()
    if not model_key and hasattr(response, "response_metadata"):
        model_key = str(response.response_metadata.get("model_name") or "").lower()

    model_key = normalize_routing_model(model_key) if model_key else model_key
    input_rate, output_rate = MODEL_INPUT_COST_PER_1K.get(model_key, (0.0, 0.0))

    input_cost = (usage["input_tokens"] / 1000.0) * input_rate
    output_cost = (usage["output_tokens"] / 1000.0) * output_rate
    return input_cost + output_cost


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    routing: NotRequired[RoutingConfig]
    cost: float

@dataclass
class Context:
    user_id: str
def build_loop_graph(llm=None):
    """The same loop as a StateGraph: agent node -> tool node -> back."""
    llm = llm or get_chat_model()
    ALL_TOOLS = ARITHMETIC_TOOLS+[sandboxing]
    model_with_tools = llm.bind_tools(ALL_TOOLS)
    tools_by_name = {t.name if hasattr(t, "name") else t.__name__: t for t in ALL_TOOLS}
    # short-term memory
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    def routing(state: LoopState) -> dict:
        """This routing decide which to model to call."""
        routing_model = get_routing_model()
        routing_model = routing_model.with_structured_output(RoutingConfig)
        response = routing_model.invoke([
            SystemMessage(content=ROUTING_SYSTEM_PROMPT),
            *state["messages"],
        ])

        if isinstance(response, dict):
            response["routing_model"] = normalize_routing_model(response.get("routing_model"))
        else:
            response = {**response.__dict__, "routing_model": normalize_routing_model(getattr(response, "routing_model", None))}

        routing_model_choice = response.get("routing_model")
        print(f"Routing model: {routing_model_choice}")
        return {
            "routing": response,
            "cost": state.get("cost", 0.0),
        }
        
    def llm_call(
        state: LoopState,
        runtime: Runtime[Context],
    ) -> dict:
        """LLM decides whether to call a tool or not."""

        routing = state.get("routing")
        if routing is not None:
            print(f"Selected routing configuration: {routing}")

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
        model_name = (
            state.get("routing", {}).get("routing_model")
            if state.get("routing")
            else getattr(llm, "model_name", None) or getattr(llm, "model", None)
        )
        cost = calculate_response_cost(response, model_name)
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "cost": state.get("cost", 0.0) + cost,
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
    builder.add_node("routing",routing)
    builder.add_node("tool_node", tool_node)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "routing")
    builder.add_edge("routing", "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("llm_call", "save_memory")

    return builder.compile(checkpointer=checkpointer,store=store)