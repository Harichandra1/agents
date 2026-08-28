from typing import Literal
import operator

from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    AnyMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

from tools import web_search_tool, code_exec_tool


load_dotenv()


# =========================================================
# AGENT REGISTRY
# =========================================================

AGENT_REGISTRY: dict[str, dict] = {
    "researcher": {
        "system_prompt": (
            "You are a research sub-agent. "
            "Be thorough. Use the available tools when needed. "
            "Provide concise findings and cite sources when possible."
        ),
        "tools": [web_search_tool],
        "model": "openai-gpt-5.4-mini",
    },

    "coder": {
        "system_prompt": (
            "You are a coding sub-agent. "
            "Write clean, correct, maintainable code. "
            "Use the available coding tools when needed."
        ),
        "tools": [code_exec_tool],
        "model": "openai-gpt-5.4-mini",
    },

    # Example of adding domain-specific agents later:
    #
    # "flight_agent": {
    #     "system_prompt": (
    #         "You are a flight-search specialist. "
    #         "Find and compare flight options."
    #     ),
    #     "tools": [flight_search_tool],
    #     "model": "openai-gpt-5.4-mini",
    # },
    #
    # "hotel_agent": {
    #     "system_prompt": (
    #         "You are a hotel-search specialist."
    #     ),
    #     "tools": [hotel_search_tool],
    #     "model": "openai-gpt-5.4-mini",
    # },

    "generic": {
        "system_prompt": (
            "You are a general-purpose sub-agent. "
            "Complete the assigned task clearly and accurately."
        ),
        "tools": [],
        "model": "openai:gpt-5.4-mini",
    },
}


# =========================================================
# MODELS
# =========================================================

parent_llm = init_chat_model("openai:gpt-5.4-mini")


# =========================================================
# STATE DEFINITIONS
# =========================================================

class Task(TypedDict):
    id: str
    agent: str
    instruction: str
    status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
    ]


class SubAgentResult(TypedDict):
    task_id: str
    result: str


class ParentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    # Tasks created by the parent
    tasks: list[Task]

    # Results returned by parallel workers
    results: Annotated[
        list[SubAgentResult],
        operator.add,
    ]

    # Routing decision
    next_action: Literal[
        "delegate",
        "respond",
    ]


class ParentDecision(TypedDict):
    action: Literal[
        "respond",
        "delegate",
    ]

    tasks: list[Task]

    # Used only when action == "respond"
    response: str | None


class WorkerState(TypedDict):
    """
    Isolated state for ONE sub-agent execution.

    The worker does NOT receive the parent's full message history.
    """

    task_id: str
    agent: str
    instruction: str


# =========================================================
# AGENT REGISTRY HELPERS
# =========================================================

def get_agent_config(agent_name: str) -> dict:
    """
    Get configuration for a registered sub-agent.

    Unknown agent names fall back to generic.
    """

    return AGENT_REGISTRY.get(
        agent_name,
        AGENT_REGISTRY["generic"],
    )


# =========================================================
# TOOL EXECUTION LOOP
# =========================================================

def run_worker_agent(
    *,
    model_name: str,
    system_prompt: str,
    tools: list,
    instruction: str,
) -> str:
    """
    Run one sub-agent.

    Important:
    bind_tools() only exposes tools to the model.
    It does NOT execute them automatically.

    This function implements the basic tool-calling loop.
    """

    llm = init_chat_model(model_name)

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    if tools:
        llm = llm.bind_tools(tools)

    messages: list[AnyMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=instruction),
    ]

    while True:

        response = llm.invoke(messages)

        # No tool calls -> final answer
        if not getattr(response, "tool_calls", None):
            return response.content or ""

        # Add AI tool-call message to history
        messages.append(response)

        # Execute every requested tool
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call["id"]

            tool = tool_map.get(tool_name)

            if tool is None:
                tool_result = (
                    f"Tool '{tool_name}' is not available."
                )
            else:
                try:
                    tool_result = tool.invoke(tool_args)
                except Exception as exc:
                    tool_result = (
                        f"Tool execution failed: {exc}"
                    )

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )


# =========================================================
# PARENT AGENT
# =========================================================

def parent_agent(state: ParentState):

    messages = state["messages"]

    # -----------------------------------------------------
    # CASE 1:
    # We already have worker results.
    #
    # Do NOT ask the LLM whether to delegate again.
    # Just synthesize the final answer.
    # -----------------------------------------------------

    if state.get("results"):

        results_text = "\n\n".join(
            f"Task {result['task_id']}:\n"
            f"{result['result']}"
            for result in state["results"]
        )

        response = parent_llm.invoke(
            [
                SystemMessage(
                    content="""
You are the parent/orchestrator agent.

You have received results from delegated sub-agents.

Your job:
1. Answer the user's original request.
2. Synthesize the sub-agent results.
3. Resolve conflicts when possible.
4. Remove unnecessary internal details.
5. Do not delegate more work.
6. Return only the final user-facing answer.
"""
                ),
                *messages,
                HumanMessage(
                    content=(
                        "Sub-agent results:\n\n"
                        f"{results_text}"
                    )
                ),
            ]
        )

        return {
            "next_action": "respond",
            "tasks": [],
            "messages": [response],
        }

    # -----------------------------------------------------
    # CASE 2:
    # First pass.
    #
    # Decide whether to answer directly or delegate.
    # -----------------------------------------------------

    decision = parent_llm.with_structured_output(
        ParentDecision
    ).invoke(
        [
            SystemMessage(
                content="""
You are the parent/orchestrator agent.

Your job is to understand the user's request and decide
whether to answer directly or delegate work to specialized
sub-agents.

Choose "respond" when:
- you can answer reliably using your own reasoning
- no specialized external work is needed
- delegation would add unnecessary complexity

Choose "delegate" when:
- specialized expertise is useful
- external research/tool use is required
- the request contains independent pieces of work
  that can be performed in parallel
- another agent can do the work better

When delegating:
- create one or more self-contained tasks
- choose an agent name from the available registry
- make each instruction precise
- do not give a worker unnecessary parent context

Available agents:
- researcher
- coder
- generic

Return:
- action
- tasks
- response

If action == "respond", tasks should be [].
If action == "delegate", response should be null.
"""
            ),
            *messages,
        ]
    )

    if decision["action"] == "respond":

        return {
            "next_action": "respond",
            "tasks": [],
            "messages": [
                AIMessage(
                    content=decision["response"] or ""
                )
            ],
        }

    return {
        "next_action": "delegate",
        "tasks": decision["tasks"],
        "messages": [],
    }


# =========================================================
# ROUTER / TASK DISPATCHER
# =========================================================

def route_after_parent(state: ParentState):

    # Parent answered directly
    if state["next_action"] == "respond":
        return END

    # Parent wants delegation.
    #
    # Send one isolated WorkerState per task.
    #
    # If there are 3 tasks:
    #
    #     Send -> worker 1
    #     Send -> worker 2
    #     Send -> worker 3
    #
    # They can run in parallel.
    return [
        Send(
            "Sub Agent",
            {
                "task_id": task["id"],
                "agent": task["agent"],
                "instruction": task["instruction"],
            },
        )
        for task in state["tasks"]
    ]


# =========================================================
# SUB-AGENT WORKER
# =========================================================

def sub_agent_worker(state: WorkerState):

    config = get_agent_config(
        state["agent"]
    )

    try:

        result = run_worker_agent(
            model_name=config["model"],
            system_prompt=config["system_prompt"],
            tools=config["tools"],
            instruction=state["instruction"],
        )

        return {
            "results": [
                {
                    "task_id": state["task_id"],
                    "result": result,
                }
            ]
        }

    except Exception as exc:

        return {
            "results": [
                {
                    "task_id": state["task_id"],
                    "result": (
                        f"Sub-agent failed: {exc}"
                    ),
                }
            ]
        }


# =========================================================
# BUILD PARENT GRAPH
# =========================================================

builder = StateGraph(ParentState)


# Parent/orchestrator
builder.add_node(
    "Parent Agent",
    parent_agent,
)


# Worker
builder.add_node(
    "Sub Agent",
    sub_agent_worker,
)


# ---------------------------------------------------------
# START -> Parent
# ---------------------------------------------------------

builder.add_edge(
    START,
    "Parent Agent",
)


# ---------------------------------------------------------
# Parent -> either:
#
# respond
#     -> END
#
# delegate
#     -> Send(...) -> Sub Agent
# ---------------------------------------------------------

builder.add_conditional_edges(
    "Parent Agent",
    route_after_parent,
)


# ---------------------------------------------------------
# Sub Agent -> Parent
#
# Every Send worker returns a result into ParentState.
#
# The `results` reducer:
#
#     Annotated[list[...], operator.add]
#
# merges parallel worker results.
# ---------------------------------------------------------

builder.add_edge(
    "Sub Agent",
    "Parent Agent",
)


# =========================================================
# COMPILE
# =========================================================

agent = builder.compile()


# =========================================================
# EXAMPLE 1: DIRECT ANSWER
# =========================================================

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Explain what context isolation means "
                    "in multi-agent systems."
                )
            )
        ],
        "tasks": [],
        "results": [],
        "next_action": "respond",
    }
)

print("\n=== DIRECT ANSWER ===\n")
print(
    result["messages"][-1].content
)


# =========================================================
# EXAMPLE 2: DELEGATION
# =========================================================

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Research the most important AI trends "
                    "right now and summarize the findings."
                )
            )
        ],
        "tasks": [],
        "results": [],
        "next_action": "respond",
    }
)

print("\n=== DELEGATED ANSWER ===\n")
print(
    result["messages"][-1].content
)