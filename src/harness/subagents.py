"""
Day 4 — sub-agent delegation with context isolation.

The parent decides whether to answer directly or split the work into tasks.
Each task is dispatched with `Send` to a worker that receives ONLY its own
instruction -- never the parent's message history. Workers return a summary,
which the parent synthesises into the final answer.

That isolation is the whole point: it is how Claude Code and Deep Agents keep
a long parent conversation from being re-paid for on every sub-task.

Note `bind_tools()` only exposes tools to the model; it does not execute
them. `_run_worker` implements the tool-calling loop that actually does.
"""
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

from harness.config import get_chat_model
from harness.tools.stubs import code_exec_tool, web_search_tool

# Sub-agent definitions. `model=None` means "use the configured default";
# set a string here to pin one worker to a specific model (Day 9 routing).
AGENT_REGISTRY: dict[str, dict] = {
    "researcher": {
        "system_prompt": (
            "You are a research sub-agent. Be thorough. Use the available "
            "tools when needed. Provide concise findings and cite sources "
            "when possible."
        ),
        "tools": [web_search_tool],
        "model": None,
    },
    "coder": {
        "system_prompt": (
            "You are a coding sub-agent. Write clean, correct, maintainable "
            "code. Use the available coding tools when needed."
        ),
        "tools": [code_exec_tool],
        "model": None,
    },
    "generic": {
        "system_prompt": (
            "You are a general-purpose sub-agent. Complete the assigned task "
            "clearly and accurately."
        ),
        "tools": [],
        "model": None,
    },
}

MAX_WORKER_TURNS = 10


class Task(TypedDict):
    id: str
    agent: str
    instruction: str
    status: Literal["pending", "running", "completed", "failed"]


class SubAgentResult(TypedDict):
    task_id: str
    result: str


class ParentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tasks: list[Task]
    results: Annotated[list[SubAgentResult], operator.add]
    next_action: Literal["delegate", "respond"]


class ParentDecision(TypedDict):
    action: Literal["respond", "delegate"]
    tasks: list[Task]
    response: str | None


class WorkerState(TypedDict):
    """Isolated state for ONE sub-agent. No parent history reaches this."""

    task_id: str
    agent: str
    instruction: str


PARENT_DELEGATION_PROMPT = """You are the parent/orchestrator agent.

Your job is to understand the user's request and decide whether to answer
directly or delegate work to specialized sub-agents.

Choose "respond" when:
- you can answer reliably using your own reasoning
- no specialized external work is needed
- delegation would add unnecessary complexity

Choose "delegate" when:
- specialized expertise is useful
- external research/tool use is required
- the request contains independent pieces of work that can run in parallel

When delegating:
- create one or more self-contained tasks
- choose an agent name from the available registry
- make each instruction precise
- do not give a worker unnecessary parent context

Available agents:
- researcher
- coder
- generic

If action == "respond", tasks should be []. If action == "delegate",
response should be null."""

PARENT_SYNTHESIS_PROMPT = """You are the parent/orchestrator agent.

You have received results from delegated sub-agents.

Your job:
1. Answer the user's original request.
2. Synthesize the sub-agent results.
3. Resolve conflicts when possible.
4. Remove unnecessary internal details.
5. Do not delegate more work.
6. Return only the final user-facing answer."""


def get_agent_config(agent_name: str) -> dict:
    """Look up a sub-agent. Unknown names fall back to `generic`."""
    return AGENT_REGISTRY.get(agent_name, AGENT_REGISTRY["generic"])


def _run_worker(
    *, model_name: str | None, system_prompt: str, tools: list, instruction: str, llm=None
) -> str:
    """
    Run one sub-agent to completion, including its tool-calling loop.

    Returns the worker's final text answer -- the ONLY thing the parent sees.
    """
    worker_llm = llm or get_chat_model(model_name)
    tool_map = {t.name: t for t in tools}
    if tools:
        worker_llm = worker_llm.bind_tools(tools)

    messages: list[AnyMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=instruction),
    ]

    for _ in range(MAX_WORKER_TURNS):
        response = worker_llm.invoke(messages)
        if not getattr(response, "tool_calls", None):
            return response.content or ""

        messages.append(response)
        for call in response.tool_calls:
            tool = tool_map.get(call["name"])
            if tool is None:
                result = f"Tool '{call['name']}' is not available."
            else:
                try:
                    result = tool.invoke(call.get("args", {}))
                except Exception as exc:
                    result = f"Tool execution failed: {exc}"
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    return "Worker hit the turn limit without producing a final answer."


def build_subagent_graph(llm=None, worker_llm=None):
    parent_llm = llm or get_chat_model()

    def parent_agent(state: ParentState) -> dict:
        # Results already in hand -> synthesise, don't re-decide.
        if state.get("results"):
            results_text = "\n\n".join(
                f"Task {r['task_id']}:\n{r['result']}" for r in state["results"]
            )
            response = parent_llm.invoke(
                [
                    SystemMessage(content=PARENT_SYNTHESIS_PROMPT),
                    *state["messages"],
                    HumanMessage(content=f"Sub-agent results:\n\n{results_text}"),
                ]
            )
            return {"next_action": "respond", "tasks": [], "messages": [response]}

        # First pass -> answer directly or split into tasks.
        decision = parent_llm.with_structured_output(ParentDecision).invoke(
            [SystemMessage(content=PARENT_DELEGATION_PROMPT), *state["messages"]]
        )

        if decision["action"] == "respond":
            return {
                "next_action": "respond",
                "tasks": [],
                "messages": [AIMessage(content=decision["response"] or "")],
            }

        return {
            "next_action": "delegate",
            "tasks": decision["tasks"],
            "messages": [],
        }

    def route_after_parent(state: ParentState):
        if state["next_action"] == "respond":
            return END
        # One isolated worker per task; they run in parallel.
        return [
            Send(
                "sub_agent",
                {
                    "task_id": task["id"],
                    "agent": task["agent"],
                    "instruction": task["instruction"],
                },
            )
            for task in state["tasks"]
        ]

    def sub_agent(state: WorkerState) -> dict:
        config = get_agent_config(state["agent"])
        try:
            result = _run_worker(
                model_name=config["model"],
                system_prompt=config["system_prompt"],
                tools=config["tools"],
                instruction=state["instruction"],
                llm=worker_llm,
            )
        except Exception as exc:
            result = f"Sub-agent failed: {exc}"
        return {"results": [{"task_id": state["task_id"], "result": result}]}

    builder = StateGraph(ParentState)
    builder.add_node("parent_agent", parent_agent)
    builder.add_node("sub_agent", sub_agent)
    builder.add_edge(START, "parent_agent")
    builder.add_conditional_edges("parent_agent", route_after_parent)
    builder.add_edge("sub_agent", "parent_agent")
    return builder.compile()
