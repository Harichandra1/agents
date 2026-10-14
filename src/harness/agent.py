"""
Day 13 — minimal coding-agent ReAct loop.

Shape: planner -> llm_call -> (tool_node -> llm_call)* -> END.
The planner writes `plan` once, then the ReAct loop runs as before.
Default tools are workspace `read` / `write` / `edit` / `bash`.
Pass `tools=` to replace the set; pass `workspace=` to jail file tools.

Deliberately skips memory, caching, sub-agents, and fan-out.
"""
import operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from harness.config import get_chat_model
from harness.planner import Plan
from harness.tools.coding import build_file_tools

SYSTEM_PROMPT = """You are a coding agent working in a single workspace.

Tools:
- read(path): read a UTF-8 file. `path` is relative to the workspace root.
- write(path, content): create or overwrite a whole file. Parent dirs are created.
- edit(path, old_string, new_string): replace exactly one occurrence in a file.
  Prefer edit for small changes; write for new files.
- bash(command): run a shell command in Docker. cwd is the workspace.
  Use this to run tests, python files, and ls. No network.

How to work:
- Inspect existing files with read before changing them.
- Put lasting code in the workspace with write or edit, not only in chat.
- After an edit or write, use bash to run the file or its tests.
- Paths stay inside the workspace. Never use `..` or absolute host paths.
- Do not claim you read, wrote, edited, or ran a command unless the tool returned.
- Follow the current plan. Do not skip steps without a reason.
- When the task is done, give a short answer: files touched and command results.
"""

PLANNER_PROMPT = """You are the planner for a coding agent.

Produce a short ordered plan (3–7 steps) for the user's task.
Each step should be something the agent can do with read, write, edit, or bash.
Do not write code. Do not execute anything. Only plan.
"""

DEFAULT_WORKSPACE = Path.cwd() / ".harness-workspace"


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    plan: list[str]


def _default_tools(workspace: str | Path):
    return build_file_tools(workspace)


def build_loop_graph(llm=None, tools=None, workspace: str | Path | None = None):
    """ReAct loop: agent node -> tool node -> back, until the model stops calling tools."""
    llm = llm or get_chat_model()
    all_tools = (
        list(tools)
        if tools is not None
        else _default_tools(workspace or DEFAULT_WORKSPACE)
    )
    model_with_tools = llm.bind_tools(all_tools)
    tools_by_name = {tool.name: tool for tool in all_tools}
    planner_structured = llm.with_structured_output(Plan)

    def planner(state: LoopState) -> dict:
        result = planner_structured.invoke(
            [SystemMessage(content=PLANNER_PROMPT), *state["messages"]]
        )
        return {
            "plan": result["plan"],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def llm_call(state: LoopState) -> dict:
        plan = state.get("plan") or []
        plan_block = ""
        if plan:
            numbered = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
            plan_block = f"\n\nCurrent plan:\n{numbered}\n"
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT + plan_block), *state["messages"]]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: LoopState) -> dict:
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
    builder.add_node("planner", planner)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile()
