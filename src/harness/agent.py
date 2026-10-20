"""
Day 13 — minimal coding-agent ReAct loop.

Shape: planner -> llm_call -> (tool_node -> llm_call)* -> [review -> llm_call ...] -> END.
The planner writes `plan` once, then the ReAct loop runs as before.
With `review_prompt=`, the first time the model stops calling tools it gets
that prompt as a new user turn (a self-critique checklist) and keeps going;
the second stop ends the run.
Default tools are workspace `read` / `write` / `edit` / `bash`.
Pass `tools=` to replace the set; pass `workspace=` to jail file tools.

Deliberately skips memory, caching, sub-agents, and fan-out.
"""
import operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from harness.config import get_chat_model
from harness.planner import Plan
from harness.tools.coding import WorkspaceSandbox, build_file_tools, truncate_head_tail

RECURSION_LIMIT = 120
MAX_LLM_CALLS = 50

SYSTEM_PROMPT = """You are a coding agent working in a single workspace.

Tools:
- read(path, start_line=1, end_line=None): read a UTF-8 file. `path` is relative
  to the workspace root. Long files come back in windows of lines; the footer
  says which lines you saw and where to continue. Find lines with `grep -n`.
- write(path, content): create or overwrite a whole file. Parent dirs are created.
- edit(path, old_string, new_string): replace exactly one occurrence in a file.
  Prefer edit for small changes; write for new files.
- bash(command): run a shell command in Docker. cwd is the workspace.
  Use this to run tests, python files, grep, and ls. No network.

How to work:
- Inspect existing files with read before changing them.
- Put lasting code in the workspace with write or edit, not only in chat.
- After an edit or write, use bash to run the file or its tests.
- If your change breaks a test that passed before, fix the change, not the test.
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


def invoke_config(**extra) -> dict:
    """LangGraph invoke config: recursion cap plus caller tags/metadata."""
    return {"recursion_limit": RECURSION_LIMIT, **extra}


class LoopState(TypedDict):
    """Graph state: message history, LLM-call counter, planner steps, review flag."""

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    plan: list[str]
    reviewed: bool


class AgentGraph:
    """Compiled graph plus an optional Docker sandbox for the whole loop."""

    def __init__(self, compiled, sandbox: WorkspaceSandbox | None = None):
        """Wrap a compiled LangGraph and optional session sandbox."""
        self._compiled = compiled
        self.sandbox = sandbox

    def invoke(self, *args, **kwargs):
        """Start the sandbox (if any), then run the compiled graph once."""
        if self.sandbox is not None:
            self.sandbox.ensure()
        return self._compiled.invoke(*args, **kwargs)

    def close(self) -> None:
        """Tear down the session container. Safe to call more than once."""
        if self.sandbox is not None:
            self.sandbox.close()
            self.sandbox = None

    def __enter__(self):
        """Context manager: ensure the sandbox is up."""
        if self.sandbox is not None:
            self.sandbox.ensure()
        return self

    def __exit__(self, *exc):
        """Context manager: always close the sandbox."""
        self.close()

    def __getattr__(self, name):
        """Forward unknown attrs to the compiled graph (e.g. get_graph)."""
        return getattr(self._compiled, name)


def build_loop_graph(
    llm=None,
    tools=None,
    workspace: str | Path | None = None,
    max_llm_calls: int = MAX_LLM_CALLS,
    sandbox: WorkspaceSandbox | None = None,
    review_prompt: str | None = None,
):
    """ReAct loop: agent node -> tool node -> back, until the model stops calling tools.

    `sandbox` replaces the default python-image sandbox (e.g. a project's own
    image); the file tools are then jailed to `sandbox.workspace`.
    `review_prompt` adds one self-review turn before the run may end.
    """
    llm = llm or get_chat_model()
    if tools is not None:
        all_tools = list(tools)
    else:
        sandbox = sandbox or WorkspaceSandbox(workspace or DEFAULT_WORKSPACE)
        all_tools = build_file_tools(
            sandbox.workspace, run_bash=sandbox.run, mount=sandbox.workdir
        )
    model_with_tools = llm.bind_tools(all_tools)
    tools_by_name = {tool.name: tool for tool in all_tools}
    planner_structured = llm.with_structured_output(Plan)

    def planner(state: LoopState) -> dict:
        """Write a short plan once, then hand off to the ReAct loop."""
        result = planner_structured.invoke(
            [SystemMessage(content=PLANNER_PROMPT), *state["messages"]]
        )
        return {
            "plan": result["plan"],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def llm_call(state: LoopState) -> dict:
        """Ask the bound model for the next reply or tool calls."""
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
        """Run each pending tool call and append ToolMessages."""
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name[call["name"]]
            try:
                observation = str(tool.invoke(call["args"]))
            except Exception as exc:
                observation = f"Error: {type(exc).__name__}: {exc}"
            results.append(
                ToolMessage(
                    content=truncate_head_tail(observation),
                    tool_call_id=call["id"],
                )
            )
        return {"messages": results}

    def review(state: LoopState) -> dict:
        """Hand the finished-looking run a self-review checklist, once."""
        return {"messages": [HumanMessage(content=review_prompt)], "reviewed": True}

    def should_continue(state: LoopState) -> Literal["tool_node", "review", "__end__"]:
        """After the LLM: tools if it called any; else review once (if set), else END."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tool_node"
        if review_prompt and not state.get("reviewed"):
            return "review"
        return END

    def after_tools(state: LoopState) -> Literal["llm_call", "__end__"]:
        """After tools: another LLM step unless the call cap is hit."""
        if state.get("llm_calls", 0) >= max_llm_calls:
            return END
        return "llm_call"

    builder = StateGraph(LoopState)
    builder.add_node("planner", planner)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_node("review", review)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "review", END])
    builder.add_edge("review", "llm_call")
    builder.add_conditional_edges("tool_node", after_tools, ["llm_call", END])
    return AgentGraph(builder.compile(), sandbox)
