"""
Day 2 — plan-and-execute, compared against pure-reactive execution.

The graph runs both paths from START so you can read them side by side on the
same request:

    planner_agent -> [human gate] -> executor_agent   (plan, then execute)
    normal_agent                                      (just answer)

The human gate is injected rather than hardcoded to `input()`, so the module
imports cleanly and can be tested without a terminal. Pass
`approve=lambda plan: True` to run unattended; the default prompts.

This becomes the Planner node in the capstone architecture.
"""
import operator
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from harness.config import get_chat_model


class Plan(TypedDict):
    """Structured-output schema for the planner."""

    plan: list[str]


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    plan: list[str]
    normal_response: str


def prompt_for_approval(plan: list[str]) -> bool:
    """Default human gate: print the plan and ask on stdin."""
    print("-" * 32)
    print("Are you okay with this plan?")
    print("-" * 32)
    for step in plan:
        print(step)
    print("-" * 32)
    return input("Enter 'yes' or 'no' by looking at the plan: ").strip().lower() == "yes"


def build_planner_graph(
    llm=None,
    planner_llm=None,
    approve: Callable[[list[str]], bool] = prompt_for_approval,
):
    """
    approve: callable taking the plan and returning True to execute it,
        False to re-plan. Defaults to an interactive stdin prompt.
    """
    llm = llm or get_chat_model()
    planner_llm = planner_llm or llm
    planner_structured = planner_llm.with_structured_output(Plan)

    def planner_agent(state: PlannerState) -> dict:
        """Produce an explicit, ordered plan before any execution happens."""
        result = planner_structured.invoke(
            [
                SystemMessage(content="You are planner agent, you plan tasks"),
                *state["messages"],
            ]
        )
        return {"plan": result["plan"]}

    def human_feedback(state: PlannerState) -> str:
        """Route on the injected approval gate: execute, or plan again."""
        return "executor_agent" if approve(state["plan"]) else "planner_agent"

    def normal_agent(state: PlannerState) -> dict:
        """The control arm: answer directly, with no plan at all."""
        response = llm.invoke(
            [SystemMessage(content="You are executor agent"), *state["messages"]]
        )
        return {"normal_response": response.content}

    def executor_agent(state: PlannerState) -> dict:
        """Execute the approved plan step by step."""
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state["plan"]))
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are executor agent. Execute the user's request by "
                        f"following this plan:\n{plan_text}\n\n"
                        "Address each step and produce a final, complete answer."
                    )
                ),
                *state["messages"],
            ]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    builder = StateGraph(PlannerState)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("executor_agent", executor_agent)
    builder.add_node("normal_agent", normal_agent)

    builder.add_edge(START, "planner_agent")
    builder.add_edge(START, "normal_agent")
    builder.add_conditional_edges(
        "planner_agent", human_feedback, ["planner_agent", "executor_agent"]
    )
    builder.add_edge("executor_agent", END)
    builder.add_edge("normal_agent", END)
    return builder.compile()
