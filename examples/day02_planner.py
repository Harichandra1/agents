"""Day 2 demo — plan-and-execute vs. a pure-reactive answer.

Prompts for plan approval on stdin. Pass --auto to approve automatically.
"""
import sys

from langchain_core.messages import HumanMessage

from harness.planner import build_planner_graph


def main() -> None:
    auto = "--auto" in sys.argv
    graph = (
        build_planner_graph(approve=lambda plan: True)
        if auto
        else build_planner_graph()
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I want to choose a laptop under Rs 80,000. "
                        "Compare a few options and recommend one."
                    )
                )
            ],
            "llm_calls": 0,
        }
    )

    print("=== plan ===")
    for step in result.get("plan", []):
        print(" -", step)
    print("\n=== planned answer ===")
    print(result["messages"][-1].content)
    print("\n=== reactive answer (no plan) ===")
    print(result.get("normal_response", ""))


if __name__ == "__main__":
    main()
