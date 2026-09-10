"""Day 1 demo — the same task through the bare loop and the graph."""
from langchain_core.messages import HumanMessage

from harness.loop import build_loop_graph, run_bare_loop


def main() -> None:
    question = "Add 3 and 4, then multiply the result by 5."

    print("=== bare loop (no framework) ===")
    for message in run_bare_loop([HumanMessage(content=question)]):
        message.pretty_print()

    print("\n=== same thing as a StateGraph ===")
    result = build_loop_graph().invoke(
        {"messages": [HumanMessage(content=question)], "llm_calls": 0}
    )
    for message in result["messages"]:
        message.pretty_print()
    print(f"\nllm_calls: {result['llm_calls']}")


if __name__ == "__main__":
    main()
