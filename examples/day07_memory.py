"""Day 1 demo — the same task through the bare loop and the graph."""
from langchain_core.messages import HumanMessage

from harness.memory import build_loop_graph


def main() -> None:
    question = "Hi my name is Hari do this task for me , Add 3 and 4, then multiply the result by 5."

    print("\n=== same thing as a StateGraph ===")
    config={
        "configurable":{
            "thread_id": "day06_memory",
        }
    }
    graph = build_loop_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "llm_calls": 0},
        context={"user_id":"hari-123"},
        config=config
    )
    for message in result["messages"]:
        message.pretty_print()
    print(f"\nllm_calls: {result['llm_calls']}")

    print("\n=== Now testing the checkpointer working capability ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is my Name?")]},
        context={"user_id":"hari-123"},
        config=config
    )
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
