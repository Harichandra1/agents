"""Day 1 demo — the same task through the bare loop and the graph."""
from langchain_core.messages import HumanMessage

from harness.memory import build_loop_graph


def main() -> None:
    question = "Tell me about Tottenham Hotspur in the london" 

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

    print("\n=== Cache note ===")
    print(
        "OpenAI prompt caching is automatic and model/account dependent. "
        "If this model/account does not expose cache metadata, the correct result is "
        "a zero cache_read value even when the request is otherwise valid."
    )


if __name__ == "__main__":
    main()
