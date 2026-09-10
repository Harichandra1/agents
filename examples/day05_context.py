"""Day 5 demo — lazy tool loading, scratchpad and compaction in one run."""
from langchain_core.messages import HumanMessage

from harness.config import get_chat_model, get_summarizer_model
from harness.context.graph import build_graph


def main() -> None:
    app = build_graph(get_chat_model(), get_summarizer_model())

    config = {"configurable": {"thread_id": "demo-thread-1"}}
    initial_state = {
        "messages": [
            HumanMessage(content="What's the weather in Hyderabad, and what's 42 * 17?")
        ],
        "scratchpad": {},
        "loaded_tools": [],
    }

    for event in app.stream(initial_state, config, stream_mode="values"):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
