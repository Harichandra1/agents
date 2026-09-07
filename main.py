from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from graph import build_graph


def main():
    chat_llm = init_chat_model("openai:gpt-5.6-luna", temperature=0)
    # Summarization doesn't need your strongest/most expensive model.
    summarizer_llm = init_chat_model("openai:gpt-5.6-luna", temperature=0)

    app = build_graph(chat_llm, summarizer_llm)

    config = {"configurable": {"thread_id": "demo-thread-1"}}
    initial_state = {
        "messages": [HumanMessage(content="What's the weather in Hyderabad, and what's 42 * 17?")],
        "scratchpad": {},
        "loaded_tools": [],
    }

    for event in app.stream(initial_state, config, stream_mode="values"):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
