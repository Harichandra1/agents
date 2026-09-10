"""Day 4 demo — direct answer vs. delegated answer."""
from langchain_core.messages import HumanMessage

from harness.subagents import build_subagent_graph


def run(graph, prompt: str, label: str) -> None:
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "tasks": [],
            "results": [],
            "next_action": "respond",
        }
    )
    print(f"\n=== {label} ===\n")
    print(result["messages"][-1].content)


def main() -> None:
    graph = build_subagent_graph()
    run(
        graph,
        "Explain what context isolation means in multi-agent systems.",
        "direct answer",
    )
    run(
        graph,
        "Research the most important AI trends right now and summarize the findings.",
        "delegated answer",
    )


if __name__ == "__main__":
    main()
