"""Day 14 demo — toy tasks through the minimal harness, traced in LangSmith."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

from harness.agent import build_loop_graph, invoke_config

load_dotenv()

WORKSPACE = Path(__file__).resolve().parent / "workspace"

TASKS = [
    # (
    #     "sum-of-squares",
    #     "Write src/sum_of_squares.py with a function that returns the sum of "
    #     "squares of a list of numbers. Run it on [1, 2, 3, 4] with bash "
    #     "(python src/sum_of_squares.py) and report the number.",
    # ),
    # (
    #     "fizzbuzz-count",
    #     "Write src/fizzbuzz.py that prints Fizz/Buzz/FizzBuzz counts for n in "
    #     "range(1, 31). Run it with bash and report the three counts.",
    # ),
    # (
    #     "anagram-edit",
    #     "Write src/anagrams.py with a function that returns True if two strings "
    #     "are anagrams (ignore case and spaces). Run it on ('Listen', 'Silent') "
    #     "and ('hello', 'world'). If a check fails, fix it with edit, re-run, "
    #     "and report both booleans.",
    # ),
    (
    "html-page",
    "Create site/index.html: a simple page with a title 'Hello' and a button. "
    "Verify it exists with bash (ls site) and report the file size.",
    ),
]


def main() -> None:
    """Run TASKS in one workspace; close the Docker sandbox when finished."""
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    project = os.getenv("LANGSMITH_PROJECT", "agent")
    print(f"LangSmith tracing: {'on' if tracing else 'off'} (project={project})")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    graph = build_loop_graph(workspace=WORKSPACE)
    try:
        for slug, question in TASKS:
            print(f"\n=== {slug} ===")
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)], "llm_calls": 0},
                config=invoke_config(
                    run_name=f"day14-{slug}",
                    tags=["day14", "minimal-harness"],
                    metadata={"task": slug},
                ),
            )
            print("plan:")
            for step in result.get("plan") or []:
                print(f"  - {step}")
            for message in result["messages"]:
                message.pretty_print()
            print(f"\nllm_calls: {result['llm_calls']}")
    finally:
        graph.close()

    wait_for_all_tracers()


if __name__ == "__main__":
    main()
