"""Power demo — one big, staged task: the agent builds a multi-file interactive web app."""
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from harness.agent import build_loop_graph, invoke_config

load_dotenv()

WORKSPACE = Path(__file__).resolve().parent / "workspace" / "showcase"

TASK = """
Build a polished, interactive "Habit Tracker" web app in the workspace, as THREE files:
  index.html, style.css, app.js   (plain HTML/CSS/JS, no libraries, no network).

Work in stages, one file at a time. Do NOT try to write everything in one tool call:
  1. write index.html: semantic structure, links style.css and app.js, an add-habit form,
     a list container, a stats panel, and a 7-day grid area.
  2. write style.css: modern dark theme with CSS variables, responsive grid, hover and
     focus states, smooth transitions, and a prefers-reduced-motion media query.
  3. write app.js: add / delete habits, click a day cell to toggle done, current streak
     and weekly completion % per habit, a simple SVG or canvas bar chart of the last
     7 days, and persistence with localStorage wrapped in try/catch.
  4. Use edit (not a full rewrite) for any fixes or additions after that.
  5. Verify with bash. There is no browser and no network, but python is available:
     check that all 3 files exist, that every id used in app.js (getElementById /
     querySelector('#...')) exists in index.html, and that tags in index.html are balanced
     (use python's html.parser). Fix anything that fails with edit and re-check.

Finish with a short report: files created, their sizes, and the verification results.
""".strip()


def main() -> None:
    """Build the showcase app in one sandbox, then close the container."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    graph = build_loop_graph(workspace=WORKSPACE, max_llm_calls=60)
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=TASK)], "llm_calls": 0},
            config=invoke_config(run_name="power-demo", tags=["power-demo"]),
        )
    finally:
        graph.close()
    print("plan:")
    for step in result.get("plan") or []:
        print(f"  - {step}")
    print("\n--- final answer ---")
    print(result["messages"][-1].content)
    print(f"\nllm_calls: {result['llm_calls']}")
    print(f"open: {WORKSPACE / 'index.html'}")


if __name__ == "__main__":
    main()
