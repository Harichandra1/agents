# The Day-N ritual

Copy this for every build day from Day 6 onward. It exists so you never spend
a build day deciding where a file goes.

## Before you start

```bash
git checkout main && git pull
git checkout -b day/NN-<slug>       # NN = the PLAN's day number
```

> Week 1's branches were named off-by-one from the plan (`Day/2` held both
> Day 1 and Day 2). Use the plan's numbering, always.

## The four files a day produces

| File | What goes in it |
|---|---|
| `src/harness/<name>.py` | The pattern, as a `build_<name>_graph()` factory |
| `examples/dayNN_<name>.py` | A runnable demo with `main()` |
| `tests/test_<name>.py` | One offline test using `ScriptedLLM` |
| `field-notes.md` | One paragraph appended under `## Day NN — <name>` |

Not every day needs all four — an infra day (caching, tracing) may have no
example — but the module and the test are non-negotiable.

## The module skeleton

```python
"""
Day NN — <pattern name>.

<What it does. Why it exists. What it becomes later in the plan.>
"""
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from harness.config import get_chat_model


class <Name>State(TypedDict):
    ...


def build_<name>_graph(llm=None):
    llm = llm or get_chat_model()          # injectable -> testable

    def some_node(state: <Name>State) -> dict:
        return {"key": value}              # a delta, never mutated state

    builder = StateGraph(<Name>State)
    builder.add_node("some_node", some_node)
    builder.add_edge(START, "some_node")
    builder.add_edge("some_node", END)
    return builder.compile()
```

## Before you commit

```bash
uv run pytest -q
uv run ruff check src examples tests
uv run python -c "import harness.<name>"     # must have no side effects
```

## Close the day

```bash
git add -A
git commit -m "Day NN — <pattern>"
git checkout main && git merge day/NN-<slug> && git push
```

**Merge the same day.** A branch is for work in progress, not for storage.
Five unmerged branches is how Week 1 ended up in five different styles.

## On rest days (6, 12, 18)

Reorganising is *not* a build-day activity. Save it for these:

- Bring anything that drifted back to `CONVENTIONS.md`.
- Promote any exercise code the capstone will import into a real module.
- Re-read `field-notes.md` and fix anything you'd no longer stand behind.
