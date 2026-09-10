# House style

Every module in `src/harness/` follows these rules. They were extracted from
Day 5 — the best code of Week 1 — and applied backwards to Days 1–4 so the
whole repo reads as one project.

When you start Day 6+, follow `docs/day-template.md` and these ten rules.

## 1. Imports come from `langchain_core`

```python
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
```

Not `langchain.messages`. Week 1 used both; `langchain_core` is the stable
public path.

## 2. Never hardcode a model string

```python
from harness.config import get_chat_model
llm = llm or get_chat_model()
```

Week 1 had four different model ids across five files, one of them malformed
(`"openai-gpt-5.4-mini"`) which crashed at runtime because LangChain could not
infer a provider from it. One place to change now.

## 3. `config.py` owns settings

Model ids, temperature, thresholds. Env-overridable. `load_dotenv()` is called
**exactly once**, there — never in a module or an example.

## 4. A module is a builder, not a script

Every module exposes `build_<name>_graph(...)` and does **nothing** at import
time. No `invoke()`, no `print()`, no file writes at module level.

```python
def build_planner_graph(llm=None, ...):
    llm = llm or get_chat_model()
    ...
    return builder.compile()
```

Injectable `llm=None` parameters are what make the offline tests possible.

## 5. Nodes return deltas, never mutated state

```python
def judge(state) -> dict:
    return {"scores": {...}}       # yes

def judge(state):
    state["scores"] = {...}        # no
    return state
```

Day 3 originally mutated in place. It happened to work, but it is not the
LangGraph contract and breaks under parallel branches.

## 6. Pick the right reducer

- `Annotated[list[AnyMessage], add_messages]` for message history (it also
  understands `RemoveMessage`, which compaction needs).
- `Annotated[list[...], operator.add]` for plain accumulator lists, e.g.
  parallel branch results.

## 7. Every module opens with a docstring

What the pattern is, why it exists, and what it becomes later in the plan.
Write it for yourself in Week 4, who will not remember Week 1.

## 8. Demos live in `examples/`

One `examples/dayNN_<name>.py` per day, with `main()` and an
`if __name__ == "__main__":` guard. Never in the module.

## 9. Tests are offline

Drive graphs with `ScriptedLLM` from `tests/conftest.py`. No API key, no
network. This is what makes `pytest` usable as a CI gate on Day 22.

## 10. One commit per day, merged the same day

`Day NN — <pattern>`, merged into `main` when the day ends. Week 1's five
unmerged branches are exactly the problem this avoids.
