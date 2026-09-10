# Field notes

One entry per build day: what clicked, what broke, and what fixed it. This is
the raw material for the final README's "patterns used" and "lessons learned"
sections, so write it the day it happens.

---

## Week 1 — Pattern Fluency

### Day 1 — Bare loop → LangGraph

Wrote the ReAct loop twice: once as a raw `while` loop with manual tool
dispatch, once as a `StateGraph`. Both live in `src/harness/loop.py` so they
can be read side by side. The bare version is about twenty lines and makes the
mechanic obvious — call the model, look for `tool_calls`, dispatch, append a
`ToolMessage`, repeat. What LangGraph actually buys is not the loop; it is
routing, state reduction, checkpointing and streaming, none of which the bare
version has. One real bug surfaced here: the arithmetic tools return `int`,
but `ToolMessage.content` must be a `str`.

### Day 2 — Planning pattern

Plan-and-execute against a pure-reactive control arm, both fanned out from
`START` on the same request so the difference is visible in one run. The
planner uses `with_structured_output` to get a `list[str]` plan rather than
free text. The lesson that generalised: the human approval gate should be an
injected callable, not a hardcoded `input()` — as originally written the
module could not be imported or tested without a terminal attached. This
becomes the Planner node in the capstone.

### Day 3 — Fan-out + judge ("Agent Fan Wars")

`Send` dispatches one isolated state per branch, and an
`Annotated[list, operator.add]` reducer merges the parallel returns. The judge
started as free text with a hand-rolled line parser that silently dropped
anything unparseable; swapping it for a Pydantic `with_structured_output`
schema deleted the parser entirely. Also learned the LangGraph contract
properly here: nodes must return a **delta**, not a mutated `state`. The
original mutated in place and worked by luck. Reused in Week 4 as best-of-N
patch generation.

### Day 4 — Sub-agent delegation

The parent decides between answering and delegating; each worker gets only its
own instruction via `Send`, never the parent's history. That isolation is the
whole point — it is how Claude Code and Deep Agents avoid re-paying for a long
parent conversation on every sub-task. Two bugs made this day non-functional
as first written, and both are worth remembering: the "tools" were plain
undecorated functions, so `tool.name` raised `AttributeError`; and two of the
three registry model ids were `"openai-gpt-5.4-mini"`, which fails provider
inference (LangChain only matches the prefixes `gpt-`, `o1`, `o3`, `chatgpt`,
`text-davinci`) and raises. Any delegation crashed. Also worth internalising:
`bind_tools()` only *exposes* tools to the model — it does not execute them.
You still write the loop.

### Day 5 — Context engineering

Three techniques in one graph: lazy tool loading (only `search_tools` and
`load_tools` stay resident; everything else is invisible until loaded),
a scratchpad virtual filesystem so large blobs never re-enter the message
history, and compaction that summarises old turns and deletes them with
`RemoveMessage`. The subtle bit is that `compact` sits in front of *every*
agent call, not just after tool runs — otherwise a long tool-free conversation
would never compact at all. Compaction removes every message and re-adds the
summary plus fresh copies of the recent ones with new ids, rather than
trusting where the reducer would append.

### Day 6 — Rest / consolidate

Normalised all of Week 1 into one codebase. The root cause of the mess was
structural, not stylistic: five day-branches all forked from near-root and
were **never merged**, and each day deleted the previous day's files, so no
single checkout had ever contained Week 1. Standardised on Day 5's style
(builder functions, injectable models, module docstrings, offline tests),
centralised every model id into `config.py`, and wrote `CONVENTIONS.md` plus
`docs/day-template.md` so later days don't re-litigate structure. Fixed the
Day 4 bugs above, plus a dependency one: `pyproject.toml` listed `dotenv`
(a package whose own PyPI summary is "Deprecated package") instead of
`python-dotenv` — imports had been working only because something else pulled
the real one in transitively.

**Rule adopted:** merge each day into `main` the same day.
