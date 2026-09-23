# Production Coding-Agent Harness

A coding-agent harness built on raw LangGraph, developed over a one-month
plan: pattern fluency → production-infra modules → minimal harness + baseline
benchmark → error-analysis improvement loop → deploy.

**Week 1 (pattern fluency) is complete.** Each day's pattern is an importable
module with a runnable demo and an offline test.

## Layout

```
src/harness/          the patterns, importable
  config.py           models + settings (single source of truth)
  loop.py             Day 1  bare ReAct loop + StateGraph equivalent
  planner.py          Day 2  plan-and-execute vs. reactive
  fanout.py           Day 3  Send fan-out -> judge -> select
  subagents.py        Day 4  delegation with context isolation
  context/            Day 5  lazy tools, scratchpad, compaction
  memory.py           Day 7  long-term memory + short-term graph state
  tools/              tool definitions shared across days
examples/             one runnable demo per day
tests/                offline tests (no API key needed)
notes/                reading notes + LangGraph API notes
docs/day-template.md  the ritual for every build day
CONVENTIONS.md        the house style
field-notes.md        what clicked / what broke, per day
```

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env      # then fill in your keys
```

## Run any day

```bash
uv run python examples/day01_loop.py          # bare loop vs. graph
uv run python examples/day02_planner.py       # add --auto to skip the approval prompt
uv run python examples/day03_fanout.py        # parallel branches + judge
uv run python examples/day04_subagents.py     # direct vs. delegated
uv run python examples/day07_memory.py        # user memory via LangGraph store
```

These call a real model and need `OPENAI_API_KEY`.

## Tests

Offline — scripted fake models, no API key, no network:

```bash
uv run pytest -q
uv run ruff check src examples tests
```

## Day 7 architecture

```mermaid
flowchart TD
    USER[User message] --> STATE[Short-term state<br/>messages + llm_calls]
    STATE --> LLM[llm_call]
    LLM -->|tool calls| TOOLS[ToolNode]
    TOOLS --> STATE
    LLM -->|no tool calls| DONE([END])

    LLM --> SEARCH[Long-term memory<br/>store.search by user]
    SEARCH --> MEMORY[(User memory store)]
    MEMORY --> PROMPT[Prompt injects relevant memories]
    PROMPT --> LLM

    LLM --> SAVE[save_memory]
    SAVE --> MEMORY
```

Short-term memory is the graph state: recent messages and tool-call count. Long-term
memory is the persisted user namespace in `runtime.store`, keyed by `user_id` and
searched before the model responds. The demo stores only simple facts like
"my name is ..." and retrieves them later when relevant.

## Configuration

Everything reads from `src/harness/config.py`, overridable via env or `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `HARNESS_MODEL` | `openai:gpt-5.6-luna` | Main chat model |
| `HARNESS_SUMMARIZER` | same as above | Compaction/summarisation (Day 9 splits this) |
| `HARNESS_TEMPERATURE` | `0` | Default temperature |
| `HARNESS_REASONING_EFFORT` | `none` | Required for gpt-5.x when binding tools (see below) |

> **Why `reasoning_effort=none`:** the gpt-5.x reasoning models reject function
> tools on `/v1/chat/completions` unless reasoning effort is `none`. Every day
> from Day 1 on binds tools, so `config.py` sets it once for the whole repo.

## Week 1 history

Days 1–5 were originally built on separate branches in five different styles.
They were normalised into this structure on Day 6; the originals are preserved
as tags:

```bash
git show archive/day-3:Fanout.py     # any original file, still there
```

See `field-notes.md` for what each day taught, and `CONVENTIONS.md` for the
style all of it was normalised to.
