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
uv run python examples/day05_context.py       # lazy tools + scratchpad + compaction
```

These call a real model and need `OPENAI_API_KEY`.

## Tests

Offline — scripted fake models, no API key, no network:

```bash
uv run pytest -q
uv run ruff check src examples tests
```

## Day 5 architecture

```mermaid
flowchart TD
    START([START]) --> COMPACT[compact]
    COMPACT --> AGENT[agent<br/>binds always-on + loaded tools]
    AGENT -->|tool calls| TOOLS[ToolNode]
    TOOLS --> COMPACT
    AGENT -->|no tool calls| DONE([END])

    TOOLS --> SCRATCH[(scratchpad)]
    COMPACT --> SUM[summarizer LLM<br/>only past threshold]
    SUM --> COMPACT
```

`compact` runs before **every** agent call, not just after tool runs —
otherwise a long tool-free conversation would never compact. It is a no-op
until the message list passes both configured limits (default: keep the last
6 messages, compact past ~3,000 estimated tokens).

## Configuration

Everything reads from `src/harness/config.py`, overridable via env or `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `HARNESS_MODEL` | `openai:gpt-5.6-luna` | Main chat model |
| `HARNESS_SUMMARIZER` | same as above | Compaction/summarisation (Day 9 splits this) |
| `HARNESS_TEMPERATURE` | `0` | Default temperature |

## Week 1 history

Days 1–5 were originally built on separate branches in five different styles.
They were normalised into this structure on Day 6; the originals are preserved
as tags:

```bash
git show archive/day-3:Fanout.py     # any original file, still there
```

See `field-notes.md` for what each day taught, and `CONVENTIONS.md` for the
style all of it was normalised to.
