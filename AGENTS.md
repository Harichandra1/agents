# AGENTS.md

Guidance for coding agents working in this repo. Humans: start with `README.md`.

## What this is

A LangGraph-based harness of **reusable coding-agent patterns**, built one
pattern per "day" over a month-long plan. Each `src/harness/<name>.py` module is
one pattern exposed as a graph-builder factory; each `examples/dayNN_<name>.py`
is a runnable demo of it; each test drives it offline with a scripted fake model.

It is a teaching artifact and a starting point — **not** a monolithic app. Do not
try to unify the modules into one abstraction.

## Setup and commands

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync                                        # install deps + the package
uv run pytest -q                               # offline test suite (no API key)
uv run ruff check src examples tests           # lint (line-length 100; E,F,I,UP,B)
uv run python -c "import harness.<name>"       # must have NO side effects
uv run python examples/day01_loop.py           # live demo — needs OPENAI_API_KEY
```

Always run `pytest` **and** `ruff` before committing. For anything that touches
model calls, also run the matching example at least once: the scripted fake
models in tests cannot catch a provider-side 400 (this is how the
`reasoning_effort` bug in Day 6 escaped the suite).

## Layout

```text
src/harness/            the patterns (edit in place; git holds the history)
  settings.py           pydantic-settings Settings — the real source of config
  config.py             get_chat_model / get_routing_model / prompt-cache helpers
  loop.py               Day 1  — bare ReAct loop + StateGraph version
  planner.py            Day 2  — plan-then-execute vs. reactive control arm
  fanout.py             Day 3  — Send fan-out, Pydantic judge, best-of-N select
  subagents.py          Day 4  — parent/worker delegation with context isolation
  context/              Day 5  — lazy tool loading, scratchpad, compaction
  memory.py             Day 7  — long-term memory, prompt caching, streaming metrics
  routing.py            Day 9  — cost-aware model routing
  production.py         Day 11 — Docker sandboxing + OpenTelemetry tracing
  metrics.py            usage extraction, latency/TTFT/TPS, cache hit accounting
  tools/                shared tools: arithmetic, registry, stubs, scratchpad,
                        meta (search_tools/load_tools), sandboxing (Docker)
examples/dayNN_*.py     frozen daily demos — one per pattern, `main()` + guard
tests/                  offline tests; `ScriptedLLM` lives in tests/conftest.py
docs/day-template.md    the per-day build ritual (follow it for new days)
docs/project-structure.md  why the repo is shaped this way
CONVENTIONS.md          the ten house rules — read before writing a module
field-notes.md          append-only log: what broke each day and what fixed it
```

`README.md`'s layout tree predates `settings.py`, `production.py`,
`tools/sandboxing.py` and `examples/day11_production.py`. Update it when you
touch the structure.

## House rules that matter most

`CONVENTIONS.md` is authoritative; these are the ones most easily violated:

- **Never hardcode a model id.** Call `get_chat_model()` / `get_routing_model()`
  from `harness.config`. Week 1 had four model strings in five files, one
  malformed (`"openai-gpt-5.4-mini"`), which crashed provider inference.
- **A module is a builder, not a script.** Expose `build_<name>_graph(llm=None, ...)`
  and do nothing at import time — no `invoke()`, no `print()`, no I/O, no client
  construction. The `llm=None` injection point is what makes the offline tests possible.
- **Nodes return deltas**, never a mutated `state`. Mutating happens to work and
  breaks under parallel branches.
- **Reducers:** `Annotated[list[AnyMessage], add_messages]` for message history
  (it understands `RemoveMessage`, which compaction needs);
  `Annotated[list[...], operator.add]` for plain accumulators like branch results.
- **Tests stay offline.** Drive graphs with `ScriptedLLM` from `tests/conftest.py`.
  No API key, no network — this suite is meant to be a CI gate.
- **Demos live in `examples/`**, never in the module.
- **One commit per day**, message `Day NN — <pattern>`, merged into `main` the
  same day. Recent commits drifted from this ("Done major"); go back to it.

Adding a new pattern? Follow `docs/day-template.md` — module + offline test are
non-negotiable, example optional for pure-infra days.

## Configuration

`src/harness/settings.py` owns all settings via pydantic-settings: prefix
`HARNESS_`, loaded from `.env` then `.env.{APP_ENV}` (later file wins,
`APP_ENV` ∈ `dev|staging|prod`). Defaults: model `openai:gpt-5.6-luna`, routing
model `openai:gpt-4o-mini`, temperature `0`, `reasoning_effort` `none`, prompt
cache off, cache key `harness-v1`. `OPENAI_API_KEY` is read unprefixed.

`.env` and `.env.{dev,staging,prod}` are gitignored and hold real secrets —
never read them into output, print them, or commit them. `.env.example` is the
template.

Note: `CONVENTIONS.md` rule 3 still says `config.py` calls `load_dotenv()` once.
It no longer does — `settings.py` handles env loading, and the only surviving
`load_dotenv()` is in `examples/day11_production.py`. Fix the rule or the example
on the next rest day rather than copying the pattern into new code.

## Known-broken state (as of 2026-10-12)

Verify before assuming your change caused these:

- `tests/test_days01_04.py::test_routing_graph_injects_state_and_runtime` **fails
  without a running Docker daemon.** Root cause: `src/harness/tools/sandboxing.py`
  calls `docker.from_env()` at module import, and `routing.py` imports it — a
  direct violation of the "no side effects at import" rule. The fix is to make
  the Docker client lazy, not to mock it in the test. 13 of 14 tests pass otherwise.
- `ruff` reports 21 errors (9 `E501`, 7 `I001`, 5 `F401`), concentrated in
  `routing.py` (8) and `examples/day09_routing.py` (5); 12 are auto-fixable.
- Several modules (`loop.py`, `memory.py`, `routing.py`, `production.py`) each
  define their own `LoopState` and `build_loop_graph`. That is intentional —
  they are successive versions of the same pattern, meant to be read side by
  side. Do not deduplicate them.
- `routing.py` and `settings.py` are missing the required module docstring;
  `settings.py`'s is a typo'd one-liner.

## Working style here

- Prefer editing a module in place over adding a `*_v2.py`. Git holds history;
  the `examples/dayNN_*.py` files are the only intentionally frozen, date-stamped
  copies.
- Append to `field-notes.md` under `## Day NN — <name>` when you learn something
  that cost you time. It feeds the README directly.
- Don't reorganize the repo as part of a feature change. Structural cleanup
  belongs on the designated rest days (6, 12, 18).
- Benchmark and eval results are append-only — never overwrite a recorded number.
