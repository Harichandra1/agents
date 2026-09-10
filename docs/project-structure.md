# The real problem isn't structure, it's that you have no git

You're not confused about folders. You're confused because editing a file destroys the previous version, so your instinct says "make a new file to be safe." That instinct is correct given your setup — and it disappears entirely the moment you have version control. Git makes history free, so you stop hoarding files.

Right now: that folder is not a repo. I rewrote your tracker in place, added two files, and the only copy of the old version is in a temp dir I happened to make. That's the whole anxiety in one example.

## Answer: two zones, different rules

"Should I make a new file per task?" — depends which zone you're in. This is the distinction you're missing:

```text
┌────────────────────┬──────────────────────────────┬───────────────────────────┐
│                    │          Exercises           │          Modules          │
├────────────────────┼──────────────────────────────┼───────────────────────────┤
│ Example            │ exercises/day03_fan_out.py   │ src/harness/routing.py    │
├────────────────────┼──────────────────────────────┼───────────────────────────┤
│ New file per task? │ Yes — one per day            │ No — edit in place        │
├────────────────────┼──────────────────────────────┼───────────────────────────┤
│ After the day ends │ Frozen. Never touched again. │ Lives and changes forever │
├────────────────────┼──────────────────────────────┼───────────────────────────┤
│ History lives in   │ The filename                 │ Git                       │
└────────────────────┴──────────────────────────────┴───────────────────────────┘

```

Week 1–2 are learning days — one file per day is right, and messy is fine. Week 3+ is building — you edit in place and let git remember.

Random is the problem, not new. The test: could you guess the filename without looking? src/harness/routing.py for Day 9 routing — yes. test2_final_v3.py — no.

## Your plan already specifies the structure

You wrote the answer yourself and didn't notice. Three lines in it are structural requirements:

* Day 0: "Scaffold the repo (uv/poetry, git init)"
* Day 12: "Finalize module APIs so Week 3 can just import them"
* Day 22: "GitHub Actions ... on every push"

Day 12 is impossible if Weeks 1–2 are twelve disconnected scratch files. Day 22 is impossible without one repo. So: one repo, importable modules. Not a preference — a constraint you already set.

```text
agent-harness/                 ← ONE repo, git init on Day 0
├─ pyproject.toml
├─ README.md                    ← the Day 24 deliverable
├─ field-notes.md               ← ONE file, append-only, dated headings
├─ src/harness/
│  ├─ loop.py                    Day 1     ├─ memory.py    Day 7
│  ├─ planner.py                 Day 2     ├─ cache.py     Day 8
│  ├─ subagents.py               Day 4     ├─ routing.py   Day 9
│  ├─ context.py                 Day 5     ├─ sandbox.py   Day 10
│  ├─ config.py                  Day 10    └─ tracing.py   Day 11
│  └─ tools/                     Day 13
├─ exercises/                    ← frozen daily scratch, day-stamped
├─ evals/
│  ├─ tasks/                     Day 15
│  └─ results/                   ← dated, append-only, NEVER overwrite
└─ .github/workflows/evals.yml  Day 22

```

Notice src/harness/ is just your Week 1–2 calendar. The structure isn't a decision you need to make — it falls out of the plan.

Two repos, not one. The harness is your portfolio piece; its commit history and README are part of the deliverable. Don't mix a 283KB HTML file with an embedded oil painting into it. The tracker stays here as its own small repo.

## Five rules that will carry you

1. git init before anything else, both folders. One commit per build day, message = Day 09 — routing. Now "what did Day 9 look like" is answerable forever, and you can delete freely.
2. Never reorganize on a build day. Your plan gives you rest days 6, 12, 18 — those are the cleanup rituals. Reorganizing is not Core work.
3. Results are append-only. Never overwrite a benchmark number. Your before/after delta is the single most convincing line on your resume (your plan says so) — you cannot reconstruct it if you clobbered the baseline.
4. One field-notes.md, not one note file per day. It feeds the README directly.
5. .gitignore from minute one: .env, .venv/, __pycache__/, .DS_Store, Icon? (you have a stray 0-byte Icon file in there now — macOS junk).

## On "go flow whatever or what"

Spend 30 minutes on Day 0 building the empty skeleton, then never think about it again. Structure decided up front is what lets you flow — you open a build day and already know where the file goes. Structure decided during a build day is what breaks flow. That's the whole trade.

Want me to scaffold the harness repo skeleton, or git init this folder and commit the tracker (with the original as the first commit, so today's redesign shows as a real diff)? Say which and I'll do it — I've changed nothing so far.