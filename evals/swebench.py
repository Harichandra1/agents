"""
Day 17 — run SWE-bench Verified Mini instances with the Day 13 coding agent.

    uv run python -m evals.swebench --phase 1
        Mini tasks 1–10 (dataset index 0–9).

    uv run python -m evals.swebench --from-report my-harness-day17.mini-rest.json
        Re-run every instance a grading report lists as unresolved/error/empty.

    uv run python -m evals.swebench --instance-ids django__django-11848 --workers 2

    --with-hints adds the dataset's hints_text (issue comments from before the
    fix) to the prompt: the "with hints" SWE-bench setting, not comparable with
    leaderboards that omit them. Each run's results.jsonl records the flag.
    The agent always gets one self-review turn (REVIEW_PROMPT) unless --no-review.

Each run writes to a fresh evals/results/<UTC stamp>/ (results are
append-only). Pass --run-dir to resume a run after a crash or credit refill:
instance_ids already in its predictions.jsonl are skipped.

Each instance runs inside its official SWE-bench image
(swebench/sweb.eval.x86_64.<id>, linux/amd64; Rosetta on Apple silicon).
The image's /testbed is copied to a host workspace and mounted back at
/testbed with the `testbed` conda env active, so bash can import the project
and run its real tests. The first pass ran bash in a bare python:3.11-slim
with no project deps, so the agent never ran one test of the repo it fixed.

Grade with `python -m evals.grade`, not run_evaluation directly; see there.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_core.tracers.langchain import wait_for_all_tracers
from pydantic import BaseModel, Field

from evals.runner import capture_patch, unique_run_dir
from harness.agent import build_loop_graph, invoke_config
from harness.config import get_chat_model
from harness.settings import settings
from harness.tools.coding import WorkspaceSandbox, install_cleanup_hooks

DATASET = "MariusHobbhahn/swe-bench-verified-mini"
MODEL_NAME = "my-harness-day17"
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "swebench"
PHASE_SIZE = 10
MINI_TOTAL = 50

IMAGE_NAMESPACE = "swebench"
IMAGE_PLATFORM = "linux/amd64"
REPO_DIR = "/testbed"
CONDA_PRELUDE = "source /opt/miniconda3/bin/activate\nconda activate testbed"
# Old Django/Sphinx test modules are slow under amd64 emulation.
SANDBOX_TIMEOUT_S = 600
SANDBOX_MEM = "4g"
SANDBOX_CPUS = 2.0
MAX_LLM_CALLS = 100
# Client-side retries with backoff ride out tokens-per-minute 429s and blips.
MODEL_RETRIES = 8
PULL_RETRIES = 3

# Gold patches never touch tests (SWE-bench moves those into test_patch, which
# the grader applies itself), so agent-written tests can only collide with it.
TEST_DIR_NAMES = {"test", "tests", "testing", "e2e"}

_WRITE_LOCK = threading.Lock()


def load_mini():
    from datasets import load_dataset

    return load_dataset(DATASET, split="test")


def instance_image(instance_id: str) -> str:
    """Published image name, as swebench's TestSpec.instance_image_key builds it.

    Not via make_test_spec: that also downloads each repo's requirements from
    raw.githubusercontent.com to build env scripts we never use, and stalled
    a whole run when GitHub was slow.
    """
    key = f"sweb.eval.x86_64.{instance_id.lower()}:latest"
    return f"{IMAGE_NAMESPACE}/{key}".replace("__", "_1776_")


def test_command(row: dict) -> str:
    """The exact command the grader runs, minus the hidden test patch."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.python import get_test_directives

    spec = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
    return " ".join([spec["test_cmd"], *get_test_directives(row)])


def sandbox_env(row: dict) -> tuple[list[str], str]:
    """(one-time setup commands, per-command prelude) from the eval script."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    extra = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]].get("eval_commands", [])
    exports = [cmd for cmd in extra if cmd.startswith("export ")]
    setup = [cmd for cmd in extra if not cmd.startswith("export ")]
    setup.append(f"git config --global --add safe.directory {REPO_DIR}")
    return setup, "\n".join([CONDA_PRELUDE, *exports])


def _docker(*args: str, quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        check=True,
        text=True,
        capture_output=quiet,
    )


def pull_image(image: str) -> None:
    """Pull the amd64 image once; Docker defaults to arm64 on Apple silicon."""
    probe = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if probe.returncode == 0:
        return
    for attempt in range(1, PULL_RETRIES + 1):
        print(f"pulling {image} ({IMAGE_PLATFORM}) try {attempt}", flush=True)
        try:
            _docker("pull", "--platform", IMAGE_PLATFORM, image, quiet=False)
            return
        except subprocess.CalledProcessError:
            if attempt == PULL_RETRIES:
                raise
            time.sleep(30 * attempt)


def prepare_workspace(image: str, dest: Path) -> Path:
    """Copy the image's /testbed to `dest`: same commit, setup edits and egg-info.

    The grader applies our diff to that same tree, so the diff is taken
    against the image's HEAD (base_commit plus SWE-bench's setup commit).
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    container_id = _docker("create", "--platform", IMAGE_PLATFORM, image).stdout.strip()
    try:
        _docker("cp", f"{container_id}:{REPO_DIR}", str(dest))
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=dest, capture_output=True, text=True
    ).stdout
    if dirty.strip():
        raise RuntimeError(f"workspace not clean after copy:\n{dirty[:2000]}")
    return dest


def build_sandbox(row: dict, image: str, workspace: Path) -> WorkspaceSandbox:
    setup, prelude = sandbox_env(row)
    return WorkspaceSandbox(
        workspace,
        image=image,
        workdir=REPO_DIR,
        platform=IMAGE_PLATFORM,
        setup=setup,
        prelude=prelude,
        timeout_s=SANDBOX_TIMEOUT_S,
        mem_limit=SANDBOX_MEM,
        cpus=SANDBOX_CPUS,
    )


def build_prompt(row: dict, test_cmd: str, with_hints: bool = False) -> str:
    failing = "\n".join(f"- {name}" for name in json.loads(row["FAIL_TO_PASS"]))
    hints = (row.get("hints_text") or "").strip()
    hints_block = ""
    if with_hints and hints:
        hints_block = f"""
Comments from the issue thread, written before the fix was made. They often say what the \
maintainers expect; where they disagree, prefer the maintainers' (later) view:
<hints>
{hints}
</hints>
"""
    return f"""You are fixing a real GitHub issue in the repository at the workspace root \
({REPO_DIR}). The project's Python environment is installed in the sandbox: bash can import \
the package and run its tests. There is no network, so do not try to install anything.

Repo: {row["repo"]}
Instance: {row["instance_id"]}

<issue>
{row["problem_statement"]}
</issue>
{hints_block}
The maintainers' tests for this fix. Some may not exist yet, and their final versions may be \
stricter than what is in the repo now. Read the names as clues: a name that mentions a \
parameter, option or input (and parametrized ids like `test_x[input-expected]`) tells you \
what the test exercises, and the test's module and class tell you which component it calls \
directly (a test in `FooTests` mentioning an option means `Foo` itself must accept it):
{failing}

Run the affected test module(s) with:
    {test_cmd}
To run one test, append its selector (`-k name` for pytest/tox, a dotted label for Django's \
runtests.py). Each bash call times out after {SANDBOX_TIMEOUT_S}s.

How to work:
1. Run the test command once before changing anything, so you know which tests already fail \
for unrelated reasons (environment drift). Do not try to fix those.
2. Find the code responsible: grep -rn, then read the relevant line ranges.
3. List every scenario the issue and its comments report as broken, including follow-up \
"simpler case" reports, and reproduce each one with a short python snippet or test before \
fixing. All of them must work after your fix.
4. Fix the root cause in the library source, the way a maintainer would:
   - fix the code that produces the wrong value, not the place where it shows up; then update \
other code that relied on the old behaviour (grep for every caller);
   - key the new behaviour on exactly the condition the issue names (if it says "when X is \
False on the model", the model's X must reach the code that decides);
   - look for precedent first: how does the codebase already handle the most similar feature \
(an existing option, flag, status, output format)? Mirror its naming and mechanism;
   - new code paths must honour the existing settings that apply to the same thing (ignore \
lists, options, flags), and reuse existing statuses and values rather than inventing new ones;
   - keep the user's input as they wrote it; do not rename or translate it;
   - prefer the simplest reading of the issue; do not add precision it does not ask for;
   - reuse the helpers, clocks and APIs the surrounding code already uses;
   - the issue asks for a behaviour change: make it the default behaviour. Do not hide it \
behind a new opt-in setting unless the issue explicitly asks for a setting;
   - new parameters get defaults so existing callers keep working; pass them through from \
the callers that know their value;
   - when you call an existing hook, event or API, pass the same kind of arguments its other \
callers pass (grep them: e.g. an identifier vs a file path), and make them identify the thing \
actually being processed (the included item, not its container);
   - change only the behaviour and output the issue is about: leave other generated text \
(index entries, messages, labels of other cases) exactly as it was;
   - when you add support for a new kind of input, make the output distinguish it the way \
the codebase already distinguishes analogous kinds (the same flags, options or labels);
   - handle every variant the issue mentions or implies (other inputs, invalid values, \
empty values, subclasses), not only the one example.
5. Re-run the test command. Tests unrelated to the issue must still pass; if one breaks, \
change your fix. An existing test that asserts exactly the behaviour the issue reports as wrong \
is expected to change: do not let it hold the bug in place. Update its expectation to the \
fixed behaviour (test edits are discarded; the maintainers' versions are used) and say so in \
your summary. Never weaken a test that is unrelated to the issue.
6. You may write tests to check your fix, but they are discarded: only changes outside test \
directories are submitted, and they are graded by the maintainers' tests.
7. Delete scratch files you created. Do not commit. Finish with a short summary of the root \
cause and your change.
"""


REVIEW_PROMPT = """Before you finish, review your change as a strict maintainer would. \
Answer each point by checking the code or running something, not from memory:

1. Condition: does the new behaviour trigger on exactly the condition the issue names, \
using the same names and terms? Is it on by default (not behind a new opt-in setting the issue \
did not ask for)? Is any new parameter passed through from the callers that know its value?
2. Root cause: is the fix where the wrong value is produced, not only where it surfaces? \
grep for other callers of what you changed; do they still behave correctly? Did you keep any \
old behaviour only because an existing test asserts it? If that test encodes the reported bug, \
fix the behaviour and update the test.
3. Breaking inputs: list inputs that could make your new code raise or misbehave (invalid, \
unserializable, None, empty, nested, subclasses, the issue's other examples). Try them. Fall \
back to the existing behaviour rather than raising where the old code did not raise.
4. Precedent and settings: does a similar feature already exist? Did you mirror its naming, \
output format, flags and statuses (a working result reported like other working results)? \
Does your new path honour the existing options for the same thing (ignore lists, etc.)? Do \
calls into existing hooks/events pass the same kind of arguments as their other callers?
5. Scope: list every output your change alters (rendered text, index entries, messages, \
return values). Is each one something the issue asks to change? Revert the rest.
6. Thread scenarios: does every case the issue and its comments report as broken now work? \
Run each one.
7. Maintainers' tests: for each listed test name, write down what that test most likely \
asserts (inputs, expected output, any mocking such as patching a clock), write it as a quick \
test, and run it together with the test command.
8. Diff: `git diff` must contain only intended source changes; delete scratch files.

Fix anything this turns up and re-run the tests. Then give your final summary."""


def is_test_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1] if parts else ""
    return (
        any(part in TEST_DIR_NAMES for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def split_patch(diff: str) -> tuple[str, list[str]]:
    """Drop test files and new top-level scratch files; return (kept, dropped paths)."""
    chunks = re.split(r"(?m)^(?=diff --git )", diff)
    kept: list[str] = []
    dropped: list[str] = []
    for chunk in chunks:
        match = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
        if not match:
            continue
        path = match.group(2)
        new_file = re.search(r"(?m)^new file mode", chunk) is not None
        # repro.py at the root, or tool output such as .tox/ or .pytest_cache/
        scratch = new_file and ("/" not in path or path.startswith("."))
        if is_test_path(path) or scratch:
            dropped.append(path)
        else:
            kept.append(chunk)
    return "".join(kept), dropped


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_done_ids(path: Path) -> set[str]:
    """Instance ids already stored, so a resumed run does not run them again."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        done.add(json.loads(line)["instance_id"])
    return done


def phase_indices(phase: int, total: int = MINI_TOTAL) -> list[int]:
    """1-based task numbers for a phase: phase 1 is tasks 1–10 (index 0–9)."""
    if phase < 1:
        raise ValueError(f"phase must be >= 1, got {phase}")
    start_task = (phase - 1) * PHASE_SIZE + 1
    end_task = min(phase * PHASE_SIZE, total)
    if start_task > total:
        return []
    return list(range(start_task - 1, end_task))


def report_retry_ids(report_path: Path) -> list[str]:
    """Instance ids a swebench grading report did not resolve."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    keys = ("unresolved_ids", "error_ids", "empty_patch_ids", "incomplete_ids")
    return sorted({iid for key in keys for iid in report.get(key, [])})


def _credit_exhausted(exc: BaseException) -> bool:
    """True only when retrying cannot help: no credit, or bad auth.

    A tokens-per-minute 429 ("try again in 1.2s") is transient and is retried
    by the client (MODEL_RETRIES); treating it as fatal stopped a 50-run at 40.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "insufficient_quota",
        "exceeded your current quota",
        "credit_balance_exhausted",
        "no credits remaining",
        "billing",
        "insufficient_funds",
        "401",
        "invalid_api_key",
    )
    return any(needle in text for needle in needles)


def bench_model(reasoning_effort: str | None = None):
    """The configured chat model with benchmark retries; effort only if overridden."""
    kwargs = {"max_retries": MODEL_RETRIES}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return get_chat_model(**kwargs)


_DJANGO_FAIL = re.compile(r"(?m)^(?:FAIL|ERROR): (\w+ \([\w.]+\))")
_PYTEST_FAIL = re.compile(r"(?m)^(?:FAILED|ERROR) (\S+::\S+)")


def failed_tests(output: str) -> set[str]:
    """Failing test ids from Django runtests or pytest -rA output."""
    return set(_DJANGO_FAIL.findall(output)) | set(_PYTEST_FAIL.findall(output))


def failure_detail(output: str, test: str, max_chars: int = 1200) -> str:
    """The traceback block for one failing test, so a reader can see what it asserts."""
    name = test.split("::")[-1].split(" ")[0]
    for header in (f"FAIL: {test}", f"ERROR: {test}", f" {name} _"):
        start = output.find(header)
        if start != -1:
            end = output.find("\n=====", start + len(header))
            nxt = output.find("\n_____", start + len(header))
            stops = [i for i in (end, nxt) if i != -1]
            return output[start : min(stops) if stops else None][:max_chars]
    return ""


def verification_summary(before: str, after: str, max_tail: int = 1500) -> str:
    """What the repo's own tests say about a candidate, relative to the baseline."""
    was, now = failed_tests(before), failed_tests(after)
    lines = [
        f"failing before the change: {len(was)}; after: {len(now)}",
        "newly failing: " + (", ".join(sorted(now - was)) or "none"),
        "newly passing: " + (", ".join(sorted(was - now)) or "none"),
    ]
    for test in sorted(now - was)[:5]:
        detail = failure_detail(after, test)
        if detail:
            lines += [f"failure of {test}:", detail]
    lines += ["output tail:", after[-max_tail:]]
    return "\n".join(lines)


def revert_test_edits(workspace: Path, dropped: list[str]) -> None:
    """Undo the agent's test edits so verification sees only the submitted patch."""
    for path in dropped:
        tracked = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{path}"], cwd=workspace, capture_output=True
        ).returncode == 0
        if tracked:
            subprocess.run(["git", "checkout", "HEAD", "--", path], cwd=workspace, check=False)
        else:
            (workspace / path).unlink(missing_ok=True)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def run_attempt(
    row: dict,
    index: int,
    attempt: int,
    run_dir: Path,
    image: str,
    test_cmd: str,
    max_llm_calls: int,
    *,
    with_hints: bool,
    reasoning_effort: str | None,
    review: bool,
) -> dict:
    """One independent agent run in its own workspace, plus a verification run."""
    instance_id = row["instance_id"]
    tag = f"{instance_id}.a{attempt}"
    workspace = prepare_workspace(image, CACHE / "workspaces" / instance_id / f"a{attempt}")
    llm = bench_model(reasoning_effort)
    graph = build_loop_graph(
        llm=llm,
        sandbox=build_sandbox(row, image, workspace),
        max_llm_calls=max_llm_calls,
        review_prompt=REVIEW_PROMPT if review else None,
    )
    try:
        baseline = graph.sandbox.run(test_cmd, max_chars=None)
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=build_prompt(row, test_cmd, with_hints))],
                "llm_calls": 0,
            },
            config=invoke_config(
                recursion_limit=2 * max_llm_calls + 10,
                run_name=f"swebench-{tag}",
                tags=["day17", "swebench-mini"],
                metadata={"instance_id": instance_id, "index": index, "attempt": attempt},
            ),
        )
        full_patch = capture_patch(workspace)
        patch, dropped = split_patch(full_patch)
        revert_test_edits(workspace, dropped)
        after = graph.sandbox.run(test_cmd, max_chars=None)
    finally:
        graph.close()

    (run_dir / "patches" / f"{tag}.patch").write_text(patch, encoding="utf-8")
    (run_dir / "patches" / f"{tag}.full.patch").write_text(full_patch, encoding="utf-8")
    (run_dir / "trajectories" / f"{tag}.json").write_text(
        json.dumps(messages_to_dict(result["messages"]), indent=1, default=str),
        encoding="utf-8",
    )
    finals = [
        _text(m.content)
        for m in result["messages"]
        if m.type == "ai" and not getattr(m, "tool_calls", None)
    ]
    llm_calls = int(result.get("llm_calls") or 0)
    print(f"[{tag}] llm_calls={llm_calls} patch={len(patch)}B dropped={dropped}", flush=True)
    return {
        "attempt": attempt,
        "patch": patch,
        "dropped_from_patch": dropped,
        "llm_calls": llm_calls,
        "hit_call_cap": llm_calls >= max_llm_calls,
        "reviewed": bool(result.get("reviewed")),
        "summary": finals[-1] if finals else "",
        "verification": verification_summary(baseline, after),
        "newly_failing": sorted(failed_tests(after) - failed_tests(baseline)),
    }


JUDGE_VOTES = 3


class PatchChoice(BaseModel):
    """The judge's pick among candidate patches."""

    analysis: str = Field(description="Compare the candidates against the issue, point by point")
    best: int = Field(description="1-based number of the candidate to submit")


JUDGE_PROMPT = """You are a senior maintainer of {repo}. Independent attempts fixed the same \
issue; pick the one patch you would merge: the one most likely to match what the maintainers \
merged and to pass the tests they wrote for it. You cannot see those tests; their names are \
below and hint at what they exercise (a test class mentioning an option means that class \
accepts it; parametrized ids encode inputs and expected outputs).

Work in this order:
A. For each maintainer test name, say which component it calls directly and what it most \
likely passes and asserts. Then, for each candidate: could that test even call it (e.g. a \
keyword the test would pass but the candidate does not accept)?
B. List every scenario the issue and its comments report as broken, including follow-up \
"simpler case" reports in the comments. Which candidates fix all of them?
C. For each newly failing repo test, decide whether it asserts behaviour that the issue or its \
comments report as broken. If so, the maintainers changed that test: it is not a negative. \
Any other newly failing test is a strong negative.
D. Then weigh: does it do what the issue asks as the default behaviour, keyed on the \
condition the issue names; root cause rather than symptom; mirrors how the codebase handles \
analogous things (helpers, clocks, argument kinds, flags, statuses); handles the inputs the \
issue implies; leaves unrelated output text unchanged.
E. Agreement between independent attempts is weak evidence; the analysis above decides.

Do not prefer a candidate because it is smaller, narrower or more conservative. Prefer the one \
that fully fixes every reported scenario in the way the maintainers' tests will exercise it.

<issue>
{issue}
</issue>
{hints}
Maintainers' test names:
{tests}

{candidates}
"""


def judge_candidates(
    row: dict,
    attempts: list[dict],
    with_hints: bool = False,
    reasoning_effort: str | None = None,
    llm=None,
    votes: int = JUDGE_VOTES,
) -> dict:
    """Pick one candidate by majority over `votes` judge samples.

    One judge sample at temperature 1 flipped between candidates on the same
    inputs, so a single sample is noise. Ties go to the earliest-voted pick.
    The model is skipped entirely when there is nothing to choose.

    The judge sees the issue, hints (if enabled), test names, diffs, the
    authors' summaries and the repo-test verification. Never hidden tests or gold.
    """
    live = [a for a in attempts if a["patch"].strip()]
    if not live:
        return {"chosen": attempts[0]["attempt"], "analysis": "all candidates empty"}
    distinct = {a["patch"] for a in live}
    if len(distinct) == 1:
        return {"chosen": live[0]["attempt"], "analysis": "all non-empty candidates identical"}
    blocks = []
    for number, cand in enumerate(live, start=1):
        same = [
            str(k) for k, other in enumerate(live, start=1)
            if other is not cand and other["patch"] == cand["patch"]
        ]
        blocks.append(
            f"<candidate number={number}>\n"
            + (f"(identical diff to candidate {', '.join(same)})\n" if same else "")
            + f"<diff>\n{cand['patch']}\n</diff>\n"
            f"<author_summary>\n{cand['summary'][:3000]}\n</author_summary>\n"
            f"<repo_tests>\n{cand['verification']}\n</repo_tests>\n</candidate>"
        )
    hints = (row.get("hints_text") or "").strip()
    prompt = JUDGE_PROMPT.format(
        repo=row["repo"],
        issue=row["problem_statement"],
        hints=f"\n<hints>\n{hints}\n</hints>\n" if with_hints and hints else "",
        tests="\n".join(f"- {t}" for t in json.loads(row["FAIL_TO_PASS"])),
        candidates="\n\n".join(blocks),
    )
    if llm is None:
        llm = bench_model(reasoning_effort)
    judge = llm.with_structured_output(PatchChoice)
    ballots: list[tuple[int, str]] = []
    for _ in range(max(1, votes)):
        choice = judge.invoke(prompt)
        number = choice.best if 1 <= choice.best <= len(live) else 1
        ballots.append((live[number - 1]["attempt"], choice.analysis))
    tally = Counter(attempt for attempt, _ in ballots)
    top = max(tally.values())
    winner = next(attempt for attempt, _ in ballots if tally[attempt] == top)
    analysis = next(text for attempt, text in ballots if attempt == winner)
    votes_text = ", ".join(f"a{a}: {n}" for a, n in sorted(tally.items()))
    return {"chosen": winner, "analysis": f"[votes {votes_text}] {analysis}"}


def run_instance(
    row: dict,
    index: int,
    run_dir: Path,
    max_llm_calls: int,
    *,
    attempts: int = 1,
    with_hints: bool = False,
    reasoning_effort: str | None = None,
    review: bool = True,
) -> dict:
    """Run `attempts` independent agents, let a blind judge pick, store the pick.

    Every candidate also goes to candidates/a<k>.jsonl (its own model name),
    so grading them shows how often the judge picks a passing one.
    """
    instance_id = row["instance_id"]
    image = instance_image(instance_id)
    print(f"[{instance_id}] index={index} image={image} attempts={attempts}", flush=True)
    pull_image(image)
    test_cmd = test_command(row)
    with ThreadPoolExecutor(max_workers=attempts) as pool:
        futures = [
            pool.submit(
                run_attempt, row, index, k, run_dir, image, test_cmd, max_llm_calls,
                with_hints=with_hints, reasoning_effort=reasoning_effort, review=review,
            )
            for k in range(1, attempts + 1)
        ]
        results = sorted((f.result() for f in futures), key=lambda a: a["attempt"])
    if attempts > 1:
        try:
            verdict = judge_candidates(row, results, with_hints, reasoning_effort)
        except Exception as exc:  # a failed judge must not throw away N attempts
            live = [a for a in results if a["patch"].strip()] or results
            pick = min(live, key=lambda a: (len(a["newly_failing"]), a["attempt"]))
            verdict = {
                "chosen": pick["attempt"],
                "analysis": f"judge failed ({type(exc).__name__}: {exc}); "
                "fell back to fewest newly failing repo tests",
            }
    else:
        verdict = {"chosen": 1, "analysis": "single attempt"}
    chosen = next(a for a in results if a["attempt"] == verdict["chosen"])
    patch = chosen["patch"]
    (run_dir / "patches" / f"{instance_id}.patch").write_text(patch, encoding="utf-8")
    _append_jsonl(
        run_dir / "predictions.jsonl",
        {"instance_id": instance_id, "model_name_or_path": MODEL_NAME, "model_patch": patch},
    )
    for cand in results:
        _append_jsonl(
            run_dir / "candidates" / f"a{cand['attempt']}.jsonl",
            {
                "instance_id": instance_id,
                "model_name_or_path": f"{MODEL_NAME}-a{cand['attempt']}",
                "model_patch": cand["patch"],
            },
        )
    record = {
        "dataset": DATASET,
        "index": index,
        "instance_id": instance_id,
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "image": image,
        "test_command": test_cmd,
        "with_hints": with_hints,
        "reasoning_effort": reasoning_effort or settings.reasoning_effort,
        "review": review,
        "attempts": [{k: v for k, v in a.items() if k != "patch"} for a in results],
        "chosen_attempt": verdict["chosen"],
        "judge_analysis": verdict["analysis"],
        "llm_calls": sum(a["llm_calls"] for a in results),
        "patch_file": f"patches/{instance_id}.patch",
        "patch_bytes": len(patch.encode("utf-8")),
        "graded": False,
    }
    _append_jsonl(run_dir / "results.jsonl", record)
    print(f"[{instance_id}] chose a{verdict['chosen']} of {attempts}", flush=True)
    return record


def grade_instance(
    instance_id: str,
    run_dir: Path,
    run_id: str,
    predictions: Path | None = None,
    model_name: str = MODEL_NAME,
) -> bool | None:
    """Grade one prediction now, while its image is still local.

    Logs accumulate under logs/run_evaluation/<run_id>/, so a final
    `evals.grade ... --rewrite_reports true` with the same run_id builds the
    combined report without any images. Returns resolved, or None if no report.
    """
    with (run_dir / "grade.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable, "-m", "evals.grade",
                "--dataset_name", DATASET, "--split", "test",
                "--predictions_path", str(predictions or run_dir / "predictions.jsonl"),
                "--instance_ids", instance_id,
                "--max_workers", "1", "--cache_level", "instance",
                "--timeout", "1800", "--run_id", run_id,
            ],
            cwd=ROOT.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report = ROOT.parent / "logs" / "run_evaluation" / run_id / model_name / instance_id
    report = report / "report.json"
    if not report.exists():
        return None
    return bool(json.loads(report.read_text())[instance_id]["resolved"])


def prune_instance(instance_id: str) -> None:
    """Drop the instance image and workspace; layers shared with others stay."""
    subprocess.run(["docker", "rmi", instance_image(instance_id)], capture_output=True)
    shutil.rmtree(CACHE / "workspaces" / instance_id, ignore_errors=True)


def select_indices(args, ds) -> list[int]:
    total = len(ds)
    if args.phase is not None:
        return phase_indices(args.phase, total)
    if args.index:
        return list(args.index)
    wanted = list(args.instance_ids or [])
    if args.from_report:
        wanted += report_retry_ids(Path(args.from_report))
    by_id = {iid: i for i, iid in enumerate(ds["instance_id"])}
    missing = [iid for iid in wanted if iid not in by_id]
    if missing:
        raise SystemExit(f"not in {DATASET}: {missing}")
    return sorted({by_id[iid] for iid in wanted})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWE-bench Mini instance(s)")
    pick = parser.add_mutually_exclusive_group(required=True)
    pick.add_argument("--phase", type=int, help="Phase of 10: 1 = tasks 1-10, ... 5 = 41-50")
    pick.add_argument("--index", type=int, nargs="+", help="0-based Mini index, one or more")
    pick.add_argument("--instance-ids", nargs="+", help="SWE-bench instance ids")
    pick.add_argument("--from-report", help="swebench report JSON; re-run what it did not resolve")
    parser.add_argument("--run-dir", help="Existing evals/results/<run> to resume")
    parser.add_argument("--workers", type=int, default=1, help="Instances run in parallel")
    parser.add_argument("--max-llm-calls", type=int, default=MAX_LLM_CALLS)
    parser.add_argument(
        "--with-hints",
        action="store_true",
        help="Add the dataset's hints_text (pre-fix issue comments). Not leaderboard-comparable.",
    )
    parser.add_argument("--reasoning-effort", help="Override HARNESS_REASONING_EFFORT")
    parser.add_argument("--no-review", action="store_true", help="Skip the self-review turn")
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Independent agent runs per instance; >1 adds a blind judge (best-of-N)",
    )
    parser.add_argument(
        "--grade-run-id",
        help="Grade each instance locally right after its run, under this swebench run_id",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="After grading, remove the instance image and workspace (needs --grade-run-id)",
    )
    args = parser.parse_args()
    if args.prune and not args.grade_run_id:
        parser.error("--prune needs --grade-run-id: an image is only safe to drop once graded")

    os.chdir(ROOT.parent)
    load_dotenv()
    install_cleanup_hooks()
    ds = load_mini()
    indices = [i for i in select_indices(args, ds) if 0 <= i < len(ds)]
    if args.run_dir:
        run_dir = Path(args.run_dir)
        (run_dir / "patches").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = unique_run_dir()
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    predictions = run_dir / "predictions.jsonl"
    done = load_done_ids(predictions)
    todo = [i for i in indices if ds[i]["instance_id"] not in done]
    print(f"run dir {run_dir}")
    print(
        f"selected {len(indices)}, already stored {len(indices) - len(todo)}, to run {len(todo)}",
        flush=True,
    )

    stop = threading.Event()
    crashed: list[str] = []

    def work(index: int) -> None:
        if stop.is_set():
            return
        row = dict(ds[index])
        try:
            run_instance(
                row,
                index,
                run_dir,
                args.max_llm_calls,
                attempts=max(1, args.attempts),
                with_hints=args.with_hints,
                reasoning_effort=args.reasoning_effort,
                review=not args.no_review,
            )
        except Exception as exc:
            crashed.append(row["instance_id"])
            print(f"CRASH [{row['instance_id']}]: {type(exc).__name__}: {exc}", flush=True)
            _append_jsonl(
                run_dir / "results.jsonl",
                {
                    "dataset": DATASET,
                    "index": index,
                    "instance_id": row["instance_id"],
                    "graded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            if _credit_exhausted(exc):
                stop.set()
                print("Stopped: API credit or rate limit. Refill, then re-run with --run-dir.")
            return
        if args.grade_run_id:
            resolved = grade_instance(row["instance_id"], run_dir, args.grade_run_id)
            print(f"[{row['instance_id']}] graded resolved={resolved}", flush=True)
            if args.attempts > 1:
                per_candidate = {
                    k: grade_instance(
                        row["instance_id"],
                        run_dir,
                        args.grade_run_id,
                        predictions=run_dir / "candidates" / f"a{k}.jsonl",
                        model_name=f"{MODEL_NAME}-a{k}",
                    )
                    for k in range(1, args.attempts + 1)
                }
                print(f"[{row['instance_id']}] candidates resolved={per_candidate}", flush=True)
            if args.prune and resolved is not None:
                prune_instance(row["instance_id"])

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for future in as_completed([pool.submit(work, i) for i in todo]):
            future.result()
    wait_for_all_tracers()
    print(f"stored now: {len(load_done_ids(predictions))} in {predictions}")
    if crashed:
        print(f"crashed ({len(crashed)}): {' '.join(crashed)}")
    if stop.is_set() or crashed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
