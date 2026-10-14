"""
Day 15/16 eval runner.

    uv run python -m evals.runner --validate
        Stock toy_repo must FAIL every task's pytest (fixtures are real).

    uv run python -m evals.runner
        Live baseline: copy toy_repo per task, run the harness, score, append
        JSONL under evals/results/. Needs OPENAI_API_KEY + Docker.

    uv run python -m evals.runner --task 001-sum-of-squares
        One task only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

from evals.tasks.catalog import TASKS, Task, task_by_id

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
TOY_REPO = ROOT / "tasks" / "toy_repo"
RESULTS = ROOT / "results"


def run_pytest(workspace: Path, paths: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def copy_toy_repo(dest: Path) -> None:
    shutil.copytree(
        TOY_REPO,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )


def check_file_contains(workspace: Path, task: Task) -> list[str]:
    missing = []
    for spec in task["file_contains"]:
        path = workspace / spec["path"]
        if not path.is_file() or spec["needle"] not in path.read_text(encoding="utf-8"):
            missing.append(f"{spec['path']} must contain {spec['needle']!r}")
    return missing


def validate() -> int:
    failed_to_fail = []
    for task in TASKS:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "repo"
            copy_toy_repo(workspace)
            passed, output = run_pytest(workspace, task["pytest"])
            if passed:
                failed_to_fail.append(task["id"])
                print(f"FAIL {task['id']}: stock repo already passes\n{output}")
            else:
                print(f"ok   {task['id']}: stock repo fails as intended")
    if failed_to_fail:
        print(f"\n{len(failed_to_fail)} fixtures are vacuously passing")
        return 1
    print(f"\n{len(TASKS)} tasks: stock toy_repo fails every check")
    return 0


def score_workspace(workspace: Path, task: Task, llm_calls: int) -> dict:
    pytest_ok, pytest_out = run_pytest(workspace, task["pytest"])
    missing = check_file_contains(workspace, task)
    over_budget = llm_calls > task["max_llm_calls"]
    passed = pytest_ok and not missing and not over_budget
    return {
        "id": task["id"],
        "tier": task["tier"],
        "pass": passed,
        "pytest_ok": pytest_ok,
        "file_contains_ok": not missing,
        "missing": missing,
        "llm_calls": llm_calls,
        "max_llm_calls": task["max_llm_calls"],
        "over_budget": over_budget,
        "pytest_tail": pytest_out[-1500:],
    }


def run_task(task: Task) -> dict:
    from harness.agent import build_loop_graph

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "repo"
        copy_toy_repo(workspace)
        graph = build_loop_graph(workspace=workspace)
        result = graph.invoke(
            {"messages": [HumanMessage(content=task["prompt"])], "llm_calls": 0},
            config={
                "run_name": f"eval-{task['id']}",
                "tags": ["day16", "custom-eval", task["tier"]],
                "metadata": {"task": task["id"], "tier": task["tier"]},
            },
        )
        llm_calls = int(result.get("llm_calls") or 0)
        row = score_workspace(workspace, task, llm_calls)
        row["plan"] = result.get("plan") or []
        return row


def append_results(rows: list[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%MZ")
    path = RESULTS / f"{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report_path = RESULTS / f"{stamp}-report.txt"
    report_path.write_text(format_report(rows, jsonl=path), encoding="utf-8")
    return path


def format_report(rows: list[dict], jsonl: Path | None = None) -> str:
    lines = ["CUSTOM EVAL REPORT", "=" * 72]
    header = f"{'id':<28} {'tier':<13} {'result':<6} pytest files calls"
    lines.append(header)
    lines.append("-" * 72)
    for row in rows:
        result = "PASS" if row["pass"] else "FAIL"
        pytest = "ok" if row["pytest_ok"] else "FAIL"
        files = "ok" if row["file_contains_ok"] else "FAIL"
        calls = f"{row['llm_calls']}/{row['max_llm_calls']}"
        lines.append(
            f"{row['id']:<28} {row['tier']:<13} {result:<6} {pytest:<6} {files:<5} {calls}"
        )
    passed = sum(1 for row in rows if row["pass"])
    lines.append("-" * 72)
    lines.append(f"Total: {passed}/{len(rows)}")
    for tier in ("easy", "medium", "hard", "adversarial"):
        group = [row for row in rows if row["tier"] == tier]
        if not group:
            continue
        ok = sum(1 for row in group if row["pass"])
        lines.append(f"  {tier:<13} {ok}/{len(group)}")
    failures = [row for row in rows if not row["pass"]]
    if failures:
        lines.append("")
        lines.append("FAILURES")
        lines.append("-" * 72)
        for row in failures:
            reasons = []
            if not row["pytest_ok"]:
                reasons.append("pytest failed")
            if row["missing"]:
                reasons.append("missing: " + "; ".join(row["missing"]))
            if row["over_budget"]:
                reasons.append(
                    f"over budget ({row['llm_calls']}/{row['max_llm_calls']})"
                )
            if row.get("error"):
                reasons.append(f"crashed: {row['error']}")
            lines.append(f"  {row['id']}")
            for reason in reasons:
                lines.append(f"    - {reason}")
            tail = (row.get("pytest_tail") or "").strip()
            if tail and not row["pytest_ok"]:
                snippet = "\n".join(tail.splitlines()[-8:])
                lines.append("    pytest:")
                for line in snippet.splitlines():
                    lines.append(f"      {line}")
            plan = row.get("plan") or []
            if plan:
                lines.append("    plan: " + " | ".join(plan))
    if jsonl is not None:
        lines.append("")
        lines.append(f"JSONL:  {jsonl}")
        lines.append(f"Report: {jsonl.with_name(jsonl.stem + '-report.txt')}")
    return "\n".join(lines) + "\n"


def crash_row(task: Task, exc: Exception) -> dict:
    return {
        "id": task["id"],
        "tier": task["tier"],
        "pass": False,
        "pytest_ok": False,
        "file_contains_ok": False,
        "missing": [],
        "llm_calls": 0,
        "max_llm_calls": task["max_llm_calls"],
        "over_budget": False,
        "pytest_tail": "",
        "plan": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_eval(task_id: str | None) -> int:
    load_dotenv()
    tasks = [task_by_id(task_id)] if task_id else TASKS
    rows = []
    for task in tasks:
        print(f"\n=== {task['id']} ({task['tier']}) ===")
        try:
            row = run_task(task)
        except Exception as exc:
            row = crash_row(task, exc)
            print(f"CRASH  {row['error']}")
        rows.append(row)
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"{status}  pytest={'ok' if row['pytest_ok'] else 'FAIL'}  "
            f"files={'ok' if row['file_contains_ok'] else 'FAIL'}  "
            f"llm_calls={row['llm_calls']}/{row['max_llm_calls']}"
        )
        if not row["pytest_ok"] and row.get("pytest_tail"):
            print(row["pytest_tail"])
        if row["missing"]:
            print("missing:", "; ".join(row["missing"]))
        if row.get("plan"):
            print("plan:")
            for step in row["plan"]:
                print(f"  - {step}")
    wait_for_all_tracers()
    path = append_results(rows)
    report = format_report(rows, jsonl=path)
    print("\n" + report)
    return 0 if all(row["pass"] for row in rows) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 15/16 custom eval runner")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--task", default=None)
    args = parser.parse_args()
    if args.validate:
        raise SystemExit(validate())
    raise SystemExit(run_eval(args.task))


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
