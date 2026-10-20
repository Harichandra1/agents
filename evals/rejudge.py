"""
Day 17 — re-run only the best-of-N judge on a finished run's saved candidates.

    uv run python -m evals.rejudge evals/results/<run> --grade-run-id <id> --tag v2

Agents are not re-run. Each candidate was already graded under
logs/run_evaluation/<grade-run-id>/<model>-a<k>/, so the score of a new pick is
known without images: a judge change can be measured in minutes, not hours.
Writes predictions.rejudge-<tag>.jsonl and rejudge-<tag>.jsonl next to the
run; never touches the run's original predictions or results.

Iterating a judge against graded candidates is tuning on that set: confirm a
change on instances it was not tuned on before trusting the number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from evals.swebench import MODEL_NAME, ROOT, judge_candidates, load_mini


def candidate_resolved(grade_run_id: str, attempt: int, instance_id: str) -> bool | None:
    report = (
        ROOT.parent / "logs" / "run_evaluation" / grade_run_id
        / f"{MODEL_NAME}-a{attempt}" / instance_id / "report.json"
    )
    if not report.exists():
        return None
    return bool(json.loads(report.read_text())[instance_id]["resolved"])


def load_attempts(run_dir: Path, record: dict) -> list[dict]:
    iid = record["instance_id"]
    out = []
    for attempt in record["attempts"]:
        patch_path = run_dir / "patches" / f"{iid}.a{attempt['attempt']}.patch"
        out.append({**attempt, "patch": patch_path.read_text(encoding="utf-8")})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge saved best-of-N candidates")
    parser.add_argument("run_dir")
    parser.add_argument("--grade-run-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()
    load_dotenv(ROOT.parent / ".env")
    run_dir = Path(args.run_dir)
    rows = {r["instance_id"]: dict(r) for r in load_mini()}
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and "attempts" in line
    ]
    picked = passed = oracle = 0
    for record in records:
        iid = record["instance_id"]
        attempts = load_attempts(run_dir, record)
        verdict = judge_candidates(
            rows[iid], attempts, record.get("with_hints", False), args.reasoning_effort
        )
        grades = {a["attempt"]: candidate_resolved(args.grade_run_id, a["attempt"], iid)
                  for a in attempts}
        chosen = next(a for a in attempts if a["attempt"] == verdict["chosen"])
        ok = bool(grades.get(verdict["chosen"]))
        picked += 1
        passed += ok
        oracle += any(grades.values())
        print(f"{iid}: chose a{verdict['chosen']} (was a{record['chosen_attempt']}) "
              f"resolved={ok} candidates={grades}", flush=True)
        with (run_dir / f"predictions.rejudge-{args.tag}.jsonl").open("a") as fh:
            fh.write(json.dumps({"instance_id": iid, "model_name_or_path": MODEL_NAME,
                                 "model_patch": chosen["patch"]}) + "\n")
        with (run_dir / f"rejudge-{args.tag}.jsonl").open("a") as fh:
            fh.write(json.dumps({"instance_id": iid, "chosen_attempt": verdict["chosen"],
                                 "resolved": ok, "candidates": grades,
                                 "judge_analysis": verdict["analysis"]}) + "\n")
    print(f"judge picked a passing candidate {passed}/{picked}; pass@N {oracle}/{picked}")


if __name__ == "__main__":
    main()
