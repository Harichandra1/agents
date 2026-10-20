"""Day 17: offline checks for the SWE-bench driver and grader. No Docker, no API key."""
import json

from evals.grade import SPHINX_PINS, pin_sphinx_specs
from evals.swebench import (
    PatchChoice,
    build_prompt,
    failed_tests,
    instance_image,
    is_test_path,
    judge_candidates,
    report_retry_ids,
    revert_test_edits,
    split_patch,
    verification_summary,
)


def _chunk(path: str, new: bool = False) -> str:
    mode = "new file mode 100644\n" if new else ""
    return (
        f"diff --git a/{path} b/{path}\n{mode}--- a/{path}\n+++ b/{path}\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )


def test_is_test_path_matches_swebench_split():
    assert is_test_path("tests/delete/models.py")
    assert is_test_path("sphinx/testing/fixtures.py")
    assert is_test_path("pkg/test_utils.py")
    assert is_test_path("conftest.py")
    assert not is_test_path("django/db/models/deletion.py")
    assert not is_test_path("sphinx/util/inspect.py")


def test_split_patch_keeps_source_and_drops_tests_and_scratch():
    diff = "".join(
        [
            _chunk("django/utils/http.py"),
            _chunk("tests/utils_tests/test_http.py"),
            _chunk("reproduce.py", new=True),
            _chunk(".tox/py39/log/1.log", new=True),
            _chunk("setup.py"),
        ]
    )
    kept, dropped = split_patch(diff)
    assert "django/utils/http.py" in kept and "setup.py" in kept
    assert dropped == ["tests/utils_tests/test_http.py", "reproduce.py", ".tox/py39/log/1.log"]
    assert kept.count("diff --git") == 2


def test_report_retry_ids_collects_everything_not_resolved(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "resolved_ids": ["a"],
                "unresolved_ids": ["c", "b"],
                "error_ids": ["d"],
                "empty_patch_ids": [],
            }
        )
    )
    assert report_retry_ids(report) == ["b", "c", "d"]


def test_pin_sphinx_specs_is_idempotent_and_shell_safe():
    specs = {
        "sphinx-doc/sphinx": {"3.1": {"pip_packages": ["tox==4.16.0"]}, "5.0": {}},
        "django/django": {"3.0": {"pip_packages": []}},
    }
    pin_sphinx_specs(specs)
    pin_sphinx_specs(specs)
    for spec in specs["sphinx-doc/sphinx"].values():
        assert [p for p in spec["pip_packages"] if p in SPHINX_PINS] == list(SPHINX_PINS)
    assert specs["django/django"]["3.0"]["pip_packages"] == []
    assert all(pin.startswith("'") and pin.endswith("'") for pin in SPHINX_PINS)


def test_prompt_names_tests_and_the_run_command():
    row = {
        "repo": "django/django",
        "instance_id": "django__django-1",
        "problem_statement": "Dates are parsed wrong.",
        "FAIL_TO_PASS": json.dumps(["test_a (x.Tests)"]),
    }
    prompt = build_prompt(row, "./tests/runtests.py x")
    assert "Dates are parsed wrong." in prompt
    assert "- test_a (x.Tests)" in prompt
    assert "./tests/runtests.py x" in prompt


def test_instance_image_matches_swebench_naming():
    assert instance_image("django__django-11848") == (
        "swebench/sweb.eval.x86_64.django_1776_django-11848:latest"
    )
    assert instance_image("sphinx-doc__sphinx-7757") == (
        "swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-7757:latest"
    )


def test_hints_only_with_flag():
    row = {
        "repo": "django/django",
        "instance_id": "django__django-1",
        "problem_statement": "Bug.",
        "hints_text": "Use the name, not the value.",
        "FAIL_TO_PASS": json.dumps(["test_a (x.Tests)"]),
    }
    assert "Use the name" not in build_prompt(row, "cmd")
    with_hints = build_prompt(row, "cmd", with_hints=True)
    assert "<hints>\nUse the name, not the value.\n</hints>" in with_hints
    assert "<hints>" not in build_prompt({**row, "hints_text": ""}, "cmd", with_hints=True)


DJANGO_OUT = """test_a (x.T) ... ok
test_b (x.T) ... FAIL
======================================================================
FAIL: test_b (x.T)
----------------------------------------------------------------------
ERROR: test_c (x.T)
"""
PYTEST_OUT = """FAILED tests/test_a.py::test_one - AssertionError
ERROR tests/test_a.py::test_two
PASSED tests/test_a.py::test_three
"""


def test_failed_tests_parses_django_and_pytest():
    assert failed_tests(DJANGO_OUT) == {"test_b (x.T)", "test_c (x.T)"}
    assert failed_tests(PYTEST_OUT) == {"tests/test_a.py::test_one", "tests/test_a.py::test_two"}


def test_verification_summary_reports_the_delta():
    summary = verification_summary("FAIL: test_b (x.T)\n", "ERROR: test_c (x.T)\n")
    assert "newly failing: test_c (x.T)" in summary
    assert "newly passing: test_b (x.T)" in summary


def _cand(attempt, patch):
    return {"attempt": attempt, "patch": patch, "summary": "s", "verification": "v"}


ROW = {
    "repo": "django/django",
    "problem_statement": "Bug {with braces}.",
    "hints_text": "",
    "FAIL_TO_PASS": json.dumps(["test_a (x.T)"]),
}


def test_judge_skips_the_model_when_there_is_no_choice():
    class Boom:
        def with_structured_output(self, *_a, **_k):
            raise AssertionError("judge model should not be called")

    same = [_cand(1, "diff A"), _cand(2, "diff A"), _cand(3, "")]
    assert judge_candidates(ROW, same, llm=Boom())["chosen"] == 1
    empty = [_cand(1, ""), _cand(2, " ")]
    assert judge_candidates(ROW, empty, llm=Boom())["chosen"] == 1


def test_judge_maps_its_pick_back_to_the_attempt_number(scripted):
    llm = scripted([PatchChoice(analysis="2 matches the issue", best=2)])
    cands = [_cand(1, ""), _cand(2, "diff A"), _cand(3, "diff B")]
    verdict = judge_candidates(ROW, cands, llm=llm, votes=1)
    assert verdict == {"chosen": 3, "analysis": "[votes a3: 1] 2 matches the issue"}


def test_judge_takes_the_majority_vote(scripted):
    llm = scripted(
        [
            PatchChoice(analysis="first", best=1),
            PatchChoice(analysis="second", best=2),
            PatchChoice(analysis="third", best=2),
        ]
    )
    cands = [_cand(1, "diff A"), _cand(2, "diff B")]
    verdict = judge_candidates(ROW, cands, llm=llm, votes=3)
    assert verdict["chosen"] == 2
    assert verdict["analysis"] == "[votes a1: 1, a2: 2] second"


def test_revert_test_edits_restores_tracked_and_removes_new(tmp_path):
    import subprocess

    repo = tmp_path / "r"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_x.py").write_text("old\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "tests" / "test_x.py").write_text("edited\n")
    (repo / "tests" / "test_new.py").write_text("new\n")
    revert_test_edits(repo, ["tests/test_x.py", "tests/test_new.py"])
    assert (repo / "tests" / "test_x.py").read_text() == "old\n"
    assert not (repo / "tests" / "test_new.py").exists()
