"""Day 15: every custom-eval task must fail on the stock toy repo."""
from evals.runner import (
    TASKS,
    capture_patch,
    ensure_git_base,
    unique_run_dir,
    validate,
)
from evals.tasks.catalog import TASKS as CATALOG


def test_catalog_ids_are_unique():
    ids = [task["id"] for task in CATALOG]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 15


def test_stock_toy_repo_fails_every_task():
    assert validate() == 0
    assert len(TASKS) == len(CATALOG)


def test_git_diff_saved_as_patch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("old\n")
    ensure_git_base(repo)
    (repo / "a.txt").write_text("new\n")
    (repo / "b.txt").write_text("added\n")
    patch = capture_patch(repo)
    assert "a.txt" in patch
    assert "b.txt" in patch
    assert "new" in patch


def test_unique_run_dir_does_not_overwrite(tmp_path, monkeypatch):
    from evals import runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS", tmp_path)
    first = unique_run_dir()
    second = unique_run_dir()
    assert first != second
    assert first.exists() and second.exists()
