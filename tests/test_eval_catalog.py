"""Day 15: every custom-eval task must fail on the stock toy repo."""
from evals.runner import TASKS, validate
from evals.tasks.catalog import TASKS as CATALOG


def test_catalog_ids_are_unique():
    ids = [task["id"] for task in CATALOG]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 15


def test_stock_toy_repo_fails_every_task():
    assert validate() == 0
    assert len(TASKS) == len(CATALOG)
