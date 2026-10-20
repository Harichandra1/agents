"""
Day 17 — grade predictions with the SWE-bench harness, after pinning test deps
that have drifted since SWE-bench Verified was validated.

    uv run python -m evals.grade \\
      --dataset_name MariusHobbhahn/swe-bench-verified-mini --split test \\
      --predictions_path evals/results/<run>/predictions.jsonl \\
      --modal true --timeout 1800 --run_id <id> --report_dir evals/results/<run>

Same CLI as `python -m swebench.harness.run_evaluation`. Pass
`--predictions_path gold` to check that the environment itself is sound.

Why: swebench 4.1.0 leaves Sphinx's `docutils` and `Pygments` unpinned, so a
fresh image gets today's releases. docutils 0.22 dropped `docutils.utils.roman`
(Sphinx 3.x then fails at import, every test errors), and Pygments 2.19 emits
whitespace tokens that old Sphinx test expectations do not contain. Both broke
gold patches, not just ours. Test specs are built in this process before being
shipped to Modal, so editing the in-memory spec table is enough.
"""
from __future__ import annotations

import runpy

SPHINX_REPO = "sphinx-doc/sphinx"
# Last releases before each break; roughly what Verified was validated against.
# Quoted: swebench joins pip_packages into one unquoted shell line, where a bare
# `<` would be an input redirect.
SPHINX_PINS = ("'docutils<0.22'", "'Pygments<2.19'")


def pin_sphinx_specs(specs: dict | None = None) -> dict:
    """Append SPHINX_PINS to every Sphinx version's pip_packages, once."""
    if specs is None:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

        specs = MAP_REPO_VERSION_TO_SPECS
    for spec in specs[SPHINX_REPO].values():
        packages = spec.setdefault("pip_packages", [])
        for pin in SPHINX_PINS:
            if pin not in packages:
                packages.append(pin)
    return specs


def main() -> None:
    pin_sphinx_specs()
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
