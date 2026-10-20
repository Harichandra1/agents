"""Point SWE-bench 4.1.0 at Modal's current sandbox filesystem API.

Modal's servers reject Sandbox.open(). swebench 4.1.0 still calls it, including
a cgroup-v1 write that does not exist on current sandboxes. Re-run this after
`uv sync` if grading starts failing with "legacy Sandbox filesystem API".
"""
from __future__ import annotations

from pathlib import Path

import swebench.harness.modal_eval.run_evaluation_modal as modal_eval

OLD_INIT = '''        # Hack for pylint
        self.write_file("/sys/fs/cgroup/cpu/cpu.shares", "2048")
'''

NEW_INIT = '''        # Hack for pylint. cgroup v1 only; current Modal sandboxes are v2.
        try:
            self.write_file("/sys/fs/cgroup/cpu/cpu.shares", "2048")
        except Exception:
            pass
'''

OLD_WRITE = '''    def write_file(self, file_path: str, content: str):
        self.sandbox.open(file_path, "w").write(content)
'''

NEW_WRITE = '''    def write_file(self, file_path: str, content: str):
        self.sandbox.filesystem.write_text(content, file_path)
'''


def main() -> None:
    path = Path(modal_eval.__file__)
    text = path.read_text(encoding="utf-8")
    if "filesystem.write_text" in text and "cgroup v1 only" in text:
        print(f"already patched {path}")
        return
    if OLD_INIT not in text or OLD_WRITE not in text:
        raise SystemExit(f"unexpected swebench modal eval source: {path}")
    updated = text.replace(OLD_INIT, NEW_INIT).replace(OLD_WRITE, NEW_WRITE)
    path.write_text(updated, encoding="utf-8")
    print(f"patched {path}")


if __name__ == "__main__":
    main()
