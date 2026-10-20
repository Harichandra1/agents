"""Offline tests for Day 13 coding tools. No Docker, no API key."""
import os

from harness.tools.coding import _pid_alive, build_file_tools, read_window, truncate_head_tail


def _tools(tmp_path, run_bash=None):
    return {t.name: t for t in build_file_tools(tmp_path, run_bash=run_bash)}


def test_write_then_read_roundtrip(tmp_path):
    tools = _tools(tmp_path)
    written = tools["write"].invoke({"path": "src/hello.py", "content": "print(1)\n"})
    assert "Wrote" in written
    assert (tmp_path / "src" / "hello.py").read_text() == "print(1)\n"
    assert tools["read"].invoke({"path": "src/hello.py"}) == "print(1)\n"


def test_read_missing_file(tmp_path):
    tools = _tools(tmp_path)
    assert "not found" in tools["read"].invoke({"path": "missing.py"}).lower()


def test_path_jail_rejects_escape(tmp_path):
    tools = _tools(tmp_path)
    outside = str(tmp_path.parent / "secret.txt")
    assert "escapes workspace" in tools["write"].invoke(
        {"path": "../secret.txt", "content": "nope"}
    )
    assert "escapes workspace" in tools["read"].invoke({"path": outside})
    assert "escapes workspace" in tools["edit"].invoke(
        {"path": "../secret.txt", "old_string": "a", "new_string": "b"}
    )
    assert not (tmp_path.parent / "secret.txt").exists()


def test_edit_replaces_unique_string(tmp_path):
    tools = _tools(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\ny = 2\n")
    result = tools["edit"].invoke(
        {"path": "app.py", "old_string": "x = 1", "new_string": "x = 3"}
    )
    assert result == "Edited app.py"
    assert (tmp_path / "app.py").read_text() == "x = 3\ny = 2\n"


def test_edit_rejects_ambiguous_match(tmp_path):
    tools = _tools(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\nx = 1\n")
    result = tools["edit"].invoke(
        {"path": "app.py", "old_string": "x = 1", "new_string": "x = 2"}
    )
    assert "2 times" in result
    assert (tmp_path / "app.py").read_text() == "x = 1\nx = 1\n"


def test_bash_uses_injected_runner(tmp_path):
    seen = {}

    def run_bash(command: str) -> str:
        seen["command"] = command
        return "ok"

    tools = _tools(tmp_path, run_bash=run_bash)
    assert tools["bash"].invoke({"command": "python src/hello.py"}) == "ok"
    assert seen["command"] == "python src/hello.py"


def test_truncate_head_tail_keeps_ends():
    text = "H" * 40 + "M" * 200 + "T" * 40
    out = truncate_head_tail(text, max_chars=50, head_chars=10, tail_chars=10)
    assert out.startswith("H" * 10)
    assert out.endswith("T" * 10)
    assert "omitted" in out
    assert "M" * 20 not in out


def test_pid_alive_detects_this_process_and_missing():
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(999_999_999) is False


def test_bash_rejects_empty_command(tmp_path):
    tools = _tools(tmp_path, run_bash=lambda _cmd: "should not run")
    assert "empty" in tools["bash"].invoke({"command": "   "}).lower()


def test_read_pages_long_files_instead_of_cutting_the_middle(tmp_path):
    tools = _tools(tmp_path)
    body = "".join(f"line {i:04d} " + "x" * 60 + "\n" for i in range(1, 501))
    (tmp_path / "big.py").write_text(body)
    first = tools["read"].invoke({"path": "big.py"})
    assert first.startswith("line 0001")
    assert "omitted" not in first
    assert "of 500 in big.py; window full, continue with start_line=" in first
    middle = tools["read"].invoke({"path": "big.py", "start_line": 250, "end_line": 252})
    assert middle.splitlines()[:3] == [body.splitlines()[i] for i in (249, 250, 251)]
    assert middle.endswith("[lines 250-252 of 500 in big.py; file continues at line 253]")


def test_read_window_rejects_start_past_end():
    assert "past the end" in read_window("a\nb\n", "f.py", start_line=9)


def test_file_tools_accept_the_container_mount_path(tmp_path):
    tools = {t.name: t for t in build_file_tools(tmp_path, mount="/testbed")}
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n")
    assert tools["read"].invoke({"path": "/testbed/pkg/mod.py"}) == "x = 1\n"
    tools["edit"].invoke(
        {"path": "/testbed/pkg/mod.py", "old_string": "x = 1", "new_string": "x = 2"}
    )
    assert (tmp_path / "pkg" / "mod.py").read_text() == "x = 2\n"
    assert "escapes workspace" in tools["read"].invoke({"path": "/testbed/../etc/passwd"})


class _FakeContainer:
    def __init__(self, labels):
        self.labels = labels
        self.removed = False
        self.calls = []

    def remove(self, force=False):
        self.removed = True

    def exec_run(self, argv, workdir=None, demux=False):
        self.calls.append((argv, workdir))
        return type("Result", (), {"output": b"ok\n", "exit_code": 0})()


def test_reaper_keeps_other_live_sandboxes_in_this_process(tmp_path, monkeypatch):
    from harness.tools import coding

    mine = coding.WorkspaceSandbox(tmp_path / "a")
    sibling = coding.WorkspaceSandbox(tmp_path / "b")
    pid = str(os.getpid())
    live = _FakeContainer({"harness.pid": pid, "harness.id": sibling.id})
    leaked = _FakeContainer({"harness.pid": pid, "harness.id": "gone"})

    class Client:
        containers = type("C", (), {"list": staticmethod(lambda **_: [live, leaked])})

    monkeypatch.setattr(coding, "_docker_client", lambda: Client)
    try:
        coding.reap_stale_sandboxes(keep_id=mine.id)
        assert leaked.removed and not live.removed
    finally:
        mine.close()
        sibling.close()


def test_sandbox_prelude_runs_under_bash_in_workdir(tmp_path):
    from harness.tools.coding import WorkspaceSandbox

    sandbox = WorkspaceSandbox(
        tmp_path, workdir="/testbed", prelude="conda activate testbed", timeout_s=7
    )
    fake = _FakeContainer({})
    sandbox._container = fake
    try:
        assert sandbox._exec("pytest -q").output == b"ok\n"
    finally:
        sandbox._container = None
        sandbox.close()
    argv, workdir = fake.calls[0]
    assert workdir == "/testbed"
    assert argv[:4] == ["timeout", "-k", "5s", "7"]
    assert argv[4:] == ["bash", "-c", "conda activate testbed\npytest -q"]
