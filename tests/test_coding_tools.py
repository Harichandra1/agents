"""Offline tests for Day 13 coding tools. No Docker, no API key."""
from harness.tools.coding import build_file_tools


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


def test_bash_rejects_empty_command(tmp_path):
    tools = _tools(tmp_path, run_bash=lambda _cmd: "should not run")
    assert "empty" in tools["bash"].invoke({"command": "   "}).lower()
