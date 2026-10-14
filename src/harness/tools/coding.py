"""
Day 13 — coding-agent tools: read / write / edit / bash.

read, write, and edit are path-jailed to `workspace`.
bash runs in Docker with that same directory mounted at /workspace, so
agent-started processes can see and change workspace files.

Pass `run_bash=` to inject a fake runner; import stays Docker-free.
"""
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, tool

MAX_READ_CHARS = 100_000
MAX_BASH_CHARS = 50_000
BASH_IMAGE = "python:3.11-slim"
BASH_TIMEOUT_S = 30


def _resolve(workspace: Path, path: str) -> Path | str:
    """Return a path inside `workspace`, or an error string."""
    root = workspace.resolve()
    candidate = Path(path)
    target = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return f"Error: path escapes workspace: {path}"
    return target


def _docker_bash(workspace: Path, command: str) -> str:
    import docker

    container = None
    try:
        client = docker.from_env()
        container = client.containers.run(
            image=BASH_IMAGE,
            command=["sh", "-c", command],
            volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            detach=True,
            network_mode="none",
            mem_limit="256m",
            nano_cpus=1_000_000_000,
            security_opt=["no-new-privileges:true"],
        )
        result = container.wait(timeout=BASH_TIMEOUT_S)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
        if len(logs) > MAX_BASH_CHARS:
            logs = logs[:MAX_BASH_CHARS] + "\n...[truncated]"
        code = result.get("StatusCode", 1)
        if code != 0:
            return f"Exit {code}:\n{logs}" if logs.strip() else f"Exit {code}"
        return logs if logs.strip() else "(no output)"
    except Exception as exc:
        return f"Error: bash failed: {exc}"
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass


def build_file_tools(
    workspace: str | Path,
    run_bash: Callable[[str], str] | None = None,
) -> list[BaseTool]:
    """Build read / write / edit / bash jailed to `workspace`."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bash_runner = run_bash or (lambda command: _docker_bash(root, command))

    @tool
    def read(path: str) -> str:
        """Read a UTF-8 text file from the workspace.

        Args:
            path: Path relative to the workspace root.
        """
        target = _resolve(root, path)
        if isinstance(target, str):
            return target
        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: not a file: {path}"
        text = target.read_text(encoding="utf-8")
        if len(text) > MAX_READ_CHARS:
            return (
                f"Error: file too large ({len(text)} chars). "
                f"Max is {MAX_READ_CHARS}."
            )
        return text

    @tool
    def write(path: str, content: str) -> str:
        """Write UTF-8 text to a file in the workspace, creating parents.

        Args:
            path: Path relative to the workspace root.
            content: Full file contents to write.
        """
        target = _resolve(root, path)
        if isinstance(target, str):
            return target
        if target.exists() and target.is_dir():
            return f"Error: path is a directory: {path}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    @tool
    def edit(path: str, old_string: str, new_string: str) -> str:
        """Replace exactly one occurrence of old_string in a workspace file.

        Args:
            path: Path relative to the workspace root.
            old_string: Exact text to find. Must match once.
            new_string: Replacement text.
        """
        if not old_string:
            return "Error: old_string must not be empty"
        target = _resolve(root, path)
        if isinstance(target, str):
            return target
        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: not a file: {path}"
        text = target.read_text(encoding="utf-8")
        matches = text.count(old_string)
        if matches == 0:
            return f"Error: old_string not found in {path}"
        if matches > 1:
            return (
                f"Error: old_string found {matches} times in {path}. "
                "Include more surrounding lines so it matches once."
            )
        target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {path}"

    @tool
    def bash(command: str) -> str:
        """Run a shell command in the sandboxed workspace (/workspace).

        Args:
            command: Shell command. cwd is the workspace root. No network.
        """
        if not command.strip():
            return "Error: empty command"
        return bash_runner(command)

    return [read, write, edit, bash]
