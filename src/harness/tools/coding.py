"""
Day 13 — coding-agent tools: read / write / edit / bash.

read, write, and edit are path-jailed to `workspace`.
bash uses one long-lived Docker container per agent run (bind-mounted
workspace). Each command is docker exec, not a new container.

The default container is a bare python image. Day 17 (SWE-bench) passes the
project's own image instead, so bash can import the repo and run its tests.

Pass `run_bash=` to inject a fake runner; import stays Docker-free.
"""
from __future__ import annotations

import atexit
import os
import signal
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, tool

MAX_TOOL_CHARS = 8_000
TOOL_HEAD_CHARS = 4_000
TOOL_TAIL_CHARS = 4_000
# A read window must fit inside MAX_TOOL_CHARS with its footer, or the tool
# node's head/tail truncation would cut the middle out of it.
READ_WINDOW_CHARS = MAX_TOOL_CHARS - 300
BASH_IMAGE = "python:3.11-slim"
BASH_WORKDIR = "/workspace"
BASH_TIMEOUT_S = 180
BASH_MEM_LIMIT = "2g"


def truncate_head_tail(
    text: str,
    max_chars: int = MAX_TOOL_CHARS,
    head_chars: int = TOOL_HEAD_CHARS,
    tail_chars: int = TOOL_TAIL_CHARS,
) -> str:
    """Keep the start and end of long tool output; drop the middle."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - head_chars - tail_chars
    return (
        f"{text[:head_chars]}\n...[{omitted} chars omitted]...\n{text[-tail_chars:]}"
    )


def read_window(
    text: str,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    max_chars: int = READ_WINDOW_CHARS,
) -> str:
    """Return whole lines start_line..end_line, at most `max_chars` of them.

    Head/tail truncation hid the middle of every large file, which is where
    the code to fix usually is. A window plus a footer lets the agent page.
    """
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if total == 0:
        return text
    start = max(start_line, 1)
    if start > total:
        return f"Error: start_line {start_line} is past the end of {path} ({total} lines)"
    stop = total if end_line is None else min(max(end_line, start), total)
    shown: list[str] = []
    size = 0
    for line in lines[start - 1 : stop]:
        if shown and size + len(line) > max_chars:
            break
        shown.append(line[:max_chars])
        size += len(shown[-1])
    last = start + len(shown) - 1
    body = "".join(shown)
    if start == 1 and last == total:
        return body
    footer = f"[lines {start}-{last} of {total} in {path}"
    if last < stop:
        footer += f"; window full, continue with start_line={last + 1}"
    elif last < total:
        footer += f"; file continues at line {last + 1}"
    if not body.endswith("\n"):
        body += "\n"
    return f"{body}{footer}]"


def _resolve(workspace: Path, path: str, mount: str | None = None) -> Path | str:
    """Return a path inside `workspace`, or an error string.

    `mount` is where bash sees the workspace (e.g. /testbed). The model copies
    paths out of bash output, so `<mount>/x` is read as workspace-relative `x`.
    """
    root = workspace.resolve()
    if mount and (path == mount or path.startswith(mount.rstrip("/") + "/")):
        path = path[len(mount.rstrip("/")) :].lstrip("/") or "."
    candidate = Path(path)
    target = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return f"Error: path escapes workspace: {path}"
    return target


SANDBOX_LABEL = "harness.sandbox"
_LIVE: set[WorkspaceSandbox] = set()
_SIGNALS_INSTALLED = False


def _pid_alive(pid: int) -> bool:
    """True if `pid` is still a live process (kill -0 probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _docker_client():
    """Lazy Docker SDK client so importing this module does not need Docker."""
    import docker

    return docker.from_env()


def reap_stale_sandboxes(*, keep_id: str | None = None) -> None:
    """Remove harness containers whose owner process is dead, or leftovers."""
    try:
        client = _docker_client()
        found = client.containers.list(all=True, filters={"label": SANDBOX_LABEL})
    except Exception:
        return
    my_pid = str(os.getpid())
    # Sandboxes still open in this process (e.g. parallel SWE-bench workers)
    # are not leftovers, even though they share our pid label.
    live_ids = {sandbox.id for sandbox in _LIVE}
    for container in found:
        labels = container.labels or {}
        if keep_id and labels.get("harness.id") == keep_id:
            continue
        owner = labels.get("harness.pid")
        stale = owner is None or not owner.isdigit() or not _pid_alive(int(owner))
        leaked_here = owner == my_pid and labels.get("harness.id") not in live_ids
        if stale or leaked_here:
            try:
                container.remove(force=True)
            except Exception:
                pass


def _close_all_live() -> None:
    """Stop every sandbox this process still holds, then reap leftovers."""
    for sandbox in list(_LIVE):
        sandbox.close()
    reap_stale_sandboxes()


def install_cleanup_hooks() -> None:
    """Register atexit + SIGINT/SIGTERM so containers die when the process does.

    Signal handlers can only be set from the main thread. Call this there first
    if sandboxes will be created in worker threads.
    """
    global _SIGNALS_INSTALLED
    if _SIGNALS_INSTALLED:
        return
    _SIGNALS_INSTALLED = True
    atexit.register(_close_all_live)

    def _on_stop(signum, _frame):
        """Signal handler: close containers, then exit with 128+signum."""
        _close_all_live()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_stop)
        except (ValueError, OSError):
            pass


class WorkspaceSandbox:
    """One Docker container for the whole agent loop; exec per bash call.

    Defaults give a bare python container with the workspace at /workspace.
    For a real project, pass its image and the path its environment expects:
    `workdir` is where the workspace is mounted, `setup` runs once after the
    container starts, and `prelude` runs (under bash) before every command,
    e.g. to activate a conda env.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        image: str = BASH_IMAGE,
        workdir: str = BASH_WORKDIR,
        platform: str | None = None,
        setup: tuple[str, ...] | list[str] = (),
        prelude: str = "",
        timeout_s: int = BASH_TIMEOUT_S,
        mem_limit: str = BASH_MEM_LIMIT,
        cpus: float = 1.0,
    ):
        """Bind this sandbox to `workspace`; container starts on first `ensure`."""
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.workdir = workdir
        self.platform = platform
        self.setup = tuple(setup)
        self.prelude = prelude
        self.timeout_s = timeout_s
        self.mem_limit = mem_limit
        self.cpus = cpus
        self.id = uuid.uuid4().hex[:12]
        self._container = None
        install_cleanup_hooks()
        _LIVE.add(self)

    def ensure(self) -> None:
        """Start (or reuse) the long-lived sleep-infinity container."""
        if self._container is not None:
            try:
                self._container.reload()
                if self._container.status == "running":
                    return
            except Exception:
                self._container = None
        reap_stale_sandboxes(keep_id=self.id)
        try:
            client = _docker_client()
            self._container = client.containers.run(
                image=self.image,
                command=["sleep", "infinity"],
                name=f"harness-sandbox-{os.getpid()}-{self.id}",
                labels={
                    SANDBOX_LABEL: "1",
                    "harness.pid": str(os.getpid()),
                    "harness.id": self.id,
                },
                volumes={str(self.workspace): {"bind": self.workdir, "mode": "rw"}},
                working_dir=self.workdir,
                platform=self.platform,
                detach=True,
                network_mode="none",
                mem_limit=self.mem_limit,
                nano_cpus=int(self.cpus * 1_000_000_000),
                security_opt=["no-new-privileges:true"],
                restart_policy={"Name": "no"},
            )
            for command in self.setup:
                self._exec(command)
        except Exception:
            self._container = None
            raise

    def _exec(self, command: str):
        """docker exec one command under the timeout, prelude first if any."""
        assert self._container is not None
        if self.prelude:
            argv = ["bash", "-c", f"{self.prelude}\n{command}"]
        else:
            argv = ["sh", "-c", command]
        return self._container.exec_run(
            ["timeout", "-k", "5s", str(self.timeout_s), *argv],
            workdir=self.workdir,
            demux=False,
        )

    def run(self, command: str, max_chars: int | None = MAX_TOOL_CHARS) -> str:
        """Exec `command` in the live container; start it first if needed.

        Output is head/tail-truncated to `max_chars` for the model; pass None
        to keep all of it (e.g. to parse a full test report).
        """
        try:
            self.ensure()
            result = self._exec(command)
            output = result.output or b""
            if isinstance(output, bytes):
                logs = output.decode("utf-8", errors="replace")
            else:
                logs = str(output)
            if max_chars is not None:
                logs = truncate_head_tail(logs, max_chars=max_chars)
            code = result.exit_code
            if code == 124:
                return f"Error: bash timed out after {self.timeout_s}s\n{logs}"
            if code:
                return f"Exit {code}:\n{logs}" if logs.strip() else f"Exit {code}"
            return logs if logs.strip() else "(no output)"
        except Exception as exc:
            return f"Error: bash failed: {exc}"

    def close(self) -> None:
        """Force-remove this container and drop it from the live set."""
        container = self._container
        self._container = None
        _LIVE.discard(self)
        if container is None:
            return
        try:
            container.remove(force=True)
        except Exception:
            pass


def _docker_bash(workspace: Path, command: str) -> str:
    """Fallback for tools built without a session sandbox: one-shot exec."""
    sandbox = WorkspaceSandbox(workspace)
    try:
        return sandbox.run(command)
    finally:
        sandbox.close()


def build_file_tools(
    workspace: str | Path,
    run_bash: Callable[[str], str] | None = None,
    mount: str | None = BASH_WORKDIR,
) -> list[BaseTool]:
    """Build read / write / edit / bash jailed to `workspace`.

    `mount` is the workspace's path inside the bash container, accepted as an
    alias for the workspace root in file-tool paths.
    """
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bash_runner = run_bash or (lambda command: _docker_bash(root, command))

    @tool
    def read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a UTF-8 text file from the workspace, optionally a line range.

        Long files come back one window at a time; a footer names the lines
        shown and the start_line for the next window. Use `grep -n` via bash
        to find the lines you need.

        Args:
            path: Path relative to the workspace root.
            start_line: First line to return, 1-based. Defaults to 1.
            end_line: Last line to return, inclusive. Defaults to end of file.
        """
        target = _resolve(root, path, mount)
        if isinstance(target, str):
            return target
        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: not a file: {path}"
        return read_window(target.read_text(encoding="utf-8"), path, start_line, end_line)

    @tool
    def write(path: str, content: str) -> str:
        """Write UTF-8 text to a file in the workspace, creating parents.

        Args:
            path: Path relative to the workspace root.
            content: Full file contents to write.
        """
        target = _resolve(root, path, mount)
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
        target = _resolve(root, path, mount)
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
        """Run a shell command in the sandboxed workspace.

        Args:
            command: Shell command. cwd is the workspace root. No network.
        """
        if not command.strip():
            return "Error: empty command"
        return bash_runner(command)

    return [read, write, edit, bash]
