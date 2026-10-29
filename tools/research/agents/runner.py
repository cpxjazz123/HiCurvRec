"""Real OpenCode CLI with cooperative cancellation and process receipts."""
from __future__ import annotations

import json
import re
import os
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from .. import settings
from ..core.evidence import atomic_json, digest_bytes, digest_file
from . import policy, progress_server


#: The roles that may not write anything. Their answer arrives through the
#: session stream instead of a file they wrote.
READ_ONLY_ROLES = ("verify", "repair", "ideate", "score")

#: The only filesystem boundary that is a guarantee rather than a request: a
#: user mount namespace in which the paths outside the allowlist do not exist.
#: opencode's own ``permission`` object is still written as a second layer, but
#: a tool that ignores it cannot escape the mount namespace.
SANDBOX_BINARY = "bwrap"

#: The interpreter that runs inside the namespace. It must exist in the
#: namespace, so it is the system one: the namespace carries /usr and /bin and
#: nothing from the research environment.
SANDBOX_PYTHON = "/usr/bin/python3"

#: The append-only progress channel. The server is a framework runtime
#: dependency: it is mounted read-only, it is the only writer of the agent's
#: progress file, and it is the agent's only write. The path is resolved,
#: because the repository is reachable through a symlinked home and a bind
#: destination under that link would create the link as a directory and break
#: the namespace.
PROGRESS_SERVER = Path(__file__).resolve().with_name("progress_server.py")
PROGRESS_FILE_NAME = "progress.jsonl"

#: Tables that can carry another session's text, another project's paths or a
#: snapshot of files the agent must not see. Only credentials, the model catalog
#: and the migration ledger survive into an agent's private opencode database.
_PRIVATE_TABLES_TO_CLEAR = (
    "event", "message", "part", "session", "session_v2", "session_message",
    "session_inbox", "session_pending", "session_share", "todo",
    "instruction_blob", "instruction_entry", "instruction_state",
    "project", "project_directory", "worktree", "workspace",
    "account", "account_state", "control_account",
)


def _read_patterns(path: Path) -> list:
    """The permission patterns that name this path, under both spellings.

    The repository is reachable through a symlinked home, and an agent may be
    given either spelling. Binding only the resolved one is enough for the mount
    namespace, but opencode matches the permission rule against the text of the
    path it was asked to read.
    """
    raw = Path(path)
    resolved = raw.resolve()
    suffix = "" if resolved.is_file() else "/**"
    patterns = [str(resolved) + suffix]
    if raw.is_absolute() and str(raw) != str(resolved):
        patterns.append(str(raw) + suffix)
    return list(dict.fromkeys(patterns))


def effective_read_paths(read_paths) -> list:
    """The read set an agent really runs with: the declared one plus the contract.

    The executor and the caller must agree on it: the fingerprint is computed
    from the allowlist the agent actually gets, so a cached answer is reused only
    when the same sources and the same contract were in front of the agent.
    """
    contract = _geometry_contract()
    paths = list(read_paths or [])
    if contract["path"] not in paths:
        paths.append(contract["path"])
    return paths


def isolation_permissions(workspace: Path, role: str = "verify",
                          read_paths=None, allow_shell: bool = False,
                          allow_write: bool = False) -> dict:
    """The opencode permission object that confines an agent to its workspace.

    opencode v2 reads a single ``permission`` object whose keys are the tool
    names and whose values are ``ask``/``allow``/``deny``, or a rule object whose
    keys are path patterns and whose last match wins. An earlier version of this
    function wrote a ``permissions`` array of action/resource/effect rules;
    opencode ignores that shape, so every agent kept a full shell and the
    boundary was never in force.

    ``read_paths`` is the allowlist of directories the role may read outside its
    own workspace, and the policy module has already refused anything wider.
    ``read`` and ``external_directory`` are deny-by-default and name exactly
    those directories plus the workspace.

    A role that may not write gets neither a shell nor an edit tool, so it
    cannot change the candidate tree or anything else; its answer is taken from
    the session stream, where the agent printed it. A role that does get a shell
    is bounded by the mount namespace instead: it can run code, but only over
    the files the policy allowed.
    """
    workspace = Path(workspace)
    runs_code = str(role) == "implementation"
    read_only = str(role) in READ_ONLY_ROLES
    permissions = {
        "bash": "allow" if (runs_code or allow_shell) else "deny",
        "edit": "allow" if (allow_write or not read_only) else "deny",
        "write": "allow" if (allow_write or not read_only) else "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "task": "deny",
        "lsp": "deny",
    }
    if read_paths is None:
        permissions["external_directory"] = "deny"
        return permissions
    allowed = []
    external = {}
    for pattern in _read_patterns(workspace):
        if pattern not in allowed:
            allowed.append(pattern)
    for entry in read_paths:
        path = Path(entry)
        for pattern in _read_patterns(path):
            if pattern not in allowed:
                allowed.append(pattern)
            external[pattern] = "allow"
        # opencode decides whether a path is outside the project at the DIRECTORY
        # level: naming only the file left every read of the candidate tree denied
        # while the exact file still matched nothing. The containing directory is
        # therefore declared too; which files inside it may be opened is still
        # bounded by `read` above.
        parent = path if path.is_dir() else path.parent
        for pattern in _read_patterns(parent):
            external.setdefault(pattern, "allow")
    permissions["read"] = {"*": "deny", **{pattern: "allow" for pattern in allowed}}
    permissions["external_directory"] = {"*": "deny", **external}
    return permissions


def session_reads(path: Path) -> dict:
    """What the agent actually read, and what it was refused.

    A report that claims to have reviewed code it could not open is not a source
    review. The session stream knows: every read tool call carries its status and
    its path, so the denied reads are a fact the framework can record instead of
    a thing a reader has to notice.
    """
    read: list = []
    denied: list = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"files_read": read, "reads_denied": denied}
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("type") != "tool_use":
            continue
        part = entry.get("part") or {}
        tool = str(part.get("tool") or "")
        state = part.get("state") or {}
        payload = state.get("input") or {}
        target = str(payload.get("path") or payload.get("pattern") or "")
        status = str(state.get("status") or "")
        error = str(state.get("error") or "")
        if tool == "execute":
            # an agent that runs a snippet reads through it: the paths are in the
            # code, and the permission outcome is in the error
            code = str(payload.get("code") or "")
            if "read(" not in code and "glob(" not in code:
                continue
            for match in re.finditer(r"(?:read|glob)\(\{?\s*path:\s*[\"']([^\"']+)",
                                     code):
                target = match.group(1)
                break
            if not target:
                continue
            if status == "error" and "ermission" in error:
                denied.append({"tool": "execute", "target": target,
                               "error": error[:120]})
            else:
                read.append(target)
            continue
        if tool not in ("read", "glob", "grep", "edit", "write"):
            continue
        if status == "error" and "ermission" in error:
            denied.append({"tool": tool, "target": target, "error": error[:120]})
        elif status == "completed" and target:
            read.append(target)
    return {"files_read": read, "reads_denied": denied}


def session_answer(path: Path) -> str:
    """The text an agent printed, read back from the opencode session stream.

    A read-only role cannot create its answer file, so the framework takes the
    answer from the session and writes the file itself.
    """
    chunks: list[str] = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") != "text":
            continue
        part = event.get("part") or {}
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n".join(chunks).strip()


def write_isolation_config(workspace: Path, role: str = "verify",
                           read_paths=None, allow_shell: bool = False,
                           allow_write: bool = False,
                           mcp_servers: dict | None = None) -> Path:
    """Write the agent's private permission boundary next to its session."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "opencode.json"
    config = {"$schema": "https://opencode.ai/config.json",
              "permission": isolation_permissions(workspace, role, read_paths,
                                                 allow_shell, allow_write)}
    if mcp_servers:
        config["mcp"] = {"servers": mcp_servers}
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def mark_workspace_project(workspace: Path) -> Path:
    """Make the workspace opencode's OWN project root.

    opencode resolves its project by walking up to the enclosing git root, so a
    workspace inside the repository makes the REPOSITORY the project: the task's
    own opencode.json is then ignored, the permission boundary never applies and
    the Agent writes into the baseline trees. An empty .git/ directory stops the
    walk at the workspace, so the boundary in opencode.json is the one that
    applies and the shell stays denied. The marker is empty on purpose: it is a
    project marker, not a repository.
    """
    marker = Path(workspace) / ".git"
    marker.mkdir(parents=True, exist_ok=True)
    return marker


def _private_provider_db(source: Path, target: Path) -> None:
    """Give the sandboxed agent its own opencode database.

    The shared database holds every session's prompt, output and file snapshot;
    mounting it would hand the reviewer the earlier reviews. Only the rows that
    make the provider route work (credentials, model catalog, migration ledger)
    are copied into a fresh database.
    """
    if not source.is_file():
        return
    reader = sqlite3.connect("file:" + str(source) + "?mode=ro", uri=True)
    writer = sqlite3.connect(str(target))
    try:
        reader.backup(writer)
        for table in _PRIVATE_TABLES_TO_CLEAR:
            try:
                writer.execute('delete from "' + table + '"')
            except sqlite3.OperationalError:
                continue
        writer.commit()
        writer.execute("vacuum")
        writer.commit()
    finally:
        writer.close()
        reader.close()


def agent_sandbox_home(workspace: Path) -> Path:
    """A private HOME for one sandboxed agent, with no shared session data.

    The directory is created outside the masked home: the namespace resolves a
    bind source against its own root, and a path under the home would either be
    invisible or would recreate the links the namespace hides.
    """
    base = Path(tempfile.gettempdir()).resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = Path("/tmp")
    home = Path(tempfile.mkdtemp(prefix="genrec-agent-", dir=str(base)))
    data = home / ".local" / "share" / "opencode"
    cache = home / ".cache" / "opencode"
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    shared = Path.home() / ".local" / "share" / "opencode"
    auth = shared / "auth.json"
    if auth.is_file():
        shutil.copy2(auth, data / "auth.json")
    _private_provider_db(shared / "opencode.db", data / "opencode.db")
    return home


def _symlink_components(path: Path) -> list:
    """The symlinks on the way to ``path``, as (link, target) pairs.

    The repository is reached through a symlinked home. The sandbox masks the
    home directory, so those links have to be recreated inside it or the path
    the agent was given simply does not exist there.
    """
    parts = Path(path).parts
    if not parts:
        return []
    current = Path(parts[0])
    found = []
    for part in parts[1:]:
        current = current / part
        if current.is_symlink():
            found.append((str(current), str(current.resolve())))
            current = Path(current.resolve())
    return found


def sandbox_command(command: list, workspace: Path, read_paths, binary: Path,
                    private_home: Path) -> list:
    """Wrap one CLI invocation in a mount namespace with an explicit allowlist.

    The sandbox is the guarantee: the repository, the other arms' results and the
    earlier agent sessions are not mounted at all, so no tool, prompt or model
    mistake can read them. ``opencode``'s own permission object is still written
    as a second layer, but the mount namespace is what holds if that layer is
    bypassed (as it was while ``PWD`` still pointed at the repository).

    The network is shared because the model provider needs it; nothing else
    outside the allowlist is reachable.
    """
    home = Path.home()
    binary = Path(binary).resolve()
    private_home = Path(private_home)
    args = [
        SANDBOX_BINARY, "--unshare-all", "--share-net", "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind-try", "/lib", "/lib",
        "--ro-bind-try", "/lib64", "/lib64",
        "--ro-bind-try", "/sbin", "/sbin",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        # Mask the home directory first, then expose only what the CLI needs.
        # The global opencode configuration is NOT exposed: it can declare MCP
        # servers and plugins, and a server that reads a file for the agent
        # would read it from the host, where the mount namespace cannot stop it.
        # The agent runs with the workspace's own opencode.json and nothing else.
        "--tmpfs", str(home),
        "--ro-bind", str(binary.parent), str(binary.parent),
        # A private database and cache: the shared session history is not visible.
        "--bind", str(private_home / ".local" / "share" / "opencode"),
        str(home / ".local" / "share" / "opencode"),
        "--bind", str(private_home / ".cache" / "opencode"),
        str(home / ".cache" / "opencode"),
        # The progress server: a framework runtime dependency, read-only.
        "--ro-bind", str(PROGRESS_SERVER), str(PROGRESS_SERVER),
    ]
    entries = [Path(workspace), *[Path(item) for item in (read_paths or [])]]
    workspace_raw = str(Path(workspace))
    # Recreate the links that the masked home took away, so the path the agent
    # was given resolves inside the namespace. They come first: a link and a
    # bind cannot occupy the same path.
    for link, target in dict.fromkeys(
            pair for entry in entries for pair in _symlink_components(entry)):
        args += ["--symlink", target, link]
    for entry in entries:
        resolved = entry.resolve()
        mode = "--bind" if entry == Path(workspace) else "--ro-bind"
        args += [mode, str(resolved), str(resolved)]
    args += ["--chdir", workspace_raw]
    return args + list(command)


#: How long a role may take. The implementer writes a mechanism and runs its
#: smoke test, which does not fit the review budget; the reviewers only read.
#: The reviewer budget follows the model: DeepSeek V4.1 Flash needs longer on
#: the long review prompts than the previous model did, and a reviewer that is
#: cancelled for time leaves the round without a verdict.
ROLE_TIMEOUT_SECONDS = {
    "implementation": 3600,
    "repair": 2400,
    "verify": 2400,
    "ideate": 2400,
    "scoring": 2400,
}


class AgentFailure(RuntimeError):
    """An agent run that produced no usable answer, with the class of failure.

    ``auth`` means the provider refused the credentials: retrying the same call
    cannot help and the caller pauses. ``timeout``, ``cancel``, ``crash`` and
    ``empty`` are execution faults of one agent and are retried in place.
    """

    def __init__(self, failure_class: str, receipt: str, message: str):
        super().__init__(message)
        self.failure_class = failure_class
        self.receipt = receipt


def class_of_failure(session: Path, *, timed_out: bool, cancelled: bool,
                     exit_code: int, wrote_answer: bool) -> str:
    """Why an agent run produced no usable answer."""
    text = ""
    if session.is_file():
        try:
            text = session.read_text(encoding="utf-8", errors="replace")[-200000:]
        except OSError:
            text = ""
    if "provider.auth" in text or "authentication token has expired" in text:
        return "auth"
    if "provider.quota" in text or "rate limit" in text.lower():
        return "quota"
    if timed_out:
        return "timeout"
    if cancelled:
        return "cancel"
    if exit_code != 0:
        return "crash"
    return "empty" if not wrote_answer else "unusable"


def _agent_environment(workspace: Path) -> dict:
    """The environment of one agent process.

    A sandboxed agent gets only the variables the CLI needs, not the parent's
    provider keys for every other service: a read of /proc/self/environ must not
    hand it credentials it has no business holding. It also gets ``PWD`` set to
    its workspace, because opencode resolves its project from ``PWD``: a child
    that inherits the server's directory reads the REPOSITORY's config and
    silently ignores the workspace boundary.
    """
    keep = {"HOME", "PATH", "USER", "LOGNAME", "SHELL", "TERM", "TMPDIR"}
    env = {key: value for key, value in os.environ.items()
           if key in keep or key.startswith(("OPENCODE_", "XDG_", "LC_"))}
    env.setdefault("HOME", str(Path.home()))
    env["PWD"] = str(Path(workspace))
    return env


def boundary_fingerprint(workspace: Path, role: str, read_paths,
                         allow_shell: bool = False, allow_write: bool = False,
                         prompt: str = "", model: str = "",
                         source_digest: str = "") -> str:
    """The identity of one run: the boundary, the task and the model.

    Two runs with the same fingerprint had the same view of the filesystem and
    were given the same task; a different fingerprint means the evidence was
    produced under different rules and may not be reused as if it were current.
    The prompt and the model are part of it, so a re-asked task with a changed
    question or a changed model is not answered by the old answer.
    """
    payload = {
        "version": policy.BOUNDARY_VERSION,
        "role": str(role),
        "workspace": str(Path(workspace).resolve()),
        "allow_shell": bool(allow_shell),
        "allow_write": bool(allow_write),
        "read_paths": sorted(str(Path(entry).resolve()) for entry in (read_paths or [])),
        "permissions": isolation_permissions(workspace, role, read_paths,
                                            allow_shell, allow_write),
        "prompt_sha256": digest_bytes(str(prompt).encode()),
        "model": str(model),
        "source_digest": str(source_digest),
    }
    return digest_bytes(json.dumps(payload, sort_keys=True).encode())


def saved_answer(workspace: Path, parser=None, assemble: bool = False) -> str:
    """The last saved record that carries a complete answer.

    A task that ran out of time still saved its records. The last one that
    parses as an answer is what the agent delivered; the caller decides whether
    a partial delivery counts (an idea does, a verdict does not).

    With ``assemble`` the records are joined in order and the parser is tried
    on the whole of them: a task that delivered its answer one block at a time
    (one proposal per record) did deliver the answer. The flag is not used for
    a verdict, where a stage record must never become the final judgement.
    """
    path = Path(workspace) / PROGRESS_FILE_NAME
    records = progress_server.read_records(path)
    if assemble and parser is not None and records:
        # A task that delivered its answer one block at a time did deliver the
        # answer: the blocks are joined in the order they were saved.
        joined = "\n\n".join(str(record.get("content", "")) for record in records
                             if str(record.get("content", "")).strip())
        if joined:
            try:
                parser(joined)
            except Exception:                              # noqa: BLE001
                pass
            else:
                return joined
    for record in reversed(records):
        content = str(record.get("content", "")).strip()
        if not content:
            continue
        if parser is None:
            return content
        try:
            parser(content)
        except Exception:                                  # noqa: BLE001
            continue
        return content
    if assemble and parser is not None and records:
        joined = "\n\n".join(str(record.get("content", "")) for record in records
                             if str(record.get("content", "")).strip())
        if joined:
            try:
                parser(joined)
            except Exception:                              # noqa: BLE001
                return ""
            return joined
    return ""


def _geometry_contract() -> dict:
    """The shared mathematical contract every agent reads, with its digest."""
    import hashlib
    path = Path(__file__).resolve().parents[1] / "contracts" / "HYPERBOLIC_GEOMETRY_CONTRACT.md"
    text = path.read_text(encoding="utf-8")
    version = "poincare-contract/v1"
    for line in text.splitlines():
        if line.startswith("Status:") and "`" in line:
            version = line.split("`")[1]
            break
    return {"path": str(path), "version": version,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_agent(task_id: str, prompt: str, output_name: str, root: Path,
              binary=None, model=None, timeout=None,
              role: str = "verify", read_paths=None, allowed_roots=None,
              candidate_tree=None, allow_shell: bool = False,
              allow_write: bool = False,
              deadline_seconds: int | None = None) -> Path:
    """Run one agent. This is the only place an opencode process is started.

    The boundary is not optional: the agent always runs in a mount namespace
    that holds the workspace, the policy-accepted read allowlist and the
    opencode runtime, and nothing else. A caller cannot widen the view (the
    policy refuses it) and cannot switch the namespace off.

    The agent's one write is its progress file: the ``append_progress`` tool
    appends a record to a file that was bound before the agent started, and the
    generic write and edit tools stay denied.
    """
    # The binary and the model come from the one source of configuration.
    binary = binary or settings.AGENT_BINARY
    model = model or settings.AGENT_MODEL
    # The clock starts HERE, at the dispatch: the initialisation, the model
    # calls, the retries and the teardown all fit in the budget, and the timer
    # is never reset. A run that spends its budget in the setup has no budget
    # left for the agent.
    dispatched = time.monotonic()
    deadline_epoch = (time.time() + float(deadline_seconds)) if deadline_seconds else 0.0
    # The role budget is the DEFAULT, and an explicit timeout wins: a caller that
    # asks for two seconds (a test, or a deliberate short probe) must not be
    # silently given the reviewer's forty minutes.
    if timeout is None:
        timeout = ROLE_TIMEOUT_SECONDS.get(str(role)) or settings.AGENT_TIMEOUT_SECONDS
    if deadline_seconds:
        timeout = min(int(timeout), int(deadline_seconds))
    if (not output_name.endswith((".json", ".txt")) or "/" in output_name
            or "\\" in output_name or not task_id.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("unsafe task id or output filename")
    root = Path(root)
    workspace = root / "agents" / task_id
    # The policy decides what the task may read. It refuses a wider view, a
    # missing path and the directory that holds the other agents' answers, so
    # the caller cannot hand the agent the repository.
    # Every research agent reads the geometry contract: it is shared background,
    # it is inside the roots, and its digest is written on the receipt so a cached
    # answer cannot survive a change of contract.
    contract = _geometry_contract()
    read_paths = list(read_paths or []) + (
        [] if contract["path"] in (read_paths or []) else [contract["path"]])
    # A task that declares no roots is refused BEFORE the shared contract widens
    # anything: the contract is background for a bounded task, not a substitute for
    # declaring the boundary.
    if allowed_roots is not None:
        allowed_roots = list(allowed_roots) + [str(Path(contract["path"]).parent)]
    allowed = policy.check_read_paths(read_paths, roots=allowed_roots,
                                      task_root=root, candidate_tree=candidate_tree)
    if shutil.which(SANDBOX_BINARY) is None:
        raise RuntimeError("bubblewrap (" + SANDBOX_BINARY + ") is required for an "
                           "agent run and is not installed")
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / output_name
    if output.exists():
        raise RuntimeError("refuse to overwrite previous role output")
    # The agent's own progress file, bound before it starts. The controller
    # writes the first record (the task's start), and the append tool adds the
    # stage records after it. Nothing else writes this file.
    progress_path = workspace / PROGRESS_FILE_NAME
    progress = progress_server.ProgressFile(
        progress_path, task_id, deadline_epoch or None,
        settings.AGENT_WRAPUP_MARGIN_SECONDS)
    progress.append(json.dumps({"goal": "the task started", "role": str(role),
                                 "deadline_seconds": deadline_seconds},
                                ensure_ascii=False))
    mcp_servers = {
        "progress": {
            "type": "local",
            "enabled": True,
            "command": [SANDBOX_PYTHON, str(PROGRESS_SERVER),
                        "--file", str(progress_path),
                        "--task-id", task_id,
                        "--deadline-epoch", str(deadline_epoch),
                        "--wrapup-margin", str(settings.AGENT_WRAPUP_MARGIN_SECONDS)],
        },
    }
    isolation = write_isolation_config(workspace, role, allowed, allow_shell,
                                       allow_write, mcp_servers)
    marker = mark_workspace_project(workspace)
    inner_binary = Path(shutil.which(binary) or binary).resolve()
    cmd = [str(inner_binary), "run", "--standalone", "--format", "json", "--auto",
           "--model", model, "--title", task_id, prompt]
    private_home = agent_sandbox_home(workspace)
    cmd = sandbox_command(cmd, workspace, allowed, inner_binary, private_home)
    env = _agent_environment(workspace)
    cancelled, timed_out, stderr = False, False, ""
    # The standard error of the agent goes to a file, not to a pipe: a pipe is
    # inherited by the agent's own children, and a child that outlives the
    # agent keeps the pipe open, so the wait for the agent never ends.
    stderr_path = workspace / "stderr.log"
    try:
        with (workspace / "session.jsonl").open("w", encoding="utf-8") as log, \
                stderr_path.open("w", encoding="utf-8") as errors:
            process = subprocess.Popen(cmd, cwd=workspace, env=env, stdout=log,
                                       stderr=errors, text=True,
                                       start_new_session=True)
            atomic_json(workspace / "running.json", {"pid": process.pid,
                        "started_at": time.time(), "task_id": task_id})
            while True:
                try:
                    process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if (root / "cancel.json").is_file():
                        cancelled = True
                    if time.monotonic() - dispatched > timeout:
                        timed_out = True
                    if cancelled or timed_out:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
            # A run that finished on its own still has to be inside the budget:
            # the budget covers the whole call, not only the part after the
            # process started.
            if not timed_out and time.monotonic() - dispatched > timeout:
                timed_out = True
    except OSError as exc:
        atomic_json(workspace / "execution.json",
                    {"geometry_contract": _geometry_contract(),
                     "task_id": task_id, "error_type": type(exc).__name__,
                     "verified": False})
        raise
    finally:
        (workspace / "running.json").unlink(missing_ok=True)
        if private_home is not None:
            shutil.rmtree(private_home, ignore_errors=True)
    try:
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stderr = ""
    # The Agent is told the absolute destination, but a model can still write the
    # answer into a nested or slightly misnamed directory of its own workspace.
    # The artifact is what matters and the caller validates its content
    # (candidate identity and source digest), so accept a bounded relocation and
    # record where it landed instead of discarding a real session.
    relocated = None
    if not output.is_file() and str(role) in READ_ONLY_ROLES:
        # A read-only role printed its answer instead of writing it.
        printed = session_answer(workspace / "session.jsonl")
        if printed:
            output.write_text(printed, encoding="utf-8")
    if not output.is_file():
        matches = sorted(workspace.rglob(output_name))
        # A repaired run archives older answers under attempt-N/. Never
        # "recover" an archived candidate/review as though this subprocess
        # had produced it: that would make a failed retry look successful.
        matches = [match for match in matches
                   if match.is_file()
                   and ".." not in match.relative_to(workspace).parts
                   and not any(part.startswith("attempt-")
                               for part in match.relative_to(workspace).parts)]
        if matches:
            relocated = matches[0]
            output = relocated
    valid = False
    if output.is_file() and process.returncode == 0 and not timed_out and not cancelled:
        try:
            text = output.read_text(encoding="utf-8")
            if output_name.endswith(".json"):
                valid = isinstance(json.loads(text), dict)
            else:
                # A plain-text answer only has to exist and say something: the
                # caller reads it and decides whether it is usable.
                valid = bool(text.strip())
        except (OSError, ValueError):
            valid = False
    record = {
        "task_id": task_id, "pid": process.pid,
        "finished_utc_epoch": time.time(), "exit_code": process.returncode,
        "timed_out": timed_out, "cancelled": cancelled, "model": model,
        "argv_shape": ["bwrap", "--unshare-all", "--share-net",
                       "opencode", "run", "--standalone", "--format", "json",
                       "--auto", "--model", "<model>", "--title", task_id, "<prompt>"],
        "sandbox_argv": [str(item) for item in cmd],
        "prompt_sha256": digest_bytes(prompt.encode()),
        "session_sha256": digest_file(workspace / "session.jsonl"),
        "isolation_config": isolation.name,
        "isolation_config_sha256": digest_file(isolation),
        "read_allowlist": [str(entry) for entry in allowed] +
                          [str(entry.resolve()) for entry in allowed
                           if str(entry.resolve()) != str(entry)],
        "geometry_contract": _geometry_contract(),
        "reads": session_reads(workspace / "session.jsonl"),
        "boundary_version": policy.BOUNDARY_VERSION,
        "boundary_sha256": boundary_fingerprint(workspace, role, allowed,
                                                allow_shell, allow_write,
                                                prompt, model),
        "elapsed_seconds": round(time.monotonic() - dispatched, 3),
        "deadline_passed": bool(timed_out),
        "progress": {
            "file": str(progress_path),
            "records": len(progress_server.read_records(progress_path)),
            "sha256": digest_file(progress_path) if progress_path.is_file() else None,
            "deadline_seconds": deadline_seconds,
            "deadline_epoch": deadline_epoch or None,
        },
        "sandbox": {
            "enabled": True,
            "binary": SANDBOX_BINARY,
            "private_home": bool(private_home),
            "network": "shared (model provider)",
            "global_config_mounted": False,
        },
        "project_marker": marker.name,
        "isolation_policy": (
            "bubblewrap mount namespace: only the workspace, the policy-accepted "
            "read allowlist and the opencode runtime exist; the repository, the "
            "other arms' results, the earlier sessions and the global opencode "
            "configuration (its MCP servers and plugins) are not mounted; plus the "
            "opencode v2 permission object (bash " +
            ("allowed" if allow_shell else "denied") +
            ", edit and write " +
            ("allowed" if allow_write else "denied") +
            ", read and external_directory deny-by-default with the explicit "
            "allowlist)"),
        "stderr_present": bool(stderr),
        "stderr_tail": (stderr or "")[-4000:],
        "output_name": output_name,
        "output_sha256": digest_file(output) if output.is_file() else None,
        "verified": valid,
        "limitations": ("The mount namespace is the filesystem guarantee; the model "
                        "provider is reachable over the shared network, so a model "
                        "that was shown a file can still send it away."),
    }
    record["failure_class"] = "none" if valid else class_of_failure(
        workspace / "session.jsonl", timed_out=timed_out, cancelled=cancelled,
        exit_code=process.returncode, wrote_answer=output.is_file())
    receipt = workspace / "execution.json"
    atomic_json(receipt, record)
    if not valid:
        # An answer is published only when the run was valid. A timed-out run can
        # leave half a report behind, and a half report read as an answer is how a
        # cancelled reviewer became a verdict.
        if output.is_file():
            partial = workspace / "partial"
            partial.mkdir(parents=True, exist_ok=True)
            output.replace(partial / output_name)
        raise AgentFailure(record["failure_class"], str(receipt),
                           "OpenCode did not produce a usable answer (" +
                           record["failure_class"] + "): " + str(receipt))
    return receipt
