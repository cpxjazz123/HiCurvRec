"""The LangChain backend for the research agents.

The OpenCode backend starts a CLI in a bubblewrap mount namespace. This backend
creates the agent with LangChain instead: ``langchain.agents.create_agent`` over a
chat model, with the project's own boundary expressed as the agent's TOOLS. The
task contract does not change: the agent runs in its own workspace under
``root/agents/<task_id>``, its answer is a file the caller parses, its progress
records travel through the same progress channel, and the run leaves the same
receipt.

The boundary is enforced by the tools, not by a mount namespace: a read outside
the workspace and the policy-approved roots is refused, and a write outside the
workspace is refused. That is a weaker isolation than bubblewrap, so the backend
is opt-in (``settings.AGENT_BACKEND = "langchain"``) and it refuses to run
without a configured chat model.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from tools.research import settings
from tools.research.agents import policy, progress_server, runner

#: The progress channel refuses a record longer than this; a long answer is
#: saved as several records, which the reader joins when it assembles an answer.
PROGRESS_CHUNK = 3500

SYSTEM_PROMPT = (
    "You are a research agent of a scientific programme. Work inside your own "
    "workspace, read only what your task allows, and answer in the exact format "
    "the task asks for. Record every meaningful step with the append_progress "
    "tool: each record must stand on its own, and the complete answer goes in "
    "the last record."
)


class AgentBackendUnavailable(RuntimeError):
    """The LangChain backend has no chat model to create an agent with."""


def available() -> bool:
    """Whether a chat model is configured for this backend."""
    return bool((settings.AGENT_CHAT_MODEL or "").strip())


def _strip_jsonc(text: str) -> str:
    """JSON with its comments removed, strings left alone.

    A regular expression cannot do this: the URLs in the configuration contain
    "//", so the comments have to be found by a scanner that knows when it is
    inside a string.
    """
    out = []
    index = 0
    length = len(text)
    in_string = False
    escaped = False
    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length:
            following = text[index + 1]
            if following == "/":
                while index < length and text[index] != "\n":
                    index += 1
                continue
            if following == "*":
                index += 2
                while index + 1 < length and not (text[index] == "*" and text[index + 1] == "/"):
                    index += 1
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def opencode_provider(model_id: str) -> tuple:
    """The base URL and key of the provider OpenCode knows for this model.

    The credential stays in the user's own OpenCode configuration: the
    repository never carries a key, and the LangChain backend reuses the
    provider the CLI already talks to, so both backends use the same model.
    """
    import json

    config = Path.home() / ".config" / "opencode" / "opencode.jsonc"
    if not config.is_file():
        return None
    try:
        data = json.loads(_strip_jsonc(config.read_text(encoding="utf-8")))
    except ValueError:
        return None
    providers = (data or {}).get("provider") or {}
    name, _, model = model_id.partition("/")
    provider = providers.get(name)
    if not provider:
        return None
    options = provider.get("options") or {}
    base_url = options.get("baseURL")
    api_key = options.get("apiKey")
    import os
    if isinstance(api_key, str) and api_key.startswith("{env:") and api_key.endswith("}"):
        # The configuration names the environment variable that holds the key.
        api_key = os.environ.get(api_key[5:-1], "")
    if not api_key:
        # The CLI's own credential file sits next to its configuration; it is the
        # same key the CLI sends, and it is never copied into the repository.
        candidate = config.parent / (name + ".key")
        if candidate.is_file():
            api_key = candidate.read_text(encoding="utf-8").strip()
    if not base_url or not api_key:
        return None
    return base_url, api_key, model


def build_model(model_id: str | None = None):
    """The chat model the agents are created with."""
    identifier = (model_id or settings.AGENT_CHAT_MODEL or "").strip()
    if not identifier:
        raise AgentBackendUnavailable(
            "the LangChain backend needs a chat model: set AGENT_CHAT_MODEL "
            "(for example 'anthropic:claude-sonnet-4-5', 'openai:gpt-4o-mini' or "
            "'opencode:commandcode/deepseek/deepseek-v4.1-flash'). The key and the "
            "base URL come from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY, "
            "and their base URLs); the 'opencode:' form reads the provider the CLI "
            "already uses.")
    if identifier.startswith("opencode:"):
        provider = opencode_provider(identifier.split(":", 1)[1])
        if provider is None:
            raise AgentBackendUnavailable(
                "the OpenCode configuration has no provider for " + identifier)
        base_url, api_key, model = provider
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                          temperature=0.3, max_tokens=8192,
                          # The provider answers with brotli, and the installed
                          # brotli/httpx pair cannot decode it; asking for gzip
                          # keeps the response readable.
                          default_headers={"Accept-Encoding": "gzip, deflate"})
    from langchain.chat_models import init_chat_model
    return init_chat_model(identifier)


def append_progress(progress, content: str) -> dict:
    """Save a record, split when it is longer than the channel allows.

    The progress channel refuses a record over 4000 characters; a long answer is
    saved as several records instead of being lost, and the reader joins them
    when it assembles an answer.
    """
    text = str(content)
    if len(text) <= PROGRESS_CHUNK:
        return progress.append(text)
    saved = progress.append(text[:PROGRESS_CHUNK])
    for start in range(PROGRESS_CHUNK, len(text), PROGRESS_CHUNK):
        saved = progress.append(text[start:start + PROGRESS_CHUNK])
    return saved


def workspace_tools(workspace: Path, read_paths, *, allow_write: bool = False,
                    progress=None):
    """The tools a LangChain agent may use, each with the project's boundary."""
    from langchain_core.tools import tool

    workspace = Path(workspace).resolve()
    allowed = [Path(entry).resolve() for entry in (read_paths or [])]

    def _readable(candidate: Path) -> bool:
        resolved = candidate.resolve()
        if resolved == workspace or workspace in resolved.parents:
            return True
        return any(resolved == entry or entry in resolved.parents
                   for entry in allowed)

    def _writable(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return resolved == workspace or workspace in resolved.parents

    @tool
    def read_file(path: str) -> str:
        """Read a file inside the task workspace or an allowed read root."""
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _readable(target):
            raise ValueError("refused: " + str(target) + " is outside the allowed roots")
        return target.read_text(encoding="utf-8", errors="replace")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a file inside the task workspace."""
        if not allow_write:
            raise ValueError("this task has no write tool")
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _writable(target):
            raise ValueError("refused: " + str(target) + " is outside the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "wrote " + str(target)

    @tool
    def list_directory(path: str = ".") -> str:
        """List a directory inside the task workspace or an allowed read root."""
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _readable(target):
            raise ValueError("refused: " + str(target) + " is outside the allowed roots")
        if not target.is_dir():
            raise ValueError("not a directory: " + str(target))
        return "\n".join(sorted(entry.name + ("/" if entry.is_dir() else "")
                                for entry in target.iterdir()))

    @tool
    def append_progress(content: str) -> str:
        """Append one stage record to this task's own progress file."""
        if progress is None:
            raise ValueError("this task has no progress channel")
        saved = append_progress(progress, content)
        return "saved as record " + str(saved["seq"]) + " of this task's progress file"

    return [read_file, write_file, list_directory, append_progress]


def run_agent(task_id: str, prompt: str, output_name: str, root: Path, *,
              model=None, timeout=None, role: str = "verify", read_paths=None,
              allowed_roots=None, candidate_tree=None, allow_shell: bool = False,
              allow_write: bool = False, deadline_seconds: int | None = None,
              source_digest: str = "") -> Path:
    """Run one agent with a LangChain agent and return its receipt path."""
    if (not output_name.endswith((".json", ".txt")) or "/" in output_name
            or "\\" in output_name
            or not task_id.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("unsafe task id or output filename")
    root = Path(root)
    workspace = root / "agents" / task_id
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / output_name
    if output.exists():
        raise RuntimeError("refuse to overwrite previous role output")
    contract = runner._geometry_contract()
    read_paths = list(read_paths or []) + (
        [] if contract["path"] in (read_paths or []) else [contract["path"]])
    if allowed_roots is not None:
        allowed_roots = list(allowed_roots) + [str(Path(contract["path"]).parent)]
    allowed = policy.check_read_paths(read_paths, roots=allowed_roots,
                                      task_root=root, candidate_tree=candidate_tree)
    deadline = (time.time() + float(deadline_seconds)) if deadline_seconds else None
    progress = progress_server.ProgressFile(workspace / runner.PROGRESS_FILE_NAME,
                                             task_id, deadline=deadline)
    tools = workspace_tools(workspace, allowed, allow_write=allow_write,
                            progress=progress)
    # The caller may pass the CLI's model id ("commandcode/deepseek/..."), which
    # is not a LangChain identifier. Only an explicit "provider:model" is taken
    # from the caller; otherwise the configured chat model is used.
    chat_model_id = model if (model and ":" in str(model)) else None
    chat_model = build_model(chat_model_id)
    from langchain.agents import create_agent
    agent = create_agent(model=chat_model, tools=tools, system_prompt=SYSTEM_PROMPT)
    started = time.time()
    answer_text = ""
    failure = ""
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                              config={"recursion_limit": 60})
        messages = result.get("messages") or []
        for message in reversed(messages):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                answer_text = content.strip()
                break
    except Exception as exc:                                   # noqa: BLE001
        failure = type(exc).__name__ + ": " + str(exc)[:300]
    if answer_text and not output.is_file():
        # A read-only role has no write tool: its reply IS the answer.
        output.write_text(answer_text + "\n", encoding="utf-8")
    if answer_text:
        try:
            append_progress(progress, answer_text)
        except Exception:                                      # noqa: BLE001
            pass
    boundary = runner.boundary_fingerprint(
        workspace, role, runner.effective_read_paths(read_paths), allow_shell,
        allow_write, prompt, settings.AGENT_CHAT_MODEL,
        source_digest=source_digest)
    receipt = {
        "task_id": task_id, "backend": "langchain", "verified": bool(output.is_file()
                                                                     and not failure),
        "finished_utc_epoch": time.time(), "elapsed_s": round(time.time() - started, 3),
        "model": settings.AGENT_CHAT_MODEL, "role": role,
        "boundary_sha256": boundary, "read_allowlist": [str(entry) for entry in allowed],
        "geometry_contract": contract, "failure": failure,
        "progress_file": str(workspace / runner.PROGRESS_FILE_NAME),
        "answer": str(output),
    }
    receipt_path = workspace / "execution.json"
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    if failure and not output.is_file():
        raise runner.AgentFailure("AGENT_CRASHED", str(receipt_path), failure)
    return receipt_path
