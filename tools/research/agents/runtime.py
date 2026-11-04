"""The agent runtime: agents created and run by the OpenAI Agents SDK.

The SDK owns the agent: ``agents.Agent`` with the tools the task may use, and
``agents.Runner`` for the run itself. The project's boundary is expressed as
those tools - a read outside the workspace and the policy-approved roots is
refused, a write outside the workspace is refused, there is no shell - and the
task contract is the one the framework has always read: the workspace is
``root/agents/<task_id>``, the answer is the file the caller parses, the
progress channel holds the records, and the run leaves ``execution.json`` with
the boundary fingerprint that says whether the answer may be reused.
"""
from __future__ import annotations

import json
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
    "the last record. Your last message is also read as your answer, so it must "
    "carry the complete answer in the requested format, not a confirmation. "
    "Never copy the task text into a file: the file must hold your own answer, "
    "and a file that repeats the instructions is treated as no answer at all. "
    "Answer with the fields the task asks for, in that order, and start with the "
    "first field; do not restate the request."
)


class AgentBackendUnavailable(RuntimeError):
    """There is no model the runtime could create an agent with."""


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


def configured_provider(model_id: str) -> tuple:
    """The base URL and key of the provider that serves this model.

    The credential stays in the user's own agent configuration: the repository
    never carries a key. The provider is found by the model name, so the setting
    reads as a model name and not as a tool name.
    """
    import os

    # The provider configuration lives in the user's own agent configuration;
    # the repository never carries a key.
    config = Path.home() / ".config" / "opencode" / "opencode.jsonc"
    if not config.is_file():
        return None
    try:
        data = json.loads(_strip_jsonc(config.read_text(encoding="utf-8")))
    except ValueError:
        return None
    providers = (data or {}).get("provider") or {}
    provider = None
    model = model_id
    name = ""
    for candidate_name, candidate in providers.items():
        models = candidate.get("models") or {}
        if model_id in models:
            provider = candidate
            model = model_id
            name = candidate_name
            break
    if provider is None:
        name, _, tail = model_id.partition("/")
        provider = providers.get(name)
        model = tail or model_id
    if not provider:
        return None
    options = provider.get("options") or {}
    base_url = options.get("baseURL")
    api_key = options.get("apiKey")
    if isinstance(api_key, str) and api_key.startswith("{env:") and api_key.endswith("}"):
        api_key = os.environ.get(api_key[5:-1], "")
    if not api_key:
        # The CLI's own credential file sits next to its configuration; it is the
        # same key, and it is never copied into the repository.
        candidate = config.parent / (name + ".key")
        if candidate.is_file():
            api_key = candidate.read_text(encoding="utf-8").strip()
    if not base_url or not api_key:
        return None
    return base_url, api_key, model


def build_model(model_id: str | None = None):
    """The model the SDK agent runs on.

    A model the configured provider serves is used with that provider's URL and
    key; otherwise the SDK's own resolution applies (the OpenAI key and base URL
    from the environment).
    """
    identifier = (model_id or settings.AGENT_CHAT_MODEL or "").strip()
    if not identifier:
        raise AgentBackendUnavailable(
            "the agent runtime needs a model: set AGENT_CHAT_MODEL "
            "(for example 'deepseek/deepseek-v4.1-flash' or "
            "'openai:gpt-4o-mini'). A model the configured provider serves is used "
            "with that provider's URL and key; otherwise the key and the base URL "
            "come from the environment (OPENAI_API_KEY and OPENAI_BASE_URL).")
    provider = configured_provider(identifier)
    if provider is not None:
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI
        base_url, api_key, model = provider
        client = AsyncOpenAI(base_url=base_url, api_key=api_key,
                             default_headers={"Accept-Encoding": "gzip, deflate"})
        return OpenAIChatCompletionsModel(model=model, openai_client=client)
    if identifier.startswith("openai:"):
        return identifier.split(":", 1)[1]
    raise AgentBackendUnavailable(
        "no provider serves " + identifier + ": the configured provider has no "
        "such model and the SDK cannot resolve it without an OpenAI key.")


def save_progress(progress, content: str) -> dict:
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


def echoes_the_prompt(text: str, prompt: str) -> bool:
    """Whether the delivered answer is the task text rather than an answer.

    An agent asked to write an answer sometimes writes the prompt back. The
    framework must not read that as a proposal, so the echo is detected here and
    treated as a missing answer.
    """
    body = str(text or "").strip()
    if not body:
        return False
    head = body[:200].lower()
    first = str(prompt or "").strip()[:120].lower()
    if first and head.startswith(first[:60]):
        return True
    return body.lower().startswith("propose exactly") and "distinct" in head


def workspace_tools(workspace: Path, read_paths, *, allow_write: bool = False,
                    progress=None, refusals: list | None = None) -> list:
    """The tools an agent may use, each with the project's boundary.

    They are plain functions: the SDK wraps them into its own tool objects, and
    the tests call them directly.
    """
    workspace = Path(workspace).resolve()
    allowed = [Path(entry).resolve() for entry in (read_paths or [])]
    refusals = refusals if refusals is not None else []

    def _refuse(tool: str, target: Path, reason: str) -> None:
        refusals.append({"tool": tool, "path": str(target), "reason": reason})

    def _readable(candidate: Path) -> bool:
        resolved = candidate.resolve()
        if resolved == workspace or workspace in resolved.parents:
            return True
        return any(resolved == entry or entry in resolved.parents
                   for entry in allowed)

    def _writable(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return resolved == workspace or workspace in resolved.parents

    def read_file(path: str) -> str:
        """Read a file inside the task workspace or an allowed read root."""
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _readable(target):
            _refuse("read_file", target, "outside the allowed roots")
            raise ValueError("refused: " + str(target) + " is outside the allowed roots")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(path: str, content: str) -> str:
        """Write a file inside the task workspace."""
        if not allow_write:
            raise ValueError("this task has no write tool")
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _writable(target):
            _refuse("write_file", target, "outside the workspace")
            raise ValueError("refused: " + str(target) + " is outside the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "wrote " + str(target)

    def list_directory(path: str = ".") -> str:
        """List a directory inside the task workspace or an allowed read root."""
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _readable(target):
            _refuse("list_directory", target, "outside the allowed roots")
            raise ValueError("refused: " + str(target) + " is outside the allowed roots")
        if not target.is_dir():
            raise ValueError("not a directory: " + str(target))
        return "\n".join(sorted(entry.name + ("/" if entry.is_dir() else "")
                                for entry in target.iterdir()))

    def append_progress(content: str) -> str:
        """Append one stage record to this task's own progress file."""
        if progress is None:
            raise ValueError("this task has no progress channel")
        saved = save_progress(progress, content)
        return "saved as record " + str(saved["seq"]) + " of this task's progress file"

    return [read_file, write_file, list_directory, append_progress]


def run_turn(agent, prompt: str, max_turns: int = 40) -> str:
    """Run the agent and return what it answered."""
    from agents import Runner
    result = Runner.run_sync(agent, prompt, max_turns=max_turns)
    return str(getattr(result, "final_output", "") or "")


def run_agent(task_id: str, prompt: str, output_name: str, root: Path, *,
              model=None, timeout=None, role: str = "verify", read_paths=None,
              allowed_roots=None, candidate_tree=None, allow_shell: bool = False,
              allow_write: bool = False, deadline_seconds: int | None = None,
              source_digest: str = "") -> Path:
    """Run one agent with the OpenAI Agents SDK and return its receipt path."""
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
    refusals: list = []
    functions = workspace_tools(workspace, allowed, allow_write=allow_write,
                                progress=progress, refusals=refusals)
    from agents import Agent, function_tool
    agent = Agent(name=task_id, instructions=SYSTEM_PROMPT,
                  tools=[function_tool(function) for function in functions],
                  model=build_model(model))
    started = time.time()
    answer_text = ""
    failure = ""
    failure_trace = ""
    try:
        answer_text = run_turn(agent, prompt)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        failure = type(exc).__name__ + ": " + str(exc)[:300]
        failure_trace = traceback.format_exc()[-1500:]
    if not output.is_file():
        # A read-only role has no write tool, and an agent that answered in the
        # chat instead of the file still delivered its answer: the records are
        # what the framework reads, so they are what the answer file carries.
        records = progress_server.read_records(workspace / runner.PROGRESS_FILE_NAME)
        joined = "\n\n".join(str(record.get("content", "")) for record in records
                             if str(record.get("content", "")).strip())
        text = joined or answer_text
        if text and not echoes_the_prompt(text, prompt):
            output.write_text(text + "\n", encoding="utf-8")
    if answer_text:
        try:
            save_progress(progress, answer_text)
        except Exception:                                      # noqa: BLE001
            pass
    boundary = runner.boundary_fingerprint(
        workspace, role, runner.effective_read_paths(read_paths), allow_shell,
        allow_write, prompt, settings.AGENT_CHAT_MODEL,
        source_digest=source_digest)
    receipt = {
        "task_id": task_id, "backend": "openai-agents-sdk", "verified": bool(output.is_file()
                                                                           and not failure),
        "finished_utc_epoch": time.time(), "elapsed_s": round(time.time() - started, 3),
        "model": settings.AGENT_CHAT_MODEL, "role": role,
        "boundary_sha256": boundary, "read_allowlist": [str(entry) for entry in allowed],
        "geometry_contract": contract, "failure": failure, "refusals": refusals,
        "failure_trace": failure_trace,
        "progress_file": str(workspace / runner.PROGRESS_FILE_NAME),
        "answer": str(output),
    }
    receipt_path = workspace / "execution.json"
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    if failure and not output.is_file():
        raise runner.AgentFailure("AGENT_CRASHED", str(receipt_path), failure)
    return receipt_path
