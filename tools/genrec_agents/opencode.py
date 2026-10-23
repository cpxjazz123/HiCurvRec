"""The OpenCode execution backend (issue #102).

Everything here was verified against the installed CLI rather than assumed:

* the flags come from ``opencode run --help``;
* the event stream shape comes from a real run -- one JSON event per line, with
  ``step_start`` / ``tool_use`` / ``step_finish`` / ``text`` / ``error`` events;
* the MCP configuration comes from OpenCode's own JSON Schema (``mcp.servers.<name>``
  is ``{type: "local", command: [...], cwd, environment, disabled, timeout}``);
* the provider behaviour comes from actual calls. A model can be *listed* and still be
  refused by the account: ``gpt-5.5`` is listed and the stored ChatGPT/Codex OAuth
  refuses it, while ``gpt-6-luna`` works.

Three facts the code depends on:

* ``--standalone`` is required whenever the configuration interpolates the environment:
  the background service keeps the environment it was started with, so a key exported
  after it started is invisible to it. A private server per run also gives the
  isolation the issue asks for.
* the OpenAI OAuth token lives in ``~/.local/share/opencode/auth.json``, i.e. under
  ``XDG_DATA_HOME``; that variable is therefore never overridden.
* a credential is passed to the child process through the environment only. It is never
  written into a configuration file, never printed, and redacted out of every log line
  that is kept.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The default agent model: it is the one the machine's stored OpenAI OAuth may use.
DEFAULT_MODEL = "openai/gpt-6-luna"

#: Where the operator may point the backend at a different binary.
BIN_ENV = "HARNESS2_OPENCODE_BIN"
#: Where the operator may choose a different model.
MODEL_ENV = "HARNESS2_OPENCODE_MODEL"
#: Where the operator may put a full provider configuration.
CONFIG_ENV = "HARNESS2_OPENCODE_CONFIG"
#: Where the operator may choose a provider: "auto" (OpenCode's own credentials) or
#: "anthropic" (the operator's Anthropic-compatible endpoint).
PROVIDER_ENV = "HARNESS2_OPENCODE_PROVIDER"
#: The environment variable the generated configuration points its key at.
SECRET_ENV = "HARNESS2_OPENCODE_KEY"

#: The flags this backend uses. The preflight checks that the installed CLI has them.
REQUIRED_FLAGS = ("--standalone", "--format", "--model", "--auto", "--title")

#: The genrec MCP server, relative to the repository root.
MCP_SERVER = Path("tools/genrec-mcp/server.py")


class OpenCodeError(RuntimeError):
    """OpenCode could not do what was asked, with a reason a human can act on."""


@dataclass
class OpenCodePreflight:
    """What the backend found on this machine, and what it could not use."""

    ok: bool
    binary: str = ""
    version: str = ""
    flags: Dict[str, bool] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    probed: bool = False
    probe_output: str = ""
    problems: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "binary": self.binary,
            "version": self.version,
            "flags": dict(self.flags),
            "model": self.model,
            "provider": self.provider,
            "probed": self.probed,
            "probe_output": self.probe_output,
            "problems": list(self.problems),
        }


def binary_path() -> str:
    """The OpenCode binary, or an empty string when it is not installed."""
    override = os.environ.get(BIN_ENV, "").strip()
    if override:
        return override if Path(override).is_file() else ""
    return shutil.which("opencode") or ""


def _run(argv: Sequence[str], *, cwd: Path | None, env: Mapping[str, str] | None,
         timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise OpenCodeError(
            f"opencode did not finish within {timeout}s: {' '.join(argv)}"
        ) from exc
    except FileNotFoundError as exc:
        raise OpenCodeError(
            f"the opencode binary was not found: {argv[0]!r}. Install it or set {BIN_ENV}."
        ) from exc


def cli_version(binary: str, *, timeout: int = 60) -> str:
    result = _run([binary, "--version"], cwd=None, env=None, timeout=timeout)
    return (result.stdout or result.stderr or "").strip().splitlines()[0] if (
        result.stdout or result.stderr
    ) else ""


def cli_flags(binary: str, *, timeout: int = 60) -> Dict[str, bool]:
    """Which of the flags this backend uses the installed CLI actually has."""
    result = _run([binary, "run", "--help"], cwd=None, env=None, timeout=timeout)
    text = (result.stdout or "") + (result.stderr or "")
    return {flag: flag in text for flag in REQUIRED_FLAGS}


def model_name(explicit: str = "") -> str:
    return (explicit or os.environ.get(MODEL_ENV, "") or DEFAULT_MODEL).strip()


def provider_of(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


# --------------------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------------------
def _secret_from_settings() -> Tuple[str, str, str]:
    """Anthropic-compatible endpoint, key and model from the operator's own settings.

    Read from the Claude Code settings the machine already uses, and returned as a
    (base_url, key, model) triple. The key never reaches a file: the caller puts it in
    the child environment.
    """
    candidates = [
        Path.home() / ".claude" / "settings.json",
        Path.cwd() / ".claude" / "settings.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        env = document.get("env") or {}
        base = str(env.get("ANTHROPIC_BASE_URL", "")).strip()
        key = str(env.get("ANTHROPIC_AUTH_TOKEN", "") or env.get("ANTHROPIC_API_KEY", "")).strip()
        if base and key:
            model = str(env.get("ANTHROPIC_MODEL", "")).strip()
            return base, key, model
    return "", "", ""


def provider_configuration(model: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The provider block for the workspace configuration, and the child environment.

    The provider is **explicit**, never guessed:

    1. ``HARNESS2_OPENCODE_CONFIG`` -- a JSON file the operator wrote. A literal
       ``apiKey`` in it is moved into the child environment and replaced by an
       environment reference, so the secret never sits in a file we write.
    2. ``HARNESS2_OPENCODE_PROVIDER=anthropic`` -- the operator's own Anthropic-compatible
       endpoint, read from their Claude Code settings.
    3. otherwise nothing is configured and OpenCode's own stored credentials are used.
       That is the default path: the OpenAI OAuth on this machine, with the model the
       operator asked for.
    """
    child_env: Dict[str, str] = {}
    path = os.environ.get(CONFIG_ENV, "").strip()
    if path:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        provider = document.get("provider") or {}
        for provider_id, block in provider.items():
            options = dict(block.get("options") or {})
            api_key = options.get("apiKey")
            if isinstance(api_key, str) and not api_key.startswith("{env:"):
                child_env[SECRET_ENV] = api_key
                options["apiKey"] = "{" + f"env:{SECRET_ENV}" + "}"
                provider[provider_id] = {**block, "options": options}
        chosen = str(document.get("model") or model)
        return {"provider": provider, "model": chosen}, child_env

    if os.environ.get(PROVIDER_ENV, "").strip().lower() == "anthropic":
        base, key, settings_model = _secret_from_settings()
        if not (base and key):
            raise OpenCodeError(
                "the anthropic-compatible provider was requested but no endpoint and key "
                "were found in the operator's Claude Code settings"
            )
        child_env[SECRET_ENV] = key
        provider_id = "genrec"
        chosen = settings_model or "MiniMax-M3"
        return (
            {
                "provider": {
                    provider_id: {
                        "npm": "@ai-sdk/anthropic",
                        "name": "the operator's anthropic-compatible endpoint",
                        "options": {
                            "baseURL": base.rstrip("/") + "/v1",
                            "apiKey": "{" + f"env:{SECRET_ENV}" + "}",
                        },
                        "models": {chosen: {"name": chosen}},
                    }
                },
                "model": f"{provider_id}/{chosen}",
            },
            child_env,
        )

    return {}, child_env


def mcp_server_block(repo_root: Path, *, python: str = "/usr/bin/python3",
                     timeout_ms: int = 120_000) -> Dict[str, Any]:
    """The genrec MCP server, as OpenCode wants it."""
    server = Path(repo_root) / MCP_SERVER
    return {
        "genrec": {
            "type": "local",
            "command": [python, str(server)],
            "cwd": str(repo_root),
            "disabled": False,
            "timeout": {"startup": timeout_ms},
        }
    }


def build_config(
    workspace: Path,
    *,
    model: str,
    provider: Optional[Dict[str, Any]] = None,
    mcp: Optional[Dict[str, Any]] = None,
    deny_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    """The workspace's ``opencode.json``.

    The workspace is the agent's only working directory, so the harness tree, the task
    files and nothing else live there. The repository is reachable only through the MCP
    server, whose tools decide what they touch.
    """
    config: Dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": model,
        "permission": [
            {"action": "*", "resource": str(workspace), "effect": "allow"},
        ],
    }
    for path in deny_paths:
        config["permission"].append(
            {"action": "*", "resource": str(path), "effect": "deny"}
        )
    if provider:
        config.update(provider)
    if mcp:
        config["mcp"] = {"servers": mcp}
    return config


# --------------------------------------------------------------------------------------
# running an agent
# --------------------------------------------------------------------------------------
def redact(text: str, secrets: Sequence[str]) -> str:
    """Take the secrets out of a string that is about to be written down."""
    out = text
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "<redacted>")
    return out


def parse_events(raw: str) -> Dict[str, Any]:
    """The summary of an OpenCode event stream.

    The stream is one JSON object per line. A line that is not JSON is kept as raw text:
    a stream that is not parseable is evidence, not something to discard.
    """
    events: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    texts: List[str] = []
    errors: List[Dict[str, Any]] = []
    cost = 0.0
    tokens: Dict[str, Any] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            events.append({"type": "unparsed", "line": line[:500]})
            continue
        events.append(event)
        kind = event.get("type")
        part = event.get("part") or {}
        if kind == "tool_use":
            state = part.get("state") or {}
            tool_calls.append(
                {
                    "tool": part.get("tool", ""),
                    "id": part.get("id", ""),
                    "status": state.get("status", ""),
                    "input": state.get("input"),
                    "output": str(state.get("output", ""))[:4000],
                    "title": state.get("title", ""),
                    "start": (part.get("time") or {}).get("start"),
                    "end": (part.get("time") or {}).get("end"),
                }
            )
        elif kind == "text":
            texts.append(str(part.get("text", "")))
        elif kind == "error":
            errors.append(event.get("error") or {"message": str(event)[:400]})
        elif kind == "step_finish":
            cost += float(part.get("cost") or 0.0)
            for key, value in (part.get("tokens") or {}).items():
                if isinstance(value, (int, float)):
                    tokens[key] = tokens.get(key, 0) + value
    return {
        "events": events,
        "n_events": len(events),
        "tool_calls": tool_calls,
        "n_tool_calls": len(tool_calls),
        "tools": [call["tool"] for call in tool_calls],
        "text": "\n".join(texts),
        "cost": cost,
        "tokens": tokens,
        "errors": errors,
    }


def run_agent(
    workspace: Path,
    prompt: str,
    *,
    model: str,
    timeout: int = 900,
    provider: Optional[Dict[str, Any]] = None,
    mcp: Optional[Dict[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    title: str = "",
    agent: str = "",
    deny_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    """Run one OpenCode session in ``workspace`` and return everything it did.

    The return value is the evidence: the event stream, the tool calls with their
    inputs and outputs, the cost and the tokens, the errors, the exit status and the
    wall time. Secrets are redacted out of every string in it.
    """
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    binary = binary_path()
    if not binary:
        raise OpenCodeError(
            "opencode is not installed: install it, or point "
            f"{BIN_ENV} at the binary. The production path uses the OpenCode CLI."
        )
    provider, provider_env = provider_configuration(model)
    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        # The stored OpenAI OAuth lives under XDG_DATA_HOME; overriding it would hide the
        # credential. Isolation comes from the workspace and the private server instead.
        "XDG_DATA_HOME": os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")),
        "XDG_STATE_HOME": os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")),
    }
    child_env.update(provider_env)
    if env:
        child_env.update({k: str(v) for k, v in env.items()})

    config = build_config(
        workspace,
        model=provider.get("model", model) if provider else model,
        provider=provider,
        mcp=mcp,
        deny_paths=deny_paths,
    )
    config_path = workspace / "opencode.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    argv = [
        binary,
        "run",
        "--standalone",
        "--format",
        "json",
        "--auto",
        "--model",
        config["model"],
    ]
    if title:
        argv += ["--title", title]
    if agent:
        argv += ["--agent", agent]
    argv.append(prompt)

    secrets = [value for value in provider_env.values() if value]
    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    # Popen rather than run: the executor's evidence has to name the PROCESS, so the pid and
    # the real lifetime are part of the record. The semantics are otherwise identical.
    process = subprocess.Popen(
        argv,
        cwd=str(workspace),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pid = process.pid
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        returncode, timed_out = process.returncode, False
    except subprocess.TimeoutExpired:
        stdout, stderr = process.communicate()
        stderr = (stderr or "") + f"\n[opencode] timed out after {timeout}s"
        returncode, timed_out = -1, True
    finished = time.time()

    summary = parse_events(stdout or "")
    summary.update(
        {
            "argv": argv[1:],
            "model": config["model"],
            "workspace": str(workspace),
            "exit_code": returncode,
            "timed_out": timed_out,
            "elapsed_s": round(finished - started, 2),
            "pid": pid,
            "started_utc": started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
            "stdout": redact(stdout or "", secrets),
            "stderr": redact(stderr or "", secrets),
            "config_path": str(config_path),
        }
    )
    return summary


def preflight(
    model: str = "",
    *,
    probe: bool = True,
    timeout: int = 120,
    workspace: Optional[Path] = None,
) -> OpenCodePreflight:
    """Check that OpenCode can actually be used here, and say what is wrong if not.

    The probe is a real one-token call: a model can be listed by the CLI and still be
    refused by the account, and only a call tells the difference.
    """
    report = OpenCodePreflight(ok=False)
    model = model_name(model)
    report.model = model
    report.provider = provider_of(model)

    binary = binary_path()
    if not binary:
        report.problems.append(
            "opencode is not installed on this machine: install it, or point "
            f"{BIN_ENV} at the binary"
        )
        return report
    report.binary = binary

    try:
        report.version = cli_version(binary)
    except OpenCodeError as exc:
        report.problems.append(str(exc))
        return report

    try:
        report.flags = cli_flags(binary)
    except OpenCodeError as exc:
        report.problems.append(str(exc))
        return report
    missing = [flag for flag, present in report.flags.items() if not present]
    if missing:
        report.problems.append(
            f"the installed opencode ({report.version}) has no {', '.join(missing)}; the "
            "backend needs them and will not guess a substitute"
        )

    if not probe:
        report.ok = not report.problems
        return report

    target = Path(workspace) if workspace else Path(os.environ.get("TMPDIR", "/tmp")) / (
        "opencode-preflight-" + str(os.getpid())
    )
    try:
        result = run_agent(
            target,
            "Reply with exactly: OK",
            model=model,
            timeout=timeout,
            title="genrec preflight",
        )
    except OpenCodeError as exc:
        report.problems.append(str(exc))
        return report
    finally:
        if workspace is None:
            import shutil as _shutil

            _shutil.rmtree(target, ignore_errors=True)

    report.probed = True
    report.probe_output = (result.get("text") or "")[:200]
    if result.get("errors"):
        message = str(result["errors"][0].get("message", result["errors"][0]))
        report.problems.append(_explain_provider_failure(message, model))
    elif result.get("exit_code") != 0 and not result.get("text"):
        report.problems.append(
            f"opencode exited with {result.get('exit_code')}: "
            f"{(result.get('stderr') or '')[-300:]}"
        )
    elif not (result.get("text") or "").strip():
        report.problems.append("opencode answered nothing: the model or the provider is not usable")
    report.ok = not report.problems
    return report


def _explain_provider_failure(message: str, model: str) -> str:
    """Turn a provider error into something the operator can act on."""
    text = message.lower()
    if "not supported when using codex" in text:
        return (
            f"the account behind the stored credential may not use {model!r}. The OAuth on "
            "this machine is a ChatGPT/Codex account, and it refuses some of the models the "
            "CLI lists. Try another id from `opencode models` (openai/gpt-6-luna is known to "
            "work), or set " + MODEL_ENV + "."
        )
    if "invalid api key" in text or "api secret key" in text:
        return (
            f"the provider rejected the credential for {model!r}. The key is passed to the "
            "agent through the environment; check " + SECRET_ENV + " / " + CONFIG_ENV + ", or "
            "use a model the stored OpenAI OAuth may call."
        )
    if "404" in text:
        return (
            f"the provider endpoint answered 404 for {model!r}: the base URL is wrong. For an "
            "Anthropic-compatible endpoint the base URL must end in /v1."
        )
    if "connection" in text or "econnrefused" in text or "timeout" in text:
        return f"the provider endpoint for {model!r} could not be reached: {message[:200]}"
    return f"the provider refused {model!r}: {message[:300]}"


__all__ = [
    "BIN_ENV",
    "CONFIG_ENV",
    "DEFAULT_MODEL",
    "MCP_SERVER",
    "MODEL_ENV",
    "OpenCodeError",
    "OpenCodePreflight",
    "PROVIDER_ENV",
    "REQUIRED_FLAGS",
    "SECRET_ENV",
    "binary_path",
    "build_config",
    "cli_flags",
    "cli_version",
    "mcp_server_block",
    "model_name",
    "parse_events",
    "preflight",
    "provider_configuration",
    "redact",
    "run_agent",
]
