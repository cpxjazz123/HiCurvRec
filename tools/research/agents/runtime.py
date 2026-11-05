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
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

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
    "Every proposal is one block that starts with 'IDEA: <identifier>' and ends "
    "with 'END IDEA', with the fields the task asks for in between; the last "
    "record you append carries that block in full. "
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


class IdeaPayload(BaseModel):
    """The JSON an ideation agent answers with."""
    idea_id: str
    title: str
    motivation: str
    novelty: str
    key_change: str
    specification: str
    cost_estimate: str
    falsifiable_hypothesis: str
    risks: list[str] = Field(default_factory=list)
    behavior_signal: str = ""
    behavior_computation: str = ""
    leakage_boundary: str = ""
    hyperbolic_operator: str = ""
    hyperbolic_location: str = ""
    coupling_path: str = ""
    coupling_equation: str = ""
    observables: str = ""
    ablation_behavior_off: str = ""
    ablation_geometry_off: str = ""
    ablation_both_on: str = ""
    spec_equations: str = ""
    spec_inputs_outputs: str = ""
    safeguards: str = ""


class RankEntry(BaseModel):
    """One ranked proposal."""
    idea_id: str
    position: int
    reason: str


class RankPayload(BaseModel):
    """The JSON a reviewer answers with."""
    ranking: list[RankEntry] = Field(default_factory=list)
    objection: str = ""


class DerivationArgument(BaseModel):
    """One argument of the derived function, as the verifier feeds it."""
    name: str
    width: int = 1
    radius: float = 0.1
    scalar: float | None = None


class ComputationEntry(BaseModel):
    """One computation of the mechanism, stated in full."""
    name: str
    definition: str
    inputs: list[str] = Field(default_factory=list)
    output: str = ""
    differentiable: bool = False
    note: str = ""


class ContinuousSurface(BaseModel):
    """The differentiable surface the reference contract is built from."""
    name: str = ""
    callable: str = ""
    symbols: list[str] = Field(default_factory=list)
    args: list[DerivationArgument] = Field(default_factory=list)
    statements: list[str] = Field(default_factory=list)
    output: str = ""
    note: str = ""


class MathSpecPayload(BaseModel):
    """The JSON a Formula Agent answers with: the mechanism as mathematics.

    The specification covers EVERY computation the mechanism touches - the
    metric, the behaviour's entry point, the assignment, the gradient path, the
    reconstruction, the switches - not only a loss.
    """
    mechanism: str = ""
    computations: list[ComputationEntry] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    switches: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    continuous_surface: ContinuousSurface = Field(default_factory=ContinuousSurface)
    incomplete: str = ""
    missing: list[str] = Field(default_factory=list)


class CodingPayload(BaseModel):
    """The JSON a Coding Agent answers with: one module, implementing the spec."""
    module: str = ""
    notes: str = ""


class MathReviewPayload(BaseModel):
    """The JSON a Math Reviewer answers with."""
    verdict: str = ""
    findings: list[str] = Field(default_factory=list)
    reason: str = ""


class DerivationPayload(BaseModel):
    """The JSON an independent derivation answers with.

    The statements ARE the formula, in SymPy syntax; the code is the derivation's
    own torch implementation of the same statements. The deterministic verifier
    compares the two, and compares the derivations with each other.

    The SIGNATURE is not in the answer: the round fixes it, and a derivation
    that invents its own arguments is a different function (an agent once wrote
    the sampling radius into the field meant for a scalar value, and the whole
    comparison broke). The agent fills the statements, the output name and the
    code; the framework owns the arguments and the symbols.
    """
    statements: list[str] = Field(default_factory=list)
    output: str = ""
    code: str = ""
    excluded: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    args: list[DerivationArgument] = Field(default_factory=list)


def output_schema(role: str):
    """The structured-output schema of a role, when the provider supports it."""
    if role == "score":
        return RankPayload
    if role == "derive":
        return DerivationPayload
    if role == "formulate":
        return MathSpecPayload
    if role == "review_math":
        return MathReviewPayload
    if role == "code":
        return CodingPayload
    return IdeaPayload


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


#: The JSON keys an agent answers with, and the field labels the framework's
#: parser reads. The model speaks JSON; the runtime renders the canonical text
#: the framework parses, so the contract on disk does not change.
JSON_FIELDS = (
    ("idea_id", "IDEA"), ("title", "TITLE"), ("motivation", "MOTIVATION"),
    ("novelty", "NOVELTY"), ("key_change", "CHANGE"), ("specification", "SPECIFICATION"),
    ("cost_estimate", "COST"), ("falsifiable_hypothesis", "HYPOTHESIS"),
    ("risks", "RISKS"), ("behavior_signal", "BEHAVIOR SIGNAL"),
    ("behavior_computation", "BEHAVIOR COMPUTATION"),
    ("leakage_boundary", "LEAKAGE BOUNDARY"),
    ("hyperbolic_operator", "HYPERBOLIC OPERATOR"),
    ("hyperbolic_location", "HYPERBOLIC LOCATION"), ("coupling_path", "COUPLING PATH"),
    ("coupling_equation", "COUPLING EQUATION"), ("observables", "OBSERVABLES"),
    ("ablation_behavior_off", "ABLATION BEHAVIOR OFF"),
    ("ablation_geometry_off", "ABLATION GEOMETRY OFF"),
    ("ablation_both_on", "ABLATION BOTH ON"), ("spec_equations", "SPEC EQUATIONS"),
    ("spec_inputs_outputs", "SPEC INPUTS OUTPUTS"), ("safeguards", "SAFEGUARDS"),
)

IDEA_JSON_INSTRUCTIONS = (
    "Answer with ONE JSON object and nothing else: no prose, no markdown fence. "
    "Its keys are: " + ", ".join(key for key, _ in JSON_FIELDS) + ". "
    "Every key is a string except 'risks', which is a list of strings."
)

FORMULATION_JSON_INSTRUCTIONS = (
    "You WRITE the mathematics of a mechanism that has already been chosen. You "
    "do not choose the mechanism and you do not change it: your job is to state "
    "it so that an implementer has no freedom left.\n"
    "Answer with ONE JSON object and nothing else, with these keys:\n"
    '{"mechanism": "the mechanism in one paragraph",\n'
    ' "computations": [{"name": "...", "definition": "the formula, in symbols",\n'
    '                   "inputs": ["..."], "output": "...",\n'
    '                   "differentiable": true, "note": "..."}],\n'
    ' "inputs": ["..."], "outputs": ["..."], "parameters": ["..."],\n'
    ' "dimensions": {"batch": "B", "latent": 32, "codes": 256},\n'
    ' "gradient_rules": [{"operator": "the straight-through estimator",\n'
    '                     "rule": "identity in the backward: x + (x_q - x).detach()"},\n'
    '                    {"operator": "the argmin", "rule": "no gradient; it decides the token"}],\n'
    ' "training_integration": {"loss": "the total objective",\n'
    '                          "reconstruction": "the codeword the assignment chose",\n'
    '                          "optimizer": "adamw, the baseline schedule"},\n'
    ' "switches": ["the behaviour switch and what it does",\n'
    '              "the geometry switch and what it does"],\n'
    ' "excluded": ["the surfaces that are deliberately out of scope, and why"],\n'
    ' "continuous_surface": {"name": "...", "callable": "module:function",\n'
    '   "symbols": ["x0", "x1"],\n'
    '   "args": [{"name": "x", "width": 2}],\n'
    '   "statements": ["d = sqrt(x0**2 + x1**2)"],\n'
    '   "output": "d"},\n'
    ' "incomplete": "", "missing": []}\n'
    "Rules: EVERY computation the mechanism touches must appear in "
    "`computations` - the metric, the behaviour's entry point, the assignment, "
    "the gradient path (say exactly how the gradient reaches the encoder), the "
    "reconstruction, and each switch. The `continuous_surface` states the "
    "DIFFERENTIABLE part of the mechanism in SymPy statements, so that an "
    "independent rebuild is possible; its callable names the function an "
    "implementer must write, and its args the arguments it must take, in order. "
    "If the idea does not define something the mechanism needs, do NOT invent "
    "it: answer with \"incomplete\": \"SPECIFICATION_INCOMPLETE\" and list "
    "the missing definitions in `missing`.")

CODING_JSON_INSTRUCTIONS = (
    "You IMPLEMENT a frozen mathematical specification. The specification is "
    "the only authority: you may not reinterpret it, simplify it, or change a "
    "sign, a scale or a limit. If you believe the specification is wrong, say so "
    "in `notes` and implement it as written.\n"
    "Answer with ONE JSON object and nothing else:\n"
    '{"module": "import torch\\n\\ndef compute(...): ...",\n'
    ' "notes": "anything the verifier must know"}\n'
    "Rules: `module` is a complete Python module that defines the callable the "
    "specification's continuous_surface names, with exactly its arguments in "
    "its order, and it computes the specification's statements. Use torch only. "
    "Do not train, do not read a dataset, do not write files.")

MATH_REVIEW_JSON_INSTRUCTIONS = (
    "You REVIEW a mathematical specification against the idea it came from. "
    "You are not checking whether it can run: you are checking whether it "
    "FAITHFULLY expresses the idea, and whether the mathematics is correct, "
    "complete and unambiguous.\n"
    "Answer with ONE JSON object and nothing else:\n"
    '{"verdict": "PASS" or "REVISE", "findings": ["..."], "reason": "..."}\n'
    "Rules: REVISE if the specification changes the idea, if a computation is "
    "missing, if a formula is ambiguous or dimensionally inconsistent, if a "
    "sign, a scale or a limit is wrong, if the gradient path is not stated, or "
    "if the switches do not do what they claim. Say in every finding WHICH part "
    "is wrong and what it must be instead. PASS only if the specification is a "
    "faithful, complete and correct formulation of the idea.")

DERIVATION_JSON_INSTRUCTIONS = (
    "You DERIVE one formula, independently. Your reply is ONE JSON object and "
    "nothing else: no prose before it, no prose after it, no markdown fence. You are given a specification and "
    "nothing else: you do not see any other derivation, its codes or its test "
    "results, and you must not ask for them.\n"
    "Answer with ONE JSON object and nothing else:\n"
    '{"symbols": ["x0", "x1", "p0", "p1", "n0", "n1", "m"],\n'
    ' "args": [{"name": "x", "width": 2, "radius": 0.4},\n'
    '          {"name": "m", "scalar": 0.2}],\n'
    ' "statements": ["dp = sqrt((x0-p0)**2 + (x1-p1)**2)",\n'
    '                "L = log(1 + exp(-(dp - dn + m)))"],\n'
    ' "output": "L",\n'
    ' "code": "import torch\\n\\ndef compute(x, m):\\n    ...",\n'
    ' "excluded": ["the argmin assignment: not differentiable"]}\n'
    "Rules: the statements are SymPy expressions, one `name = expression` per "
    "line, evaluated in order, and `output` names the last one. `symbols` lists "
    "every free symbol in the order the arguments consume them (a tensor "
    "argument with width w consumes w symbols, one per column). `code` is a "
    "complete Python module defining `compute(**kwargs)` with exactly the "
    "argument names in `args`, returning a one-dimensional tensor, and it must "
    "compute the SAME formula as the statements - the verifier compares them. "
    "Use torch only. Do not train, do not touch a dataset, do not write files.")

RANK_JSON_INSTRUCTIONS = (
    "Answer with ONE JSON object and nothing else: no prose, no markdown fence. "
    "Its keys are 'ranking' (a list of objects with 'idea_id', 'position' and "
    "'reason') and 'objection' (a string)."
)


def extract_json(text: str) -> dict | None:
    """The JSON object an agent answered with, if it answered with one."""
    body = str(text or "")
    start = body.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(body)):
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(body[start:index + 1])
                    except ValueError:
                        break
                    if isinstance(payload, dict):
                        return payload
                    break
        start = body.find("{", start + 1)
    return None


def render_idea_json(payload: dict) -> str:
    """The canonical proposal block the framework's parser reads."""
    lines = []
    for key, label in JSON_FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(item) for item in value)
        lines.append(label + ": " + str(value).replace("\n", " ").strip())
    return "\n".join(lines) + "\nEND IDEA"


def render_rank_json(payload: dict) -> str:
    """The canonical ranking lines the scorer's parser reads."""
    lines = []
    for entry in payload.get("ranking") or []:
        if not isinstance(entry, dict):
            continue
        identifier = str(entry.get("idea_id", "")).strip()
        position = entry.get("position")
        reason = str(entry.get("reason", "")).strip()
        if identifier and position is not None:
            lines.append("RANK: " + identifier + " | position=" + str(position) +
                         " | reason=" + reason)
    if payload.get("objection"):
        lines.append("OBJECTION: " + str(payload["objection"]).replace("\n", " ").strip())
    return "\n".join(lines)


def extract_answer_block(text: str) -> str | None:
    """The proposal block an agent delivered, if it delivered one.

    The agents are asked for one block per proposal, ``IDEA: <id>`` ... ``END
    IDEA``. A block that was split over two progress records is still
    contiguous in the joined text, but a block that was cut short is not: the
    remainder is taken, because a truncated proposal with a readable identifier
    is worth more to the framework than a refusal to parse anything.
    """
    body = str(text or "")
    start = body.rfind("IDEA:")
    if start < 0:
        return None
    end = body.find("END IDEA", start)
    block = body[start:end + len("END IDEA")] if end >= 0 else body[start:].strip()
    return block.strip() or None


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
    """Run the agent and return what it answered, as text.

    With a structured output the answer is a model instance; it is rendered as
    JSON, which is the format the runtime renders into the block the framework
    reads.
    """
    from agents import Runner
    result = Runner.run_sync(agent, prompt, max_turns=max_turns)
    output = getattr(result, "final_output", "")
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, dict):
        return json.dumps(output)
    return str(output or "")


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
    instructions = SYSTEM_PROMPT
    if role == "derive":
        instructions = SYSTEM_PROMPT + "\n" + DERIVATION_JSON_INSTRUCTIONS
    elif role == "formulate":
        instructions = SYSTEM_PROMPT + "\n" + FORMULATION_JSON_INSTRUCTIONS
    elif role == "review_math":
        instructions = SYSTEM_PROMPT + "\n" + MATH_REVIEW_JSON_INSTRUCTIONS
    elif role == "code":
        instructions = SYSTEM_PROMPT + "\n" + CODING_JSON_INSTRUCTIONS
    elif role == "score":
        instructions = SYSTEM_PROMPT + "\n" + RANK_JSON_INSTRUCTIONS
    else:
        instructions = SYSTEM_PROMPT + "\n" + IDEA_JSON_INSTRUCTIONS
    schema = None
    if str(settings.AGENT_STRUCTURED_OUTPUT or "off").lower() == "on":
        schema = output_schema(role)
    try:
        agent = Agent(name=task_id, instructions=instructions,
                      tools=[function_tool(function) for function in functions],
                      model=build_model(model), output_type=schema)
    except Exception:                                        # noqa: BLE001
        # A provider that does not honour response_format=json_schema must not
        # cost the round: the agent is built without the schema and answers in
        # the text the runtime can still render.
        agent = Agent(name=task_id, instructions=instructions,
                      tools=[function_tool(function) for function in functions],
                      model=build_model(model))
    started = time.time()
    answer_text = ""
    failure = ""
    failure_trace = ""
    failure = ""
    failure_trace = ""
    structured_fallback = False
    timed_out = False
    # The budget is a wall clock: the CLI backend killed its subprocess when the
    # budget was spent, and the SDK backend must do the same, or an agent that
    # loops holds the whole round.
    budget = float(deadline_seconds or timeout or settings.AGENT_TIMEOUT_SECONDS)
    box: dict = {}

    def _turn() -> None:
        try:
            box["text"] = run_turn(agent, prompt)
        except Exception as exc:                               # noqa: BLE001
            box["error"] = exc

    worker_thread = threading.Thread(target=_turn, daemon=True)
    worker_thread.start()
    worker_thread.join(budget)
    if worker_thread.is_alive():
        timed_out = True
        failure = ("TIMEOUT: the agent did not finish within " + str(int(budget)) +
                   " seconds; what it saved is what it delivered")
    try:
        answer_text = box.get("text", "")
    except Exception as exc:                                   # noqa: BLE001
        if schema is not None:
            # The provider refused the schema (response_format=json_schema is not
            # universal): ask the same question without it, once.
            structured_fallback = True
            agent = Agent(name=task_id, instructions=instructions,
                          tools=[function_tool(function) for function in functions],
                          model=build_model(model))
            try:
                answer_text = run_turn(agent, prompt)
            except Exception as retry_exc:                       # noqa: BLE001
                import traceback
                failure = type(retry_exc).__name__ + ": " + str(retry_exc)[:300]
                failure_trace = traceback.format_exc()[-1500:]
        else:
            import traceback
            failure = type(exc).__name__ + ": " + str(exc)[:300]
            failure_trace = traceback.format_exc()[-1500:]
    if timed_out:
        answer_text = ""
    if not output.is_file():
        # A read-only role has no write tool, and an agent that answered in the
        # chat instead of the file still delivered its answer: the records are
        # what the framework reads, so they are what the answer file carries.
        records = progress_server.read_records(workspace / runner.PROGRESS_FILE_NAME)
        joined = "\n\n".join(str(record.get("content", "")) for record in records
                             if str(record.get("content", "")).strip())
        text = joined or answer_text
        payload = extract_json(answer_text) or extract_json(text)
        block = None
        if isinstance(payload, dict):
            if payload.get("idea_id"):
                block = render_idea_json(payload)
            elif payload.get("ranking"):
                block = render_rank_json(payload)
            elif payload.get("verdict"):
                # a review verdict: the canonical JSON on disk
                block = json.dumps({key: payload.get(key) for key in
                                    ("verdict", "findings", "reason")
                                    if payload.get(key) is not None},
                                   indent=1, ensure_ascii=False)
            elif payload.get("computations"):
                # a formulated specification: the canonical JSON on disk is the
                # one form that cannot be mangled by the progress records
                block = json.dumps({key: payload.get(key) for key in
                                    ("mechanism", "computations", "inputs", "outputs",
                                     "parameters", "dimensions", "gradient_rules",
                                     "training_integration", "switches", "excluded",
                                     "continuous_surface", "incomplete", "missing")
                                    if payload.get(key) is not None},
                                   indent=1, ensure_ascii=False)
            elif payload.get("module"):
                # an implementation: the module is the answer
                block = str(payload.get("module"))
            elif payload.get("statements"):
                # A derivation is a formula and the code that computes it: the
                # canonical JSON on disk is the one form that cannot be mangled
                # by joining progress records (which inserts separators inside
                # a string and used to turn a reshape(-1, 2) into a syntax error).
                block = json.dumps({key: payload.get(key) for key in
                                    ("symbols", "args", "statements", "output",
                                     "code", "excluded") if payload.get(key)},
                                   indent=1, ensure_ascii=False)
        block = block or extract_answer_block(text) or extract_answer_block(answer_text)
        if block and not echoes_the_prompt(block, prompt):
            output.write_text(block + "\n", encoding="utf-8")
        elif text and not echoes_the_prompt(text, prompt):
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
        "structured_output": bool(schema), "structured_output_fallback": structured_fallback,
        "timed_out": timed_out, "budget_seconds": budget,
        "failure_trace": failure_trace,
        "progress_file": str(workspace / runner.PROGRESS_FILE_NAME),
        "answer": str(output),
    }
    receipt_path = workspace / "execution.json"
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    if failure and not output.is_file():
        raise runner.AgentFailure("timeout" if timed_out else "AGENT_CRASHED",
                                  str(receipt_path), failure)
    return receipt_path
