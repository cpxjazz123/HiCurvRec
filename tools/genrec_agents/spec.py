"""First-party specifications for the agent harnesses (the native replacement for the
third-party framework's ``core.spec``).

These dataclasses used to come from the third-party framework's spec module. They are re-declared here so that
every active module imports nothing outside this repository: the fields, their meaning and their
defaults are the same, only the owner changed.

    ChannelSpec   which backend answers a task, and with which model
    BenchPrompts  the wording of the four prompt parts a benchmark hands its harness
    BenchSpec     the benchmark itself: its name, its prompts, its channels, its split rules
    RunView       one run's outcome, as the reports read it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

__all__ = ["ChannelSpec", "BenchPrompts", "BenchSpec", "RunView"]


@dataclass
class ChannelSpec:
    """One backend that can answer a task: its kind, its model and its budget."""

    kind: str
    model: str
    effort: str = ""
    steps: int = 0
    timeout: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "model": self.model, "effort": self.effort,
                "steps": self.steps, "timeout": self.timeout}


@dataclass
class BenchPrompts:
    """The wording the benchmark hands to its harness, part by part."""

    work: str = "task"
    domain_word: str = "domain"
    inv_levers: str = ""
    dom_levers: str = ""
    yardstick: str = ""
    deliverable_word: str = "deliverable"
    domain_guidance: str = ""
    selection_checks: str = ""
    selection_lead: str = ""
    selection_conformance: str = ""
    prompt_pack: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"work": self.work, "domain_word": self.domain_word,
                "inv_levers": self.inv_levers, "dom_levers": self.dom_levers,
                "yardstick": self.yardstick, "deliverable_word": self.deliverable_word,
                "domain_guidance": self.domain_guidance,
                "selection_checks": self.selection_checks,
                "selection_lead": self.selection_lead,
                "selection_conformance": self.selection_conformance,
                "prompt_pack": self.prompt_pack}


@dataclass
class BenchSpec:
    """The benchmark: its name, where its runs live, how its tasks are split, its channels."""

    name: str
    runs_root: Path
    install_mode: str = "copy"
    prompts: BenchPrompts = field(default_factory=BenchPrompts)
    channels: Dict[str, ChannelSpec] = field(default_factory=dict)
    domain_word: str = "domain"
    leak_globs: Tuple[str, ...] = ()
    allow_subagents: bool = False
    reserved_scripts: FrozenSet[str] = frozenset()
    proposer_deliverables: str = "contents"
    state_deliverable: bool = False
    obs_trunc: int = 50_000

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "runs_root": str(self.runs_root),
                "install_mode": self.install_mode, "prompts": self.prompts.as_dict(),
                "channels": {k: v.as_dict() for k, v in self.channels.items()},
                "domain_word": self.domain_word, "leak_globs": list(self.leak_globs),
                "allow_subagents": self.allow_subagents,
                "reserved_scripts": sorted(self.reserved_scripts),
                "proposer_deliverables": self.proposer_deliverables,
                "state_deliverable": self.state_deliverable, "obs_trunc": self.obs_trunc}


@dataclass
class RunView:
    """One run's outcome, as the reports and the library read it."""

    label: str
    run_id: str
    run_dir: Path
    ok: bool
    delivered: bool = False
    cost_usd: float = 0.0
    wall_s: float = 0.0
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "run_id": self.run_id, "run_dir": str(self.run_dir),
                "ok": self.ok, "delivered": self.delivered, "cost_usd": self.cost_usd,
                "wall_s": self.wall_s, "error": self.error}
