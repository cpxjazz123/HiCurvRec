"""The G_H1 probe contract: one lightweight probe per genuinely new arm (issue #109).

G_H1 is the history-conditioned level-1 predictability gain: the share of the frequency
prior's cross entropy that the history-conditioned probe removes,

    G_H1 = (CE_L1(train-frequency prior) - CE_L1(history GRU)) / CE_L1(train-frequency prior)

It is the issue's primary Stage-3-oriented signal, and the issue is exact about how it may
be produced: one fresh small GRU run at seed 42 on train data only, one probe per new SID, a
fixed budget, a fixed user-disjoint split, a frequency prior fitted only on the probe's train
users, and no early stopping or sweep on Stage-3 feedback. Nothing here may read a
validation or test label, and a probe that has already been run for an export may not be run
again under another name.

The probe's architecture and training live in ``scripts/history_probe.py`` and
``scripts/history_probe_runner.py``; this module is the contract around them: what a request
must look like, what a result must carry, how leakage is checked, and how a second probe for
the same export is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The only seed a probe may use.
PROBE_SEED = 42

#: The probe's architecture and budget, which issue #109 freezes.
FIXED_BUDGET: Dict[str, Any] = {
    "seq_len": 20,
    "token_embed_dim": 128,
    "gru_hidden": 128,
    "gru_layers": 1,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 512,
    "epochs": 6,
    "dev_user_fraction": 0.10,
    "grad_clip": 5.0,
    "label_smoothing": 0.0,
}

#: The version of the probe implementation, recorded with every result.
PROBE_VERSION = "history_probe/fix48"

#: Where the probe registry is persisted.
DEFAULT_REGISTRY = REPO_ROOT / "results" / "harness3_runs" / "probe_registry.json"


class ProbeRefusal(RuntimeError):
    """A prohibited probe was refused. The reason says which rule and why."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"[{rule}] {detail}")
        self.rule = rule
        self.detail = detail


@dataclass
class ProbeRequest:
    """What a probe run must declare."""

    mechanism_id: str
    sid_path: Path
    seed: int = PROBE_SEED
    budget: Dict[str, Any] = field(default_factory=lambda: dict(FIXED_BUDGET))
    probe_version: str = PROBE_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "sid_path": str(self.sid_path),
            "seed": self.seed,
            "budget": dict(self.budget),
            "probe_version": self.probe_version,
        }


def sha256_file(path: Path) -> str:
    """The digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_request(request: ProbeRequest) -> None:
    """Refuse a request that departs from the frozen seed, budget or probe version."""
    if request.seed != PROBE_SEED:
        raise ProbeRefusal(
            "single_seed",
            f"a probe runs at seed {PROBE_SEED} and this one asks for {request.seed}; issue "
            "#109 allows one probe per arm and forbids a seed search",
        )
    for key, value in FIXED_BUDGET.items():
        if request.budget.get(key) != value:
            raise ProbeRefusal(
                "fixed_budget",
                f"the probe's {key} is frozen at {value!r} and this request asks for "
                f"{request.budget.get(key)!r}; issue #109 freezes the budget",
            )
    if request.probe_version != PROBE_VERSION:
        raise ProbeRefusal(
            "probe_version",
            f"the probe version is frozen at {PROBE_VERSION!r} and this request names "
            f"{request.probe_version!r}",
        )
    if not request.mechanism_id:
        raise ProbeRefusal("missing_mechanism", "a probe must name the arm it measures")
    if not Path(request.sid_path).is_file():
        raise ProbeRefusal(
            "missing_export",
            f"the export the probe would read is missing: {request.sid_path}",
        )


def split_identity(train_rows: Sequence[int], dev_rows: Sequence[int]) -> Dict[str, Any]:
    """The identity of a split: its sizes and the digests of its row ids."""
    return {
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_row_id_sha256": _digest(train_rows),
        "dev_row_id_sha256": _digest(dev_rows),
    }


def _digest(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_leakage(train_users: Sequence[int], dev_users: Sequence[int]) -> Dict[str, Any]:
    """The leakage check: the two sides of the split must share no user.

    The frequency prior is fitted on the train users and the probe is scored on the dev
    users, so a user on both sides would make G_H1 a measurement of memorisation.
    """
    train = set(int(user) for user in train_users)
    dev = set(int(user) for user in dev_users)
    overlap = sorted(train & dev)
    if overlap:
        raise ProbeRefusal(
            "split_leakage",
            f"{len(overlap)} user(s) appear on both sides of the split, the first being "
            f"{overlap[0]}; the frequency prior and the probe's score would then share a user",
        )
    return {"train_users": len(train), "dev_users": len(dev), "overlap": 0}


# ------------------------------------------------------------------ the probe registry
def load_registry(path: Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    """The probe registry: the exports a probe has already been run for."""
    path = Path(path)
    if not path.is_file():
        return {"version": 1, "probes": []}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {"version": document.get("version", 1), "probes": document.get("probes") or []}


def save_registry(registry: Dict[str, Any], path: Path = DEFAULT_REGISTRY) -> str:
    """Write the probe registry atomically and return the digest of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(registry, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return sha256_file(path)


def check_not_probed(
    request: ProbeRequest, registry: Dict[str, Any], sid_sha256: str = ""
) -> None:
    """Refuse a second probe for an export, whatever the arm is called."""
    digest = sid_sha256 or sha256_file(Path(request.sid_path))
    for entry in registry.get("probes") or ():
        if entry.get("sid_sha256") == digest:
            raise ProbeRefusal(
                "no_repeated_probe",
                f"a probe was already run for this export as {entry.get('mechanism_id')!r}; "
                "issue #109 allows one lightweight probe per genuinely new arm, and a probe "
                "under a renamed arm is the same probe",
            )
        if entry.get("mechanism_id") == request.mechanism_id:
            raise ProbeRefusal(
                "no_repeated_probe",
                f"a probe was already run for {request.mechanism_id!r}; issue #109 allows one "
                "lightweight probe per genuinely new arm",
            )


# ------------------------------------------------------------------ running the probe
def run_probe(
    request: ProbeRequest,
    *,
    runner: Optional[Callable[[Any, Any, Any], List[Dict[str, Any]]]] = None,
    registry: Optional[Dict[str, Any]] = None,
    registry_path: Path = DEFAULT_REGISTRY,
    device: str = "cpu",
    sid_loader: Optional[Callable[[Path], Any]] = None,
) -> Dict[str, Any]:
    """Run the probe and return the record the evidence chain keeps.

    The runner is injectable so the contract can be tested without a GPU; the default is the
    repository's own runner, which is the implementation the archived probe scores came from.
    """
    check_request(request)
    sid_sha256 = sha256_file(Path(request.sid_path))
    registry = registry if registry is not None else load_registry(registry_path)
    check_not_probed(request, registry, sid_sha256)

    default_runner = runner is None
    if default_runner:
        runner = _default_runner()

    # torch is only needed to place the probe on a device; the contract itself, and its
    # tests, must not depend on it
    try:
        import torch

        device_object = torch.device(device) if isinstance(device, str) else device
    except ImportError:  # pragma: no cover - the contract is usable without torch
        device_object = device

    if sid_loader is None:
        import numpy as np

        sid_loader = np.load
    sid = sid_loader(Path(request.sid_path))
    # the runner's configuration object needs the probe's own module, which needs numpy;
    # an injected runner brings its own configuration
    config = _probe_config(request.budget) if default_runner else request.budget
    results = runner(sid, config, device_object, seeds=(request.seed,))
    if not results:
        raise ProbeRefusal("no_probe_result", "the probe produced no result")
    metrics = results[0]

    prior_ce = metrics.get("prior_ce_l1")
    probe_ce = metrics.get("probe_ce_l1")
    # the issue defines G_H1 as the RELATIVE gain: the share of the prior's cross entropy
    # that the history-conditioned probe removes. The runner reports the absolute gain.
    relative_gain = (
        float((prior_ce - probe_ce) / prior_ce)
        if prior_ce not in (None, 0) and probe_ce is not None
        else None
    )
    record = {
        "mechanism_id": request.mechanism_id,
        "sid_path": str(request.sid_path),
        "sid_sha256": sid_sha256,
        "seed": request.seed,
        "probe_version": request.probe_version,
        "budget": dict(request.budget),
        "prior_ce_l1": metrics.get("prior_ce_l1"),
        "probe_ce_l1": metrics.get("probe_ce_l1"),
        "g_h1": relative_gain,
        "gain_l1_absolute": metrics.get("gain_l1"),
        "top1_acc_l1": metrics.get("probe_top1_acc_l1"),
        "top5_acc_l1": metrics.get("probe_top5_acc_l1"),
        "top20_acc_l1": metrics.get("probe_top20_acc_l1"),
        "prior_top1_acc_l1": metrics.get("prior_top1_acc_l1"),
        "train_rows": metrics.get("train_rows"),
        "dev_rows": metrics.get("dev_rows"),
        "split": {
            "train_row_id_sha256": metrics.get("train_row_id_sha256"),
            "dev_row_id_sha256": metrics.get("dev_row_id_sha256"),
            "train_rows": metrics.get("train_rows"),
            "dev_rows": metrics.get("dev_rows"),
        },
        "fit_seconds": metrics.get("fit_seconds"),
        "level_diagnostics": {
            key: metrics.get(key)
            for key in (
                "gain_l2",
                "gain_l3",
                "probe_ce_l2",
                "probe_ce_l3",
                "probe_top1_acc_l2",
                "probe_top1_acc_l3",
            )
            if key in metrics
        },
        "note": (
            "the level-1 number is the primary one; L2 and L3 see the earlier target tokens, so "
            "their numbers are diagnostics and are not interchangeable with it"
        ),
    }
    registry = {
        "version": registry.get("version", 1),
        "probes": list(registry.get("probes") or ())
        + [
            {
                "mechanism_id": request.mechanism_id,
                "sid_sha256": sid_sha256,
                "seed": request.seed,
                "probe_version": request.probe_version,
                "train_row_id_sha256": metrics.get("train_row_id_sha256"),
                "dev_row_id_sha256": metrics.get("dev_row_id_sha256"),
            }
        ],
    }
    save_registry(registry, registry_path)
    return record


def _probe_config(budget: Dict[str, Any]):
    """The runner's configuration object, built from the frozen budget."""
    import sys

    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from history_probe import ProbeConfig

    return ProbeConfig(**{key: budget[key] for key in budget if key in ProbeConfig.__annotations__})


def _default_runner():
    """The repository's own probe runner."""
    import sys

    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from history_probe_runner import run_arm

    return run_arm


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one probe and print its record as JSON."""
    import argparse

    parser = argparse.ArgumentParser(prog="tools.genrec_agents.history_probe")
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    args = parser.parse_args(list(argv) if argv is not None else None)
    request = ProbeRequest(mechanism_id=args.mechanism_id, sid_path=Path(args.sid))
    try:
        record = run_probe(request, registry_path=Path(args.registry), device=args.device)
    except ProbeRefusal as refusal:
        print(json.dumps({"ok": False, "rule": refusal.rule, "detail": refusal.detail}, indent=2))
        return 1
    print(json.dumps(record, indent=2, default=str))
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REGISTRY",
    "FIXED_BUDGET",
    "PROBE_SEED",
    "PROBE_VERSION",
    "ProbeRefusal",
    "ProbeRequest",
    "check_leakage",
    "check_not_probed",
    "check_request",
    "load_registry",
    "run_probe",
    "save_registry",
    "split_identity",
]
