"""Wrap the shipped pretrained head into the Teacher contract (issue #221).

`pretrain_head.py` saved a bare state_dict, which no loader can validate: the
class counts, the hidden width and the vocabulary all lived outside the file.
This converts it once, then proves the conversion is lossless by comparing the
logits of the legacy head and the reloaded Teacher on a fixed batch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"))

import teacher  # noqa: E402

LEGACY = HERE / "head_pretrained_directed.pt"
META = HERE / "head_pretrained_directed.json"
OUT = HERE / "head_pretrained_directed.teacher.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    state = torch.load(LEGACY, map_location="cpu")
    meta = json.loads(META.read_text())
    n_subtypes, n_orbits = int(meta["n_subtypes"]), int(meta["n_orbits"])
    dim = int(state["phi.0.weight"].shape[1])
    hidden = int(state["phi.0.weight"].shape[0])
    config = {"dim": dim, "hidden": hidden, "n_subtypes": n_subtypes,
              "n_orbits": n_orbits, "interaction": True}

    legacy_head = teacher.build_head(config)
    legacy_head.load_state_dict(state, strict=True)
    payload = teacher.wrap_state_dict(
        state, config=config,
        vocab={"mode": "directed", "n_subtypes": n_subtypes, "n_orbits": n_orbits},
        extra={"source": str(LEGACY.relative_to(REPO)),
               "source_sha256": sha256(LEGACY),
               "heldout_subtype_macro_f1": meta["heldout"]["subtype"]["macro_f1"],
               "heldout_orbit_macro_f1": meta["heldout"]["role"]["macro_over_families"]})
    torch.save(payload, OUT)

    reloaded, loaded = teacher.load_teacher(
        OUT, expect_vocab={"mode": "directed", "n_subtypes": n_subtypes, "n_orbits": n_orbits})
    x = torch.randn(6, 4, dim, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        before_type, before_role = legacy_head(x)
        after_type, after_role = reloaded(x)
    assert torch.equal(before_type, after_type), "subtype logits changed through the round trip"
    assert torch.equal(before_role, after_role), "role logits changed through the round trip"
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())

    record = {"teacher": str(OUT.relative_to(REPO)), "sha256": sha256(OUT),
              "source": str(LEGACY.relative_to(REPO)), "source_sha256": sha256(LEGACY),
              "config": loaded["config"], "vocab": loaded["vocab"],
              "predictions_identical": True,
              "frozen": True,
              "heldout": {"subtype_macro_f1": meta["heldout"]["subtype"]["macro_f1"],
                          "orbit_macro_over_families": meta["heldout"]["role"]["macro_over_families"]}}
    (HERE / "head_pretrained_directed.teacher.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
