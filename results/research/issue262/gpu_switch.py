"""Issue #262: GPU read-only switch replay diagnostic for the CPU/GPU export gap."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
Q = OUT / "_raw" / "q" / "102.npz"
ARM = "TIGER_module_budget_prefix_rq"


def main() -> None:
    final = sorted((ROOT / "results/stage3_T5Train" / ARM).glob("**/test_final.json"))[0]
    fd = json.loads(final.read_text())
    checkpoint = Path(fd["stage2_checkpoint"])
    binding = Path(fd["archive_binding"])
    tree = Path(json.loads(binding.read_text())["source_tree"])
    with np.load(Q, allow_pickle=False) as blob:
        sid3 = blob["sid3"].astype(np.int64)
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae",
                                  "assignment_floor", "validation_entry"):
            del sys.modules[name]
    sys.path.insert(0, str(tree))
    training = importlib.import_module("train_rqvae")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).cuda()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    embeddings = torch.from_numpy(np.load(ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy")).cuda()
    def export():
        with torch.inference_mode():
            return model.get_indices(embeddings, infer_use_sk=False).cpu().numpy().astype(np.int64)
    result = {}
    original = export()
    result["no_intervention"] = {
        "exact_sid_match": bool(np.array_equal(original[:, :3], sid3)),
        "changed_fraction": float((original[:, :3] != sid3).any(axis=1).mean()),
    }
    was = model.behavior_on
    model.behavior_on = False
    changed = export()
    model.behavior_on = was
    result["behavior_off"] = {"exact_sid_match": bool(np.array_equal(changed[:, :3], sid3)),
                              "changed_fraction": float((changed[:, :3] != sid3).any(axis=1).mean())}
    model.curvature_on = False
    changed = export()
    result["curvature_off"] = {"exact_sid_match": bool(np.array_equal(changed[:, :3], sid3)),
                               "changed_fraction": float((changed[:, :3] != sid3).any(axis=1).mean())}
    model.curvature_on = True
    out = {"schema": "genrec.issue262.gpu_switch/1", "arm": ARM,
           "device": torch.cuda.get_device_name(0),
           "result": result,
           "scope": "read-only single-GPU replay; no training, no optimizer, no SID write"}
    (OUT / "GPU_SWITCH_102.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
