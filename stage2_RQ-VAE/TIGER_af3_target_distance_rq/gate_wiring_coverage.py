"""Gates 5 (wiring) and 6 (coverage) of issue #155, without any training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import target_distance as td  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402

BATCHES, SIZE = 8, 1024


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    table = json.loads((root / "results" / "research" / "issue155" / "af3_edges.json").read_text())
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_of_node = {index: int(item) for index, item in enumerate(raw["item_ids"])}
    all_ids = sorted(item_of_node.values())
    pairs = []
    for (tail, head), u in zip(table["edges"], table["u"]):
        left, right = item_of_node.get(int(tail)), item_of_node.get(int(head))
        if left is not None and right is not None:
            pairs.append((left, right, float(u)))
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[torch.from_numpy(np.asarray(all_ids[:4096]))])
    model.train()
    generator = torch.Generator().manual_seed(15530)
    census = {"batches": BATCHES, "with_both": 0, "high": [], "low": []}
    for _ in range(BATCHES):
        pick = torch.randperm(len(all_ids), generator=generator)[:SIZE].numpy()
        batch_ids = np.asarray(all_ids)[pick]
        position = {int(item): index for index, item in enumerate(batch_ids)}
        high = low = 0
        for left, right, u in pairs:
            if left in position and right in position:
                if u >= 1.0:
                    high += 1
                elif u <= 0.0:
                    low += 1
        census["high"].append(high)
        census["low"].append(low)
        census["with_both"] += int(high > 0 and low > 0)

    probe = rows[torch.from_numpy(np.asarray(all_ids[:1024]))]
    td.STRUCTURE_OFF = True
    model.eval()
    model(probe)
    off_tokens = torch.stack(model.rq.last_codes, dim=1).clone()
    td.STRUCTURE_OFF = False
    model(probe)
    on_tokens = torch.stack(model.rq.last_codes, dim=1).clone()
    report = {
        "census": census,
        "coverage_fraction": census["with_both"] / BATCHES,
        "gate6_pass": bool(census["with_both"] / BATCHES >= 0.95),
        # the loss never touches the codes; the switch changes nothing but the two terms,
        # so the tokens must be identical with the supervision off and on
        "tokens_identical_off_on": bool(torch.equal(off_tokens, on_tokens)),
    }
    target = root / "results" / "research" / "issue155" / "gate_wiring.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
