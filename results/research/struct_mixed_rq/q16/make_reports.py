"""A fresh both-on report and the two inactive states, in a new timestamped
directory; nothing from q15 is overwritten."""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix"))

import torch
torch.set_num_threads(1)

import probe_q9 as P
import structure as S
import train_rqvae as T

STAMP = datetime.now().strftime("%Y%m%dT%H%M%S")
OUT = ROOT / "results/research/struct_mixed_rq/q16" / ("reports_" + STAMP)
OUT.mkdir(parents=True, exist_ok=False)

both = P.run(OUT / "both_on")

saved = T.BEHAVIOUR_ON
T.BEHAVIOUR_ON = False
try:
    off = P.run(OUT / "behaviour_off")
finally:
    T.BEHAVIOUR_ON = saved

saved_sample = S.sample_triples
S.sample_triples = lambda graph, size, rng: []
try:
    empty = P.run(OUT / "empty_triples")
finally:
    S.sample_triples = saved_sample

def edge(report):
    ident = report["identity"]
    return {
        "structure_forward": ident["structure_forward"],
        "structure_zero_reason": ident["structure_zero_reason"],
        "triples_sampled": ident["triples_sampled"],
        "triples_accepted": ident["triples_accepted"],
        "z_struct_rows": ident["z_struct_rows"],
        "latent_z_struct": (report["latent"]["structure"] or {}).get("z_struct"),
    }

(OUT / "boundaries.json").write_text(json.dumps({
    "both_on": edge(both), "behaviour_off": edge(off), "empty_triples": edge(empty),
}, indent=1))
print(json.dumps({k: edge(v) for k, v in
                 {"both_on": both, "behaviour_off": off, "empty_triples": empty}.items()}, indent=1))
