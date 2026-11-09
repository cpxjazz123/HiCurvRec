"""Persist the both-on report and the two inactive-graph boundaries from the
current producer, by the real report path (no mocks of the model or the loss)."""
import json
import sys
from pathlib import Path

ROOT = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix"))

import torch
torch.set_num_threads(1)

import probe_q9 as P
import structure as S
import train_rqvae as T

OUT = ROOT / "results/research/struct_mixed_rq/q15"

both = P.run(OUT / "report_both_on")

saved = T.BEHAVIOUR_ON
T.BEHAVIOUR_ON = False
try:
    off = P.run(OUT / "report_behaviour_off")
finally:
    T.BEHAVIOUR_ON = saved

saved_sample = S.sample_triples
S.sample_triples = lambda graph, size, rng: []
try:
    empty = P.run(OUT / "report_empty_triples")
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
        "state_status": (report["states"]["structure"] or {}).get("status"),
        "losses": report["losses"],
    }

(OUT / "boundaries.json").write_text(json.dumps({
    "both_on": edge(both),
    "behaviour_off": edge(off),
    "empty_triples": edge(empty),
}, indent=1))
print(json.dumps(edge(off), indent=1))
print(json.dumps(edge(empty), indent=1))
