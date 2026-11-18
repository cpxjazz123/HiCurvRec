"""Try to collapse the codebook on purpose, and record which layer refuses.

The claim under test is "no Stage2 mechanism collapses the codebook". A claim like that is
only worth what its adversaries are worth, so this probe drives five hostile quantisers
through the *installed* guard (a patched copy of the frozen baseline, imported in a fresh
interpreter) and prints what happened to each.

The five are chosen to cover the ways a mechanism can defeat a guard:

    classic_collapse    the collapse that killed 193/256 codewords in round 1: plain argmin
                        on a cost matrix where a few codewords are cheap
    bypass_indices      the mechanism assigns in its own `forward`, never calling `_indices`
    duplicate_codebook  every codeword is used, but two of them are the same vector
    encoder_collapse    the encoder maps every item to the same vector, so the assignment is
                        decided entirely by tie-breaking
    frozen_codebook     control: a codebook that never moves while the encoder trains; this
                        is *not* a collapse and must not be refused

Each verdict is one of
    REPAIRED    the assignment was changed to satisfy the floor (and the run continued)
    REFUSED     the guard raised, so the run dies at that step
    SURVIVED    the run continued and produced an artifact
    ESCAPED     the run continued, and the exported artifact is degenerate

`ESCAPED` is the interesting column: it names exactly what the guard does not cover.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
SCRATCH = REPO / "results" / "research" / "qda" / "anticollapse_check" / "adversarial"

PROBE = r'''
import json, sys, torch
from model.layers import VQLayer

mode = sys.argv[1]
layer = VQLayer(256, 32)
report = {"mode": mode}

def used(index):
    return int(torch.unique(index).numel())

def sid_from(index):
    """The identifier this assignment would export, read back under the artifact rule."""
    import json, os, tempfile
    from qda_artifact_probe import audit_sid
    path = os.path.join(tempfile.mkdtemp(dir=os.getcwd()), "item_sids.json")
    codes = {str(i): [int(v), int(v), int(v)] for i, v in enumerate(index.tolist())}
    open(path, "w").write(json.dumps(codes))
    return {k: audit_sid(path)[k] for k in ("status", "detail")}

def collapse_cost(items=4096, live=8):
    cost = torch.rand(items, 256) + 10.0
    cost[:, :live] = torch.rand(items, live)
    return cost

if mode == "classic_collapse":
    cost = collapse_cost()
    report["before"] = used(torch.argmin(cost, dim=-1))
    index = layer._indices(cost, False)
    report["after"] = used(index)
    report["verdict"] = "REPAIRED" if report["after"] > report["before"] else "SURVIVED"

elif mode == "bypass_indices":
    class Own(VQLayer):
        def forward(self, x, infer_use_sk=False):
            latent = x.view(-1, self.dim)
            index = torch.argmin(self._distances(latent), dim=-1)   # not _indices
            x_q = torch.nn.functional.embedding(index, self.get_code_embs()).view(x.shape)
            return x + (x_q - x).detach(), (torch.zeros((), dtype=x.dtype),)*2, 0, index
    hostile = Own(256, 32)
    x = torch.randn(4096, 32)
    cost = collapse_cost()
    hostile._distances = lambda latent: cost                      # the collapsed cost
    with torch.no_grad():
        _, _, _, index = hostile(x)
    report["after"] = used(index)
    report["verdict"] = "REPAIRED" if report["after"] > 8 else "SURVIVED"
    report["artifact"] = sid_from(index)

elif mode == "duplicate_codebook":
    codes = layer.get_code_embs()
    with torch.no_grad():
        codes[1] = codes[0]
    layer.eval()
    try:
        with torch.no_grad():
            layer(torch.randn(4096, 4) if layer.dim == 4 else torch.randn(4096, layer.dim))
    except Exception as error:
        report["verdict"] = "REFUSED"
        report["error"] = type(error).__name__ + ": " + str(error)[:90]
    else:
        report["verdict"] = "SURVIVED"

elif mode == "encoder_collapse":
    x = torch.zeros(4096, 32)                                     # every item identical
    with torch.no_grad():
        _, _, _, index = layer(x)
    report["used"] = used(index)
    report["verdict"] = "SURVIVED" if report["used"] == 256 else "REFUSED"
    report["note"] = "all 256 codes used, all of them by tie-breaking"
    report["artifact"] = sid_from(index)

elif mode == "frozen_codebook":
    before = layer.get_code_embs().detach().clone()
    with torch.no_grad():
        layer(torch.randn(4096, 32))
    after = layer.get_code_embs().detach()
    report["moved"] = float((after - before).abs().max())
    report["verdict"] = "SURVIVED"

print("VERDICT " + json.dumps(report))
'''


def run() -> dict:
    SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    shutil.copytree(BASELINE, SCRATCH, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs", "results"))
    sys.path.insert(0, str(REPO / "tools" / "research"))
    from stage2.anticollapse import apply_floor

    apply_floor(SCRATCH)
    shutil.copy2(REPO / "tools" / "research" / "stage2" / "anticollapse" / "artifact.py",
                 SCRATCH / "qda_artifact_probe.py")
    rows = []
    for mode in ("classic_collapse", "bypass_indices", "duplicate_codebook",
                 "encoder_collapse", "frozen_codebook"):
        done = subprocess.run([sys.executable, "-c", PROBE, mode], cwd=str(SCRATCH),
                              capture_output=True, text=True, timeout=900)
        line = [ln for ln in done.stdout.splitlines() if ln.startswith("VERDICT ")]
        if done.returncode != 0 or not line:
            rows.append({"mode": mode, "verdict": "ERROR",
                         "error": (done.stderr or done.stdout)[-300:]})
            continue
        rows.append(json.loads(line[-1][len("VERDICT "):]))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return {"rows": rows}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=1))
    width = max(len(row["mode"]) for row in result["rows"])
    for row in result["rows"]:
        extra = row.get("error") or row.get("note") or (
            "used %s -> %s" % (row.get("before"), row.get("after"))
            if "before" in row else "")
        print("%-*s  %-9s  %s" % (width, row["mode"], row["verdict"], extra))
