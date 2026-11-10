"""Generate the 64-scalar statements for mixed_sq and compare against torch.

The statements are written by this script, stored in the candidate contract,
and checked here at the q33 tolerances.  Run with the Stage2 interpreter.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/fs04/ar57/wenyu/GeneRec")
CANDIDATE = ROOT / "stage2_RQ-VAE" / "TIGER_struct_mixed_rq_ste_fix"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CANDIDATE))

from tools.research.stage2.reference import build_reference  # noqa: E402
import mixed_geometry as MG  # noqa: E402

E = 16
H = 16
TAU = 1e-3


def build_calls():
    symbols = [f"L_{i}" for i in range(32)] + [f"R_{i}" for i in range(32)]
    statements = ["E = " + " + ".join(f"(L_{i} - R_{i})**2" for i in range(E))]
    statements.append("rl2 = " + " + ".join(f"L_{i}**2" for i in range(E, 32)))
    statements.append("rr2 = " + " + ".join(f"R_{i}**2" for i in range(E, 32)))
    statements.append("rl = sqrt(rl2)")
    statements.append("rr = sqrt(rr2)")
    for i in range(E, 32):
        statements.append(f"bl_{i} = tanh(rl) * L_{i} / rl")
        statements.append(f"br_{i} = tanh(rr) * R_{i} / rr")
    statements.append("num = " + " + ".join(f"(bl_{i} - br_{i})**2" for i in range(E, 32)))
    statements.append("bl2 = " + " + ".join(f"bl_{i}**2" for i in range(E, 32)))
    statements.append("br2 = " + " + ".join(f"br_{i}**2" for i in range(E, 32)))
    statements.append("t = num / ((1 - bl2) * (1 - br2))")
    series = statements + ["series = t - t**2/3 + 8*t**3/45", "d = E + series"]
    asym = statements + ["asym = asinh(sqrt(t))**2", "d = E + asym"]
    common = {
        "callable": "mixed_geometry:mixed_sq",
        "symbols": symbols,
        "args": [{"name": "left", "width": 32}, {"name": "right", "width": 32}],
    }
    return [dict(common, statements=series, output="d", domain="t < 1e-3", name="mixed_distance_16_series",
                 note="series branch of phi; sympytorch has no Piecewise"),
            dict(common, statements=asym, output="d", domain="t >= 1e-3", name="mixed_distance_16_asym",
                 note="asymptotic branch of phi; the series branch is covered by the sibling call")]


def ball(n, seed, radius=0.9):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, H, generator=g, dtype=torch.float64)
    h = h / h.norm(dim=-1, keepdim=True) * (radius * torch.rand(n, 1, generator=g, dtype=torch.float64))
    return h


def cases():
    g = torch.Generator().manual_seed(7)
    n = 64
    left = torch.randn(n, 32, generator=g, dtype=torch.float64)
    right = torch.randn(n, 32, generator=g, dtype=torch.float64)
    left[:, E:] = ball(n, 11)
    right[:, E:] = ball(n, 12)
    yield "random", left, right
    near_l = left.clone()
    near_r = right.clone()
    near_r[:, E:] = near_l[:, E:] + 1e-6 / (H ** 0.5)
    near_r[:, :E] += 1e-3
    yield "near-collision", near_l, near_r
    b_l = left.clone()
    b_r = right.clone()
    b_l[:, E:] = ball(n, 21, 0.9)
    b_r[:, E:] = ball(n, 22, 0.9)
    yield "boundary-0.9", b_l, b_r


def compare(reference, left, right):
    left_ref = left.clone().requires_grad_(True)
    right_ref = right.clone().requires_grad_(True)
    out_ref = reference(left=left_ref, right=right_ref)
    left_g = left.clone().requires_grad_(True)
    right_g = right.clone().requires_grad_(True)
    out = MG.mixed_sq(left_g, right_g)
    dvalue = float((out_ref - out).abs().max())
    g_left, = torch.autograd.grad(out.sum(), left_g, retain_graph=True)
    g_right, = torch.autograd.grad(out.sum(), right_g, retain_graph=True)
    gref_left, = torch.autograd.grad(out_ref.sum(), left_ref, retain_graph=True)
    gref_right, = torch.autograd.grad(out_ref.sum(), right_ref, retain_graph=True)
    dg = max(float((gref_left - g_left).abs().max()), float((gref_right - g_right).abs().max()))
    return dvalue, dg


def main():
    calls = build_calls()
    series_reference, _ = build_reference(calls[0])
    asym_reference, _ = build_reference(calls[1])
    report = {"tolerances": {"value": [0.0, 1e-12], "gradient": [0.0, 1e-10]}}
    worst_v = 0.0
    worst_g = 0.0
    for name, left, right in cases():
        reference = series_reference if name == "near-collision" else asym_reference
        v, g = compare(reference, left, right)
        worst_v = max(worst_v, v)
        worst_g = max(worst_g, g)
        report[name] = {"value_abs": v, "grad_abs": g,
                        "value_pass": v <= 1e-12, "grad_pass": g <= 1e-10}
    left = torch.randn(8, 32, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    right = torch.randn(8, 32, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    left[:, E:] = ball(8, 31)
    right[:, E:] = ball(8, 32)
    left.requires_grad_(True)
    right.requires_grad_(True)
    gradcheck = torch.autograd.gradcheck(
        lambda a, b: asym_reference(left=a, right=b).mean(), (left, right),
        eps=1e-6, atol=1e-8, rtol=1e-5)
    e_on = MG.mixed_sq(left, right, True)
    e_off = MG.mixed_sq(left, right, False)
    euclid = ((left[:, :E] - right[:, :E]) ** 2).sum(-1) + ((left[:, E:] - right[:, E:]) ** 2).sum(-1)
    report["gradcheck_reference"] = bool(gradcheck)
    report["euclid_branch_max_abs"] = float((e_off - euclid).abs().max())
    report["euclid_branch_pass"] = bool(torch.allclose(e_off, euclid, rtol=0.0, atol=1e-12))
    report["geometry_on_off_differ"] = float((e_on - e_off).abs().max())
    report["worst_value_abs"] = worst_v
    report["worst_grad_abs"] = worst_g
    report["pass"] = bool(worst_v <= 1e-12 and worst_g <= 1e-10 and gradcheck
                          and report["euclid_branch_pass"])
    print(json.dumps(report, indent=1))

    contract_path = CANDIDATE / "reference_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["calls"] = [c for c in contract.get("calls", []) if c.get("callable") != "mixed_geometry:mixed_sq"]
    contract["calls"].extend(calls)
    contract_path.write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n")
    print("contract updated:", contract_path)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
