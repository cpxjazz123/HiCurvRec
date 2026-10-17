"""Rewire the A1-R2 trees (#34) from the A1 tree's own source.

Every A1-R2 condition is a cold start from the same Exp1a Resample recipe; the only
things that change between them are the enrichment weight and, for the occupancy
condition, one extra barrier term. This script applies those two edits to a copy of
A1's trainer so the three trees cannot drift apart in anything else.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OLD_IMPORT = """import hbpe_a1 as _hbpe
from hbpe_a1 import (
    _compute_reference_gap,
    _precompute_matched_negatives,
    _soft_prefix_distribution,
    hbpe_l1_loss,
)"""

NEW_IMPORT = """import hbpe_a1r2 as _hbpe
from hbpe_a1r2 import (
    H_REF_L1,
    LAMBDA_ER1,
    LAMBDA_OCC,
    OCC_SAMPLE,
    USE_OCCUPANCY_BARRIER,
    _compute_reference_gap,
    _precompute_matched_negatives,
    _soft_prefix_distribution,
    hbpe_l1_loss,
    hbpe_l1_occupancy_loss,
)"""

# The metric list and the stats plumbing are identical in all three conditions; the
# occupancy barrier simply appends its own entries.
OLD_LISTS = """        hbpe_losses: list[float] = []
        hbpe_s1_pos: list[float] = []
        hbpe_s1_neg: list[float] = []
        hbpe_er1: list[float] = []
        hbpe_enc_grads: list[float] = []
        hbpe_temp = 0.0"""

NEW_LISTS = """        hbpe_losses: list[float] = []
        hbpe_s1_pos: list[float] = []
        hbpe_s1_neg: list[float] = []
        hbpe_er1: list[float] = []
        hbpe_enc_grads: list[float] = []
        hbpe_temp = 0.0
        hbpe_l_occ: list[float] = []
        hbpe_h_pi: list[float] = []
        hbpe_eff_codes: list[float] = []
        hbpe_hard_codes: list[float] = []
        occ_enc_grads: list[float] = []"""

OLD_TAIL = """                hbpe_losses.append(stats_a1["L_A1"])
                hbpe_s1_pos.append(stats_a1["S1_pos"])
                hbpe_s1_neg.append(stats_a1["S1_neg"])
                hbpe_er1.append(stats_a1["Soft_ER1"])
                hbpe_enc_grads.append(enc_grad_norm)
                hbpe_temp = float(T1)"""

NEW_TAIL = """                hbpe_losses.append(stats_a1["L_A1"])
                hbpe_s1_pos.append(stats_a1["S1_pos"])
                hbpe_s1_neg.append(stats_a1["S1_neg"])
                hbpe_er1.append(stats_a1["Soft_ER1"])
                hbpe_enc_grads.append(enc_grad_norm)
                hbpe_temp = float(T1)

                if USE_OCCUPANCY_BARRIER:
                    occ_rows = torch.randint(
                        len(train_ids), (OCC_SAMPLE,), generator=occ_gen
                    )
                    occ_z = raw_module.encoder(
                        all_embeddings[
                            torch.from_numpy(train_ids)[occ_rows]
                        ].to(device, non_blocking=True)
                    )
                    loss_occ, stats_occ = hbpe_l1_occupancy_loss(
                        occ_z, c1_det, T1, H_REF_L1
                    )
                    occ_grad_norm = 0.0
                    if loss_occ.requires_grad and loss_occ.grad_fn is not None:
                        occ_grads = torch.autograd.grad(
                            loss_occ, tuple(raw_module.encoder.parameters()),
                            retain_graph=True, allow_unused=True,
                        )
                        occ_flat = [g.reshape(-1) for g in occ_grads if g is not None]
                        if occ_flat:
                            occ_grad_norm = float(torch.cat(occ_flat).norm().item())
                    hbpe_l_occ.append(stats_occ["L_occ"])
                    hbpe_h_pi.append(stats_occ["H_pi"])
                    hbpe_eff_codes.append(stats_occ["effective_codes_exp_H"])
                    hbpe_hard_codes.append(float(stats_occ["hard_distinct_l1_codes"]))
                    occ_enc_grads.append(occ_grad_norm)"""

OLD_ADD = """            if epoch_weight > 0.0:
                ramp_a1 = min(max((epoch - 1500) / 1500.0, 0.0), 1.0) if EPOCHS > 1 else 1.0
                loss = loss + 0.02 * ramp_a1 * loss_a1"""

NEW_ADD = """            if epoch_weight > 0.0:
                ramp_a1 = min(max((epoch - 1500) / 1500.0, 0.0), 1.0) if EPOCHS > 1 else 1.0
                loss = loss + LAMBDA_ER1 * ramp_a1 * loss_a1
                if USE_OCCUPANCY_BARRIER:
                    loss = loss + LAMBDA_OCC * ramp_a1 * loss_occ"""

OLD_FIELDS = """        hbpe_fields = {
            "L_A1": avg_l_a1,
            "S1_pos": avg_s1_pos,
            "S1_neg": avg_s1_neg,
            "Soft_ER1": avg_soft_er1,
            "temperature": hbpe_temp,
            "encoder_gradient_norm": avg_enc_grad,
        }"""

NEW_FIELDS = """        hbpe_fields = {
            "L_A1": avg_l_a1,
            "S1_pos": avg_s1_pos,
            "S1_neg": avg_s1_neg,
            "Soft_ER1": avg_soft_er1,
            "lambda_er1": LAMBDA_ER1,
            "temperature": hbpe_temp,
            "encoder_gradient_norm": avg_enc_grad,
            "lambda_occ": LAMBDA_OCC if USE_OCCUPANCY_BARRIER else 0.0,
            "L_occ": float(np.mean(hbpe_l_occ)) if hbpe_l_occ else 0.0,
            "H_pi": float(np.mean(hbpe_h_pi)) if hbpe_h_pi else 0.0,
            "H_ref_L1": H_REF_L1 if USE_OCCUPANCY_BARRIER else 0.0,
            "effective_codes_exp_H": float(np.mean(hbpe_eff_codes)) if hbpe_eff_codes else 0.0,
            "hard_distinct_L1_codes": float(np.mean(hbpe_hard_codes)) if hbpe_hard_codes else 0.0,
            "occupancy_encoder_gradient_norm": (
                float(np.mean(occ_enc_grads)) if occ_enc_grads else 0.0
            ),
        }"""

OLD_PRE = """    d_ref1: float | None = None
"""

NEW_PRE = """    d_ref1: float | None = None
    # The occupancy barrier samples the catalogue uniformly, so it needs its own
    # stream rather than the pair sampler's. Kept on CPU to match the item ids.
    occ_gen = torch.Generator().manual_seed(SEED + 31337 + rank)
"""

OLD_ID = """EXPERIMENT_ARM = "hbpe_a1"
if EXPERIMENT_ARM != "hbpe_a1":
    raise RuntimeError(
        f"only the 'hbpe_a1' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
    )"""


def rewire(tree: str, arm: str, occupancy: bool) -> None:
    path = REPO_ROOT / "stage2_RQ-VAE" / tree / "train_rqvae.py"
    text = path.read_text(encoding="utf-8")

    for old, new, label in (
        (OLD_IMPORT, NEW_IMPORT, "imports"),
        (OLD_LISTS, NEW_LISTS, "metric lists"),
        (OLD_PRE, NEW_PRE, "occupancy sampler init"),
        (OLD_TAIL, NEW_TAIL, "stats tail"),
        (OLD_ADD, NEW_ADD, "loss addition"),
        (OLD_FIELDS, NEW_FIELDS, "metric fields"),
    ):
        assert text.count(old) == 1, f"{tree}: expected exactly one {label} block"
        text = text.replace(old, new)

    new_id = (
        f'EXPERIMENT_ARM = "{arm}"\n'
        f'if EXPERIMENT_ARM != "{arm}":\n'
        f"    raise RuntimeError(\n"
        f"        f\"only the {arm!r} mechanism may run in this tree; "
        f"got {{EXPERIMENT_ARM!r}}\"\n"
        f"    )"
    )
    assert text.count(OLD_ID) == 1, f"{tree}: identity block not found"
    text = text.replace(OLD_ID, new_id)

    if not occupancy:
        # Dead weight in the two pure-weight conditions: drop the barrier and the
        # sampler it needed.
        text = text.replace(NEW_TAIL, OLD_TAIL)
        text = text.replace(
            NEW_ADD,
            OLD_ADD.replace("0.02 * ramp_a1 * loss_a1", "LAMBDA_ER1 * ramp_a1 * loss_a1"),
        )
        text = text.replace(NEW_FIELDS, OLD_FIELDS.replace(
            '            "temperature": hbpe_temp,',
            '            "lambda_er1": LAMBDA_ER1,\n            "temperature": hbpe_temp,',
        ))
        text = text.replace(NEW_PRE, OLD_PRE)
        text = text.replace(
            "        hbpe_l_occ: list[float] = []\n"
            "        hbpe_h_pi: list[float] = []\n"
            "        hbpe_eff_codes: list[float] = []\n"
            "        hbpe_hard_codes: list[float] = []\n",
            "",
        )
        text = text.replace(
            "from hbpe_a1r2 import (\n"
            "    H_REF_L1,\n"
            "    LAMBDA_ER1,\n"
            "    LAMBDA_OCC,\n"
            "    OCC_SAMPLE,\n"
            "    USE_OCCUPANCY_BARRIER,\n"
            "    _compute_reference_gap,\n"
            "    _precompute_matched_negatives,\n"
            "    _soft_prefix_distribution,\n"
            "    hbpe_l1_loss,\n"
            "    hbpe_l1_occupancy_loss,\n"
            ")",
            "from hbpe_a1r2 import (\n"
            "    LAMBDA_ER1,\n"
            "    _compute_reference_gap,\n"
            "    _precompute_matched_negatives,\n"
            "    _soft_prefix_distribution,\n"
            "    hbpe_l1_loss,\n"
            ")",
        )

    path.write_text(text, encoding="utf-8")
    print(f"rewired {tree} -> arm {arm} (occupancy={occupancy})")


if __name__ == "__main__":
    rewire("TIGER_HBPE_A1_R2_W005", "a1r2_w005", occupancy=False)
    rewire("TIGER_HBPE_A1_R2_W010", "a1r2_w010", occupancy=False)
    rewire("TIGER_HBPE_A1_R2_W010_OCC", "a1r2_w010_occ", occupancy=True)
    sys.exit(0)
