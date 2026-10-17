"""Rewire the A3-R2 trees (#36) from the A3 tree's own source.

Both conditions keep A3's two ER terms and its positive-coverage floors and add the
negative-sharing cap; only the occupancy guard differs. Both need the frozen
canonical Resample checkpoint loaded as a pure teacher: it is never optimised, never
back-propagated through, and only read to measure B_neg(tau) and the L1 entropy
reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OLD_IMPORT = """import hbpe_a3 as _hbpe3
from hbpe_a3 import (
    _compute_reference_gap,
    _continuous_l2_residual_proxy,
    _precompute_matched_negatives,
    _soft_prefix_distribution,
    hbpe_coverage_enrichment_loss,
    RHO1,
    RHO12,
)"""

NEW_IMPORT = """import hbpe_a3r2 as _hbpe3
from hbpe_a3r2 import (
    H_REF_L1,
    LAMBDA_ER1,
    LAMBDA_ER12,
    LAMBDA_NEG,
    LAMBDA_OCC,
    OCC_SAMPLE,
    RHO1,
    RHO12,
    TEACHER_CHECKPOINT,
    USE_OCCUPANCY_GUARD,
    _compute_reference_gap,
    _continuous_l2_residual_proxy,
    _precompute_matched_negatives,
    _soft_prefix_distribution,
    hard_prefix_sharing,
    hbpe_l1_occupancy_loss,
    hbpe_negcap_enrichment_loss,
    teacher_negative_sharing,
)"""

OLD_LISTS = """        hbpe_l_er1: list[float] = []
        hbpe_l_er12: list[float] = []
        hbpe_l_cov1: list[float] = []
        hbpe_l_cov12: list[float] = []
        hbpe_s1_pos: list[float] = []
        hbpe_s1_neg: list[float] = []
        hbpe_s12_pos: list[float] = []
        hbpe_s12_neg: list[float] = []
        hbpe_er1: list[float] = []
        hbpe_er12: list[float] = []
        hbpe_enc_grads: list[float] = []
        hbpe_t1 = 0.0
        hbpe_t2 = 0.0"""

NEW_LISTS = OLD_LISTS + """
        hbpe_l_neg1: list[float] = []
        hbpe_l_neg12: list[float] = []
        hbpe_cap1_active: list[float] = []
        hbpe_cap12_active: list[float] = []
        hbpe_b1_neg: list[float] = []
        hbpe_b12_neg: list[float] = []
        hbpe_l_occ: list[float] = []
        hbpe_h_pi: list[float] = []
        hbpe_eff_codes: list[float] = []
        hbpe_hard_codes: list[float] = []
        occ_enc_grads: list[float] = []
        hard_stats: dict[str, list[float]] = {}"""

# Load the frozen teacher once, next to the model, and freeze it.
OLD_TEACHER_ANCHOR = """    parameters = list(model.parameters())
    if OPTIMIZER.lower() == "adagrad":"""

NEW_TEACHER_ANCHOR = """    # Frozen canonical Resample, used ONLY to measure the negative-sharing reference
    # and the L1 entropy target. No gradient reaches it and it is not in `parameters`.
    teacher = RQVAE(config, in_dim=len(embeddings[0])).to(device)
    _teacher_payload = torch.load(TEACHER_CHECKPOINT, map_location=device)
    teacher.load_state_dict(_teacher_payload.get("state_dict", _teacher_payload), strict=True)
    teacher.eval()
    for _teacher_parameter in teacher.parameters():
        _teacher_parameter.requires_grad_(False)
    teacher_c1 = teacher.rq.vq_layers[0].get_code_embs().detach()
    teacher_c2 = teacher.rq.vq_layers[1].get_code_embs().detach()
    with torch.no_grad():
        teacher_d_ref1 = _compute_reference_gap(
            teacher.encoder(train_embeddings[:8192].to(device)), teacher_c1
        )
        teacher_d_ref2 = _compute_reference_gap(
            _continuous_l2_residual_proxy(
                teacher.encoder(train_embeddings[:8192].to(device)), teacher_c1
            ),
            teacher_c2,
        )
    if rank == 0:
        print(
            f"[HBPE-A3-R2] teacher loaded (read-only): "
            f"D_ref1={teacher_d_ref1:.8f} D_ref2={teacher_d_ref2:.8f}",
            flush=True,
        )

    parameters = list(model.parameters())
    if OPTIMIZER.lower() == "adagrad":"""

OLD_STEP = """                loss_a3, stats_a3 = hbpe_coverage_enrichment_loss(p1_a, p1_p, p1_n, p2_a, p2_p, p2_n, rho1=RHO1, rho12=RHO12)

                enc_grad_norm = 0.0
                if loss_a3.requires_grad and loss_a3.grad_fn is not None:
                    grads = torch.autograd.grad(
                        loss_a3, tuple(raw_module.encoder.parameters()), retain_graph=True, allow_unused=True
                    )
                    flat_grads = [g.reshape(-1) for g in grads if g is not None]
                    if flat_grads:
                        enc_grad_norm = float(torch.cat(flat_grads).norm().item())
"""

NEW_STEP = """                # Teacher reference on the SAME items, no_grad, using the teacher's
                # own distance-gap scale times the same tau.
                with torch.no_grad():
                    anchor_emb = all_embeddings[source_ids.cpu()].to(device, non_blocking=True)
                    b1_neg, b12_neg = teacher_negative_sharing(
                        teacher.encoder(anchor_emb), teacher.encoder(neg_emb),
                        teacher_c1, teacher_c2,
                        teacher_d_ref1, teacher_d_ref2, tau,
                    )

                loss_a3, stats_a3 = hbpe_negcap_enrichment_loss(
                    p1_a, p1_p, p1_n, p2_a, p2_p, p2_n, b1_neg, b12_neg,
                    rho1=RHO1, rho12=RHO12,
                )

                enc_grad_norm = 0.0
                if loss_a3.requires_grad and loss_a3.grad_fn is not None:
                    grads = torch.autograd.grad(
                        loss_a3, tuple(raw_module.encoder.parameters()), retain_graph=True, allow_unused=True
                    )
                    flat_grads = [g.reshape(-1) for g in grads if g is not None]
                    if flat_grads:
                        enc_grad_norm = float(torch.cat(flat_grads).norm().item())

                hbpe_l_neg1.append(stats_a3["L_neg1"])
                hbpe_l_neg12.append(stats_a3["L_neg12"])
                hbpe_cap1_active.append(stats_a3["negcap1_active"])
                hbpe_cap12_active.append(stats_a3["negcap12_active"])
                hbpe_b1_neg.append(stats_a3["B1_neg"])
                hbpe_b12_neg.append(stats_a3["B12_neg"])

                for key, value in hard_prefix_sharing(
                    z_a, z_p, z_n, c1_det, c2_det
                ).items():
                    hard_stats.setdefault(key, []).append(value)

                if USE_OCCUPANCY_GUARD:
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
                    occ_enc_grads.append(occ_grad_norm)
"""

OLD_ADD = """            if epoch_weight > 0.0:
                ramp_a3 = min(max((epoch - 1500) / 1500.0, 0.0), 1.0) if EPOCHS > 1 else 1.0
                loss = loss + ramp_a3 * loss_a3"""

NEW_ADD = """            if epoch_weight > 0.0:
                ramp_a3 = min(max((epoch - 1500) / 1500.0, 0.0), 1.0) if EPOCHS > 1 else 1.0
                loss = loss + ramp_a3 * loss_a3
                if USE_OCCUPANCY_GUARD:
                    loss = loss + LAMBDA_OCC * ramp_a3 * loss_occ"""

OLD_FIELDS = """            "rho1": float(RHO1),
            "rho12": float(RHO12),
            "temperature": hbpe_t1,
            "encoder_gradient_norm": avg_enc_grad,
        }"""

NEW_FIELDS = """            "rho1": float(RHO1),
            "rho12": float(RHO12),
            "temperature": hbpe_t1,
            "encoder_gradient_norm": avg_enc_grad,
            "lambda_er1": LAMBDA_ER1,
            "lambda_er12": LAMBDA_ER12,
            "lambda_neg": LAMBDA_NEG,
            "L_neg1": float(np.mean(hbpe_l_neg1)) if hbpe_l_neg1 else 0.0,
            "L_neg12": float(np.mean(hbpe_l_neg12)) if hbpe_l_neg12 else 0.0,
            "negcap1_active_rate": float(np.mean(hbpe_cap1_active)) if hbpe_cap1_active else 0.0,
            "negcap12_active_rate": float(np.mean(hbpe_cap12_active)) if hbpe_cap12_active else 0.0,
            "B1_neg": float(np.mean(hbpe_b1_neg)) if hbpe_b1_neg else 0.0,
            "B12_neg": float(np.mean(hbpe_b12_neg)) if hbpe_b12_neg else 0.0,
            "lambda_occ": LAMBDA_OCC if USE_OCCUPANCY_GUARD else 0.0,
            "L_occ": float(np.mean(hbpe_l_occ)) if hbpe_l_occ else 0.0,
            "H_pi": float(np.mean(hbpe_h_pi)) if hbpe_h_pi else 0.0,
            "H_ref_L1": H_REF_L1 if USE_OCCUPANCY_GUARD else 0.0,
            "effective_codes_exp_H": float(np.mean(hbpe_eff_codes)) if hbpe_eff_codes else 0.0,
            "hard_distinct_L1_codes": float(np.mean(hbpe_hard_codes)) if hbpe_hard_codes else 0.0,
            "occupancy_encoder_gradient_norm": (
                float(np.mean(occ_enc_grads)) if occ_enc_grads else 0.0
            ),
            **{key: float(np.mean(values)) for key, values in sorted(hard_stats.items())},
        }"""

OLD_GEN = """    d_ref1: float | None = None
    d_ref2: float | None = None
"""

NEW_GEN = """    d_ref1: float | None = None
    d_ref2: float | None = None
    # The occupancy guard samples the catalogue uniformly, so it gets its own stream.
    occ_gen = torch.Generator().manual_seed(SEED + 31337 + rank)
"""

OLD_ID = """EXPERIMENT_ARM = "hbpe_a3"
if EXPERIMENT_ARM != "hbpe_a3":
    raise RuntimeError(
        f"only the 'hbpe_a3' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
    )"""

# The occupancy block, factored out so the plain negcap condition can drop it by
# exact match instead of by slicing a formatted string.
OCC_BLOCK = """
                if USE_OCCUPANCY_GUARD:
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
                    occ_enc_grads.append(occ_grad_norm)
"""

OCC_LISTS = """        hbpe_l_occ: list[float] = []
        hbpe_h_pi: list[float] = []
        hbpe_eff_codes: list[float] = []
        hbpe_hard_codes: list[float] = []
        occ_enc_grads: list[float] = []
"""

OCC_FIELDS = """            "lambda_occ": LAMBDA_OCC if USE_OCCUPANCY_GUARD else 0.0,
            "L_occ": float(np.mean(hbpe_l_occ)) if hbpe_l_occ else 0.0,
            "H_pi": float(np.mean(hbpe_h_pi)) if hbpe_h_pi else 0.0,
            "H_ref_L1": H_REF_L1 if USE_OCCUPANCY_GUARD else 0.0,
            "effective_codes_exp_H": float(np.mean(hbpe_eff_codes)) if hbpe_eff_codes else 0.0,
            "hard_distinct_L1_codes": float(np.mean(hbpe_hard_codes)) if hbpe_hard_codes else 0.0,
            "occupancy_encoder_gradient_norm": (
                float(np.mean(occ_enc_grads)) if occ_enc_grads else 0.0
            ),
"""

OCC_IMPORTS = (
    "    H_REF_L1,\n", "    LAMBDA_OCC,\n", "    OCC_SAMPLE,\n",
    "    USE_OCCUPANCY_GUARD,\n", "    hbpe_l1_occupancy_loss,\n",
)


def rewire(tree: str, arm: str, occupancy: bool) -> None:
    path = REPO_ROOT / "stage2_RQ-VAE" / tree / "train_rqvae.py"
    text = path.read_text(encoding="utf-8")

    for old, new, label in (
        (OLD_IMPORT, NEW_IMPORT, "imports"),
        (OLD_LISTS, NEW_LISTS, "metric lists"),
        (OLD_GEN, NEW_GEN, "sampler init"),
        (OLD_TEACHER_ANCHOR, NEW_TEACHER_ANCHOR, "teacher load anchor"),
        (OLD_STEP, NEW_STEP, "loss step"),
        (OLD_ADD, NEW_ADD, "loss addition"),
        (OLD_FIELDS, NEW_FIELDS, "metric fields"),
    ):
        assert text.count(old) == 1, f"{tree}: expected exactly one {label} block"
        text = text.replace(old, new)

    if not occupancy:
        # Dead weight in the plain negcap condition: drop the guard and everything
        # that existed only to serve it.
        for chunk in (OCC_BLOCK, OCC_LISTS, OCC_FIELDS):
            assert text.count(chunk) == 1, f"{tree}: occupancy chunk not found once"
            text = text.replace(chunk, "")
        text = text.replace(
            "                loss = loss + ramp_a3 * loss_a3\n"
            "                if USE_OCCUPANCY_GUARD:\n"
            "                    loss = loss + LAMBDA_OCC * ramp_a3 * loss_occ",
            "                loss = loss + ramp_a3 * loss_a3",
        )
        text = text.replace(NEW_GEN, OLD_GEN)
        for line in OCC_IMPORTS:
            text = text.replace(line, "")

    assert text.count(OLD_ID) == 1, f"{tree}: identity block not found"
    text = text.replace(OLD_ID, (
        f'EXPERIMENT_ARM = "{arm}"\n'
        f'if EXPERIMENT_ARM != "{arm}":\n'
        f"    raise RuntimeError(\n"
        f"        f\"only the {arm!r} mechanism may run in this tree; "
        f"got {{EXPERIMENT_ARM!r}}\"\n"
        f"    )"
    ))

    path.write_text(text, encoding="utf-8")
    print(f"rewired {tree} -> arm {arm} (occupancy={occupancy})")


if __name__ == "__main__":
    rewire("TIGER_HBPE_A3_R2_NEGCAP", "a3r2_negcap", occupancy=False)
    rewire("TIGER_HBPE_A3_R2_NEGCAP_OCC", "a3r2_negcap_occ", occupancy=True)
    sys.exit(0)
