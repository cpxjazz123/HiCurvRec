"""Rewire the A2-R2 trees (#35) from the A2 tree's own source.

All three conditions share the loss plumbing; they differ only in ``LAMBDA_ER12``
and whether ``s12`` is built from a stop-gradiented ``s1``. ``L_ER1``'s own ``s1``
is never detached in any of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OLD_IMPORT = """import hbpe_a2 as _hbpe2
from hbpe_a2 import (
    _compute_reference_gap,
    _continuous_l2_residual_proxy,
    _precompute_matched_negatives,
    _soft_prefix_distribution,
    hbpe_joint_enrichment_loss,
)"""

NEW_IMPORT = """import hbpe_a2r2 as _hbpe2
from hbpe_a2r2 import (
    LAMBDA_ER1,
    LAMBDA_ER12,
    STOPGRAD_L1_IN_S12,
    _compute_reference_gap,
    _continuous_l2_residual_proxy,
    _precompute_matched_negatives,
    _soft_prefix_distribution,
    hard_prefix_sharing,
    hbpe_joint_enrichment_loss,
)"""

OLD_LISTS = """        hbpe_l_er1: list[float] = []
        hbpe_l_er12: list[float] = []
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
        hbpe_g_er1: list[float] = []
        hbpe_g_er12: list[float] = []
        hbpe_g_cosine: list[float] = []
        hard_stats: dict[str, list[float]] = {}"""

OLD_STEP = """                loss_a2, stats_a2 = hbpe_joint_enrichment_loss(p1_a, p1_p, p1_n, p2_a, p2_p, p2_n)

                enc_grad_norm = 0.0
                if loss_a2.requires_grad and loss_a2.grad_fn is not None:
                    grads = torch.autograd.grad(
                        loss_a2, tuple(raw_module.encoder.parameters()), retain_graph=True, allow_unused=True
                    )
                    flat_grads = [g.reshape(-1) for g in grads if g is not None]
                    if flat_grads:
                        enc_grad_norm = float(torch.cat(flat_grads).norm().item())
"""

NEW_STEP = """                loss_a2, l_er1, l_er12, stats_a2 = hbpe_joint_enrichment_loss(
                    p1_a, p1_p, p1_n, p2_a, p2_p, p2_n
                )

                # The two ER terms are differentiated separately so their individual
                # strength on the Encoder and the angle between them are visible.
                enc_params = tuple(raw_module.encoder.parameters())
                g_er1 = g_er12 = 0.0
                g_cosine = 0.0
                if l_er1.requires_grad and l_er1.grad_fn is not None:
                    flat1 = [
                        g.reshape(-1) for g in torch.autograd.grad(
                            l_er1, enc_params, retain_graph=True, allow_unused=True
                        ) if g is not None
                    ]
                    g_er1 = float(torch.cat(flat1).norm().item()) if flat1 else 0.0
                    if l_er12.requires_grad and l_er12.grad_fn is not None:
                        flat2 = [
                            g.reshape(-1) for g in torch.autograd.grad(
                                l_er12, enc_params, retain_graph=True, allow_unused=True
                            ) if g is not None
                        ]
                        if flat1 and flat2:
                            v1 = torch.cat(flat1)
                            v2 = torch.cat(flat2)
                            g_er12 = float(v2.norm().item())
                            g_cosine = float(
                                torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-12)
                            )
                enc_grad_norm = float(
                    torch.cat([
                        g.reshape(-1) for g in torch.autograd.grad(
                            loss_a2, enc_params, retain_graph=True, allow_unused=True
                        ) if g is not None
                    ]).norm().item()
                )

                # Discrete prefix statistics on the same batch, so the run can show
                # when the soft ratio climbs while the hard one falls.
                for key, value in hard_prefix_sharing(
                    z_a, z_p, z_n, c1_det, c2_det
                ).items():
                    hard_stats.setdefault(key, []).append(value)
                hbpe_g_er1.append(g_er1)
                hbpe_g_er12.append(g_er12)
                hbpe_g_cosine.append(g_cosine)
"""

OLD_FIELDS = """        hbpe_fields = {
            "L_ER1": avg_l_er1,
            "L_ER12": avg_l_er12,
            "S1_pos": avg_s1_pos,
            "S1_neg": avg_s1_neg,
            "S12_pos": avg_s12_pos,
            "S12_neg": avg_s12_neg,
            "Soft_ER1": avg_soft_er1,
            "Soft_ER12": avg_soft_er12,
            "T1": hbpe_t1,
            "T2": hbpe_t2,
            "encoder_gradient_norm": avg_enc_grad,
        }"""

NEW_FIELDS = """        hbpe_fields = {
            "L_ER1": avg_l_er1,
            "L_ER12": avg_l_er12,
            "S1_pos": avg_s1_pos,
            "S1_neg": avg_s1_neg,
            "S12_pos": avg_s12_pos,
            "S12_neg": avg_s12_neg,
            "Soft_ER1": avg_soft_er1,
            "Soft_ER12": avg_soft_er12,
            "T1": hbpe_t1,
            "T2": hbpe_t2,
            "encoder_gradient_norm": avg_enc_grad,
            "lambda_er1": LAMBDA_ER1,
            "lambda_er12": LAMBDA_ER12,
            "stopgrad_l1_in_s12": STOPGRAD_L1_IN_S12,
            "g_er1": float(np.mean(hbpe_g_er1)) if hbpe_g_er1 else 0.0,
            "g_er12": float(np.mean(hbpe_g_er12)) if hbpe_g_er12 else 0.0,
            "g_er1_er12_cosine": float(np.mean(hbpe_g_cosine)) if hbpe_g_cosine else 0.0,
            **{key: float(np.mean(values)) for key, values in sorted(hard_stats.items())},
        }"""

OLD_ID = """EXPERIMENT_ARM = "hbpe_a2"
if EXPERIMENT_ARM != "hbpe_a2":
    raise RuntimeError(
        f"only the 'hbpe_a2' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
    )"""


def rewire(tree: str, arm: str) -> None:
    path = REPO_ROOT / "stage2_RQ-VAE" / tree / "train_rqvae.py"
    text = path.read_text(encoding="utf-8")

    for old, new, label in (
        (OLD_IMPORT, NEW_IMPORT, "imports"),
        (OLD_LISTS, NEW_LISTS, "metric lists"),
        (OLD_STEP, NEW_STEP, "loss step"),
        (OLD_FIELDS, NEW_FIELDS, "metric fields"),
    ):
        assert text.count(old) == 1, f"{tree}: expected exactly one {label} block"
        text = text.replace(old, new)

    assert text.count(OLD_ID) == 1, f"{tree}: identity block not found"
    text = text.replace(OLD_ID, (
        f'EXPERIMENT_ARM = "{arm}"\n'
        f'if EXPERIMENT_ARM != "{arm}":\n'
        f"    raise RuntimeError(\n"
        f"        f\"only the {arm!r} mechanism may run in this tree; "
        f"got {{EXPERIMENT_ARM!r}}\"\n"
        f"    )"
    ))

    # The loss is now already weighted inside hbpe_joint_enrichment_loss.
    old_add = "                loss = loss + ramp_a2 * loss_a2"
    assert text.count(old_add) == 1, f"{tree}: loss addition not found"
    text = text.replace(old_add, "                loss = loss + ramp_a2 * loss_a2")

    path.write_text(text, encoding="utf-8")
    print(f"rewired {tree} -> arm {arm}")


if __name__ == "__main__":
    rewire("TIGER_HBPE_A2_R2_L2W002", "a2r2_l2w002")
    rewire("TIGER_HBPE_A2_R2_L2W005", "a2r2_l2w005")
    rewire("TIGER_HBPE_A2_R2_L2W005_SGL1", "a2r2_l2w005_sgL1")
    sys.exit(0)
