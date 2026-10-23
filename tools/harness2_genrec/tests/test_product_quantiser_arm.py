"""The product-geodesic arm's own regression tests (issue #112, JOIN E).

The arm's implementation is in ``stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser``. These tests
pin the two things a later change could silently break:

1. the patched level-0 assignment equals a DIRECT computation of the declared formula -- the
   mechanism's mathematics, recomputed by the verifier rather than read from the arm;
2. the product metric is not a decoration: the joint assignment is not simply one factor's
   assignment, and it differs from the host's Euclidean one.

They need torch, so run them under the project interpreter:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest \
        tools/harness2_genrec/tests/test_product_quantiser_arm.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools" / "cgv") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "cgv"))

torch = pytest.importorskip("torch")

import engine  # noqa: E402

TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser"
C_S, C_B, SCALE = 1.0, 2.0, 1.4937731472770945


@pytest.fixture(scope="module")
def trainer():
    loaded, saved_path, saved_mods = engine.load_trainer(TREE, module_name="product_arm_test")
    yield loaded
    engine.unload_trainer(saved_path, saved_mods)


def _patched_layer(trainer, seed: int = 7):
    torch.manual_seed(seed)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=32)
    layer = model.rq.vq_layers[0]
    trainer._install_product_assignment(layer, C_S, C_B, SCALE)
    layer._hyp_assignment_active = True
    return layer


def test_the_patched_assignment_equals_a_direct_computation(trainer):
    """The mechanism's mathematics, recomputed independently of the arm's patch."""
    from scripts.hyp_losses import pair_poincare_distance
    from hyperbolic import expmap0

    layer = _patched_layer(trainer)
    latent = torch.randn(16, 32)
    with torch.no_grad():
        distances = layer._distances(latent)
        codes = layer.get_code_embs()
        half = latent.shape[1] // 2
        direct = (
            pair_poincare_distance(expmap0(SCALE * latent[:, :half], C_S),
                                   expmap0(codes[:, :half], C_S), C_S) ** 2
            + pair_poincare_distance(expmap0(SCALE * latent[:, half:], C_B),
                                     expmap0(codes[:, half:], C_B), C_B) ** 2
        )
    assert float((distances - direct).abs().max()) == 0.0, "the patch must BE the declared formula"


def test_the_product_metric_is_not_a_decoration(trainer):
    """The joint assignment is neither factor's assignment, and it differs from the host's."""
    from hyperbolic import expmap0
    from scripts.hyp_losses import pair_poincare_distance

    layer = _patched_layer(trainer)
    latent = torch.randn(64, 32)
    with torch.no_grad():
        distances = layer._distances(latent)
        codes = layer.get_code_embs()
        half = latent.shape[1] // 2
        d_s = pair_poincare_distance(expmap0(SCALE * latent[:, :half], C_S),
                                     expmap0(codes[:, :half], C_S), C_S) ** 2
        d_b = pair_poincare_distance(expmap0(SCALE * latent[:, half:], C_B),
                                     expmap0(codes[:, half:], C_B), C_B) ** 2
        joint = distances.argmin(-1)
        only_s = d_s.argmin(-1)
        only_b = d_b.argmin(-1)
        host = torch.cdist(latent, codes).argmin(-1)
    assert float((joint == only_s).double().mean()) < 0.99, "the joint is the semantic factor alone"
    assert float((joint == only_b).double().mean()) < 0.99, "the joint is the behaviour factor alone"
    assert float((joint != host).double().mean()) > 0.01, "the product metric must move assignments"


def test_the_declared_parameters_are_the_measured_ones(trainer):
    """The scale and the relative curvatures come from the measurement, not from taste."""
    config = trainer._HYP_ARM_CONFIGS["hyp_product_c1"]
    assert config["mechanism"] == "product"
    assert config["curvature"] == C_S
    assert config["curvature_b"] == C_B * 1.0
    assert config["tangent_scale"] == pytest.approx(SCALE, rel=0, abs=0)
    assert config["curvature_b"] / config["curvature"] == pytest.approx(2.0)
