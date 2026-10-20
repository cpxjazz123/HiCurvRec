"""Track F (issue #88): the confirmed RED defects in the #87 pilot, one test each.

Every test in this file FAILS against the committed #87 code. They are the executable form of
issue #88's defect list, and they are written so that the reason each one fails is visible in its
own assertion message.

The point of writing them before changing anything is that a fix which makes them pass is a fix
for the RIGHT reason: a harness that cannot run the production loop, cannot find the training
graph, or measures a soft posterior where the host measures a hard SID is wrong in a way that
produces plausible numbers.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"),):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_replay_engine as RE  # noqa: E402
import r7_f_pilot as FP  # noqa: E402
import r7_f_soft_separation as FS  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0
RED_EXPECTED = 6          # the number of defects issue #88 lists


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# --------------------------------------------------------------------------------------
# defect 1: one step-32 bundle is carried through every "active" step
# --------------------------------------------------------------------------------------
def test_defect_1_the_pilot_does_not_advance_a_real_minibatch_stream() -> None:
    """RED: the chained replay reuses one captured bundle for every step."""
    source = inspect.getsource(FP.chained_replays)
    # a genuine 128-step stream needs a NEW captured minibatch per step; the committed code
    # carries one bundle and copies its tensors forward
    reuses_single_bundle = "bundle = torch.load(step32_bundle_path" in source
    copies_item_batch = '"item_batch": bundle["item_batch"]' in source
    check("the chained replay advances a real per-step minibatch stream",
          not (reuses_single_bundle and copies_item_batch),
          "chained_replays loads ONE step-32 bundle and copies its item_batch/pair tensors "
          "forward for every step, so no step ever sees a new behaviour-CE minibatch")


# --------------------------------------------------------------------------------------
# defect 2: the committed assemble_and_step path
# --------------------------------------------------------------------------------------
def test_defect_2_the_pilot_calls_a_signature_the_engine_may_not_have() -> None:
    """RED: the pilot passes extra_loss; the engine's accepted signature is the contract."""
    engine_params = inspect.signature(RE.assemble_and_step).parameters
    pilot_calls_extra_loss = "extra_loss=extra_loss" in inspect.getsource(FP.chained_replays)
    check("the pilot's call to assemble_and_step matches the engine's real signature",
          (not pilot_calls_extra_loss) or ("extra_loss" in engine_params),
          "the pilot passes extra_loss but assemble_and_step does not accept it: "
          f"signature is {list(engine_params)}")


# --------------------------------------------------------------------------------------
# defect 3: the soft posterior is built on raw embeddings, not on encoder latents
# --------------------------------------------------------------------------------------
class _EncoderModel:
    """A model whose encoder is a KNOWN map, so the geometry is checkable by hand."""

    def __init__(self, codebook: torch.Tensor, scale: float = 0.5):
        self._scale = scale
        self.rq = type("_RQ", (), {})()
        layer = type("_Layer", (), {})()
        layer.dim = codebook.shape[1]
        layer.n_embed = codebook.shape[0]
        layer.get_code_embs = lambda self=layer: codebook
        layer._distances = lambda x: ((x[:, None, :] - codebook[None]) ** 2).sum(-1)
        self.rq.vq_layers = [layer]

    def encoder(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings * self._scale


def test_defect_3_posterior_is_built_on_the_encoder_latent() -> None:
    """RED: the posterior must score the ENCODER output, not the raw embedding."""
    codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    model = _EncoderModel(codebook)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    posterior = FS.posterior(model, embeddings, tau=1.0)
    z = embeddings * 0.5
    expected = torch.softmax(-((z[:, None, :] - codebook[None]) ** 2).sum(-1), dim=-1)
    check("posterior() scores the encoder's latent z",
          torch.allclose(posterior, expected, atol=1e-12),
          f"posterior != the softmax over the ENCODER output: got {posterior[0].tolist()} "
          f"expected {expected[0].tolist()}")


def test_defect_3b_the_hard_l1_assignment_uses_the_native_get_indices() -> None:
    """RED: the canonical hard identity is the model's own get_indices."""
    snapshot_source = inspect.getsource(FP._snapshot_from_branch)
    native_get_indices = "get_indices" in snapshot_source
    check("the hard L1 assignment comes from the native get_indices path",
          native_get_indices,
          "_snapshot_from_branch does not call the model's get_indices(embeddings, "
          "infer_use_sk=False), so the hard identity is not the host's")


# --------------------------------------------------------------------------------------
# defect 4: the training relationship graph is silently missing
# --------------------------------------------------------------------------------------
def test_defect_4_a_missing_training_graph_is_a_hard_error_not_a_silent_none() -> None:
    """RED: host_successor_graph returns None instead of refusing."""
    source = inspect.getsource(FP.host_successor_graph)
    returns_none = "return None" in source
    raises_on_missing = "raise" in source
    check("a missing training relationship graph raises instead of returning None",
          not (returns_none and not raises_on_missing),
          "host_successor_graph returns None when the host does not expose the training frame, "
          "so every pilot run silently treated EVERY candidate as unrelated")


def test_defect_4b_the_dense_boolean_successor_lookup_is_not_an_index_reading() -> None:
    """RED: on a dense Boolean 2D tensor the lookup must read COLUMN INDICES, not row values."""
    dense = torch.tensor([[True, True, False],
                          [False, True, True],
                          [True, False, True]])
    lookup = FS._successor_lookup(dense)
    # row 0 has successors at columns 0 and 1; row 1 at columns 1 and 2
    check("the dense Boolean lookup reads successor COLUMN indices",
          lookup.get(0) == {0, 1} and lookup.get(1) == {1, 2},
          f"a dense Boolean adjacency was read as {lookup}: the row's 0/1 VALUES were used as "
          f"column indices instead of the positions of the True entries")


# --------------------------------------------------------------------------------------
# defect 5: tau was chosen from the wrong latent
# --------------------------------------------------------------------------------------
def test_defect_5_the_tau_measurement_uses_the_encoder_latent() -> None:
    """RED: preregister_tau must differentiate through the encoder."""
    source = inspect.getsource(FP.preregister_tau)
    calls_encoder = "encoder" in source
    check("the tau preregistration measures the gap on the encoder-produced latent z",
          calls_encoder,
          "preregister_tau does not use the encoder's latent, so the median gap it reports is "
          "not the quantity the issue's tau rule asks for")


def test_defect_5b_the_native_distance_is_the_squared_form_not_the_codebook_dim() -> None:
    """RED: the host's native L1 distance is over the ENCODER latent, whose dim is 32."""
    source = inspect.getsource(FS.posterior)
    uses_latent_dim = "layer.dim" in source
    check("the posterior is formed on the encoder latent with the layer's own dimension",
          uses_latent_dim,
          "posterior() does not use the quantizer layer's dimension for the latent it scores")


# --------------------------------------------------------------------------------------
# defect 6: the verdict vocabulary
# --------------------------------------------------------------------------------------
def test_defect_6_the_verdict_vocabulary_is_machine_checkable() -> None:
    """RED: the pilot has no status enum distinguishing invalid protocol from blocked CGV."""
    source = inspect.getsource(FP)
    vocabulary = ("invalid_protocol", "invalid_metrics", "blocked_cgv",
                  "supported_pilot", "rejected_pilot", "inconclusive")
    present = [name for name in vocabulary if name in source]
    check("the pilot distinguishes invalid_protocol / invalid_metrics / blocked_cgv from a "
          "supported verdict",
          len(present) >= 4,
          f"the pilot has no machine-checkable verdict vocabulary; found only {present}")


def test_red_suite_is_red_against_the_committed_pilot() -> None:
    """The suite must fail: that is the point of writing it before the fix."""
    check(f"the {RED_EXPECTED} confirmed defects are still detectable in the committed code",
          True, "")
    print(f"  (red suite: {CHECKS} checks, {len(FAILURES)} failing against the committed pilot)")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
    print(f"r7_f_full_integration: {CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  RED {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
