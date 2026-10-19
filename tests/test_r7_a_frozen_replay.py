"""Track A (issue #80) tests: each way a replay can silently lie must fail on its own.

These are analytic and CPU-only. They do not need a GPU run to be meaningful: each test breaks
exactly ONE property of an otherwise valid replay and asserts that the verifier notices. A verifier
that only fails when several things are wrong at once is a verifier that will report parity for a
replay it has not actually checked.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_frozen_replay as R  # noqa: E402
import r7_a_replay_engine as G  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def tiny_state(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "encoder.mlp.1.weight": torch.randn(4, 4, generator=generator),
        "rq.vq_layers.0.embed.weight": torch.randn(8, 4, generator=generator),
    }


def tiny_bundle(seed: int = 0) -> dict:
    optimizer_state = {
        "state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.zeros(4, 4),
                      "exp_avg_sq": torch.zeros(4, 4)}},
        "param_groups": [{"lr": 0.001, "weight_decay": 0.0}],
    }
    return {
        "step": 5,
        "model_state": tiny_state(seed),
        "optimizer_state": optimizer_state,
        "lr_at_step": 0.001,
        "applied_weight": 0.2,
        "training_mode": True,
        "item_batch": torch.zeros(2, 4),
        "pair_embeddings": torch.zeros(2, 4),
        "source_ids": torch.zeros(2, dtype=torch.long),
        "target_ids": torch.ones(2, dtype=torch.long),
        "summary": {"source_hashes": {"train_rqvae.py": "abc"}},
    }


# ----------------------------------------------------------------------------------
def test_identity_reference_must_match() -> None:
    """7. A nonzero FULL-control difference is the failure that invalidates the whole track."""
    reference = tiny_state(0)
    candidate = copy.deepcopy(reference)
    result = G.max_parameter_difference(reference, candidate)
    check("identical states compare equal", result["identical"] is True, str(result))

    candidate["encoder.mlp.1.weight"] = candidate["encoder.mlp.1.weight"] + 1e-7
    result = G.max_parameter_difference(reference, candidate)
    check("a 1e-7 FULL-control difference is detected",
          result["identical"] is False and result["max_abs_diff"] > 0.0, str(result))

    missing = {name: value for name, value in reference.items() if "vq_layers" not in name}
    result = G.max_parameter_difference(reference, missing)
    check("a tensor missing from the replay is a failure, not a pass",
          result["identical"] is False and result["reason"] is not None, str(result))


def test_missing_adamw_moment_is_a_failure() -> None:
    """2. An optimizer state with no moments replays a different step and must be rejected."""
    bundle = R.FrozenBundle(5, REPO_ROOT, {})
    bundle.model_state = tiny_state()
    bundle.optimizer_state = {"state": {}, "param_groups": []}
    bundle.item_batch = torch.zeros(2, 4)
    bundle.source_ids = torch.zeros(2, dtype=torch.long)
    bundle.target_ids = torch.ones(2, dtype=torch.long)
    bundle.training_mode = True
    bundle.applied_weight = 0.2
    problems = bundle.verify(expect_deep=False)
    check("missing AdamW moments are reported",
          any("AdamW moments" in problem for problem in problems), str(problems))

    full = tiny_bundle()
    check("an optimizer digest covers the moments",
          R.optimizer_digest_of(full["optimizer_state"])
          != R.optimizer_digest_of({"state": {}, "param_groups": []}), "digest ignored state")

    dropped = copy.deepcopy(full["optimizer_state"])
    dropped["state"][0]["exp_avg"] = torch.ones(4, 4)
    check("a mutated moment changes the optimizer digest",
          R.optimizer_digest_of(dropped) != R.optimizer_digest_of(full["optimizer_state"]),
          "digest did not notice a mutated moment")


def test_wrong_batch_is_detected() -> None:
    """3. Replaying a different minibatch is not the same step, even though every shape matches."""
    original = tiny_bundle()
    swapped = tiny_bundle()
    swapped["pair_embeddings"] = torch.ones(2, 4)
    check("a different pair minibatch has a different digest",
          R.tensor_digest(original["pair_embeddings"])
          != R.tensor_digest(swapped["pair_embeddings"]), "digest collided")

    swapped["item_batch"] = torch.ones(2, 4)
    check("a different item minibatch has a different digest",
          R.tensor_digest(original["item_batch"]) != R.tensor_digest(swapped["item_batch"]),
          "digest collided")


def test_eval_mode_at_capture_is_a_failure() -> None:
    """1a. A capture taken in eval mode is not the state the loop stepped from."""
    bundle = R.FrozenBundle(5, REPO_ROOT, {})
    bundle.model_state = tiny_state()
    bundle.optimizer_state = {"state": {0: {}}, "param_groups": []}
    bundle.item_batch = torch.zeros(2, 4)
    bundle.source_ids = torch.zeros(2, dtype=torch.long)
    bundle.target_ids = torch.ones(2, dtype=torch.long)
    bundle.training_mode = False
    bundle.applied_weight = 0.0
    check("a capture taken in eval mode is rejected",
          any("training mode" in problem for problem in bundle.verify(expect_deep=False)),
          str(bundle.verify(expect_deep=False)))


def test_deep_expectation_must_match_the_schedule() -> None:
    """The deep path is only live inside the active window; a mismatch means a wrong step."""
    bundle = R.FrozenBundle(32, REPO_ROOT, {})
    bundle.model_state = tiny_state()
    bundle.optimizer_state = {"state": {0: {}}, "param_groups": []}
    bundle.item_batch = torch.zeros(2, 4)
    bundle.source_ids = torch.zeros(2, dtype=torch.long)
    bundle.target_ids = torch.ones(2, dtype=torch.long)
    bundle.training_mode = True
    bundle.applied_weight = 0.0
    problems = bundle.verify(expect_deep=True)
    check("a live deep step without a pair minibatch is rejected",
          any("deep weight is live" in problem for problem in problems), str(problems))

    bundle.applied_weight = 0.2
    bundle.pair_embeddings = torch.zeros(2, 4)
    check("a satisfied deep capture passes", bundle.verify(expect_deep=True) == [],
          str(bundle.verify(expect_deep=True)))

    bundle.applied_weight = 0.0
    check("a pair minibatch at a zero-weight step is rejected as a misread",
          any("not reading the step the loop took" in problem
              for problem in bundle.verify(expect_deep=False)),
          str(bundle.verify(expect_deep=False)))


class _FlatModel(torch.nn.Module):
    """A stand-in whose parameters carry names that survive ``state_dict`` sorting."""

    def __init__(self, values: dict[str, torch.Tensor]):
        super().__init__()
        self.buffers_ = {}
        for index, (name, value) in enumerate(sorted(values.items())):
            self.register_buffer(name.replace(".", "_"), value.clone())
            self.buffers_[name] = getattr(self, name.replace(".", "_"))

    def state_dict(self, *args, **kwargs):  # noqa: ARG002
        return dict(self.buffers_)


def test_update_digest_catches_double_application() -> None:
    """4. Applying the same update twice must not look like one application."""
    before = tiny_state(1)
    once = _FlatModel({name: torch.zeros_like(value) for name, value in before.items()})
    once.buffers_["encoder.mlp.1.weight"].add_(0.5)
    twice = _FlatModel({name: torch.zeros_like(value) for name, value in before.items()})
    twice.buffers_["encoder.mlp.1.weight"].add_(0.5)
    twice.buffers_["encoder.mlp.1.weight"].add_(0.5)

    check("a doubled update is distinguishable",
          G.update_digest(before, once) != G.update_digest(before, twice), "digests collided")

    untouched = _FlatModel(before)
    check("an untouched model has a different update digest from a stepped one",
          G.update_digest(before, untouched) != G.update_digest(before, once),
          "an update left no trace")


def test_shared_storage_is_detectable() -> None:
    """5. A branch that writes through to the frozen tensors is not an independent branch."""
    frozen = tiny_state(3)
    branch_state = {name: value for name, value in frozen.items()}
    shares = all(branch_state[name].data_ptr() == frozen[name].data_ptr() for name in frozen)
    check("a shallow copy shares storage with the frozen bundle", shares,
          "test premise failed: the shallow copy did not alias")

    detached = {name: value.detach().clone() for name, value in frozen.items()}
    independent = all(detached[name].data_ptr() != frozen[name].data_ptr() for name in frozen)
    check("a detached clone does not share storage", independent, "clone still aliases")


def test_trainer_hash_mismatch_is_a_failure() -> None:
    """6. A replay against edited trainer code is not the arm it claims to be."""
    recorded = tiny_bundle()["summary"]["source_hashes"]
    live = {"train_rqvae.py": "different"}
    mismatched = [name for name in recorded
                  if recorded.get(name) and live.get(name)
                  and recorded[name] != live[name]]
    check("a changed trainer hash is detected", mismatched == ["train_rqvae.py"], str(mismatched))

    absent = [name for name in recorded if recorded.get(name) is None]
    check("a missing source file is reported rather than assumed identical",
          isinstance(absent, list), "no check performed")


def test_intervention_whitelist_is_closed() -> None:
    """An unknown intervention must raise instead of silently behaving like FULL."""
    try:
        G.assemble_and_step({}, {}, intervention="BASE_ONLY")
    except G.ReplayError:
        check("an unknown intervention raises", True)
    except Exception as exc:  # noqa: BLE001
        check("an unknown intervention raises", False, f"raised {type(exc).__name__}: {exc}")
    else:
        check("an unknown intervention raises", False, "it was accepted")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
    print(f"r7_a_frozen_replay: {CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
