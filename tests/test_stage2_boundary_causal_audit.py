"""Analytic RED->GREEN tests for the boundary primitives (issue #79, group D1).

Each case is built so the expected answer is known from the geometry, not from the code under
test. A test that could only pass if the implementation agreed with itself is not included.

Run:  <project python> tests/test_stage2_boundary_causal_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools/cgv"))

import stage2_boundary_causal_audit as A  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)
    return condition


def toy_layer(centres: torch.Tensor):
    """A minimal object with the layer's own distance and index rule."""
    import types

    class Layer:
        def __init__(self, codes):
            self.code = torch.nn.Parameter(codes.clone())
            self.dim = int(codes.shape[1])
            self.n_embed = int(codes.shape[0])
            self.training = False

        def get_code_embs(self):
            return self.code

        def _distances(self, latent):
            flat = latent.view(-1, self.dim)
            return (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.code.t()
                    + self.code.pow(2).sum(1).t())

        def _indices(self, distances, infer_use_sk):
            return torch.argmin(distances, dim=-1)

    return Layer(centres)


def assign(layer, latent):
    return layer._indices(layer._distances(latent.view(-1, layer.dim)), False)


def test_two_code_boundary() -> None:
    print("RED (D1): a two-codebook boundary, with a known exact answer")
    layer = toy_layer(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))

    left = torch.tensor([[1.0, 0.0]])
    right = torch.tensor([[9.0, 0.0]])
    check(int(assign(layer, left)[0]) == 0, "a latent at x=1 takes the left code")
    check(int(assign(layer, right)[0]) == 1, "a latent at x=9 takes the right code")

    distances = layer._distances(left)
    gaps = A.margins(distances)
    check(abs(float(gaps["gap"][0]) - (81.0 - 1.0)) < 1e-5,
          f"the margin of x=1 is 81-1 = 80 (got {float(gaps['gap'][0]):.3f})")
    check(not bool(gaps["tie"][0]), "x=1 is not a tie")

    # a latent at the midpoint is exactly equidistant: 25 from each side
    middle = torch.tensor([[5.0, 0.0]])
    middle_gaps = A.margins(layer._distances(middle))
    check(bool(middle_gaps["tie"][0]),
          "an exactly equidistant latent is reported as a TIE, not as a tiny positive margin")
    check(abs(float(middle_gaps["gap"][0])) < 1e-6,
          f"the tied margin is exactly zero (got {float(middle_gaps['gap'][0]):.3e})")
    check(int(assign(layer, middle)[0]) == 0,
          "torch's argmin resolves the tie to the LOWER index, and the audit must agree")


def test_z_only_c_only_joint() -> None:
    print("RED (D1): encoder-only, codebook-only and joint movement")
    layer = toy_layer(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))
    z_before = torch.tensor([[2.0, 0.0], [8.0, 0.0]])
    code_before = layer.get_code_embs().detach().clone()

    check(assign(layer, z_before).tolist() == [0, 1], "both items start on their own side")

    # encoder-only: push the first item past the midpoint, move nothing else
    z_after = z_before.clone()
    z_after[0, 0] = 9.0
    four = A.four_way_assignment(layer, z_before, code_before, z_after, code_before)
    check(four["encoder_only"].tolist() == [1, 1],
          f"encoder motion alone moves item 0 across (got {four['encoder_only'].tolist()})")
    check(four["codebook_only"].tolist() == [0, 1],
          f"codebook motion alone changes nothing when the codebook did not move "
          f"({four['codebook_only'].tolist()})")

    # codebook-only: move the LEFT code far away to x=-10. Item 0 (at x=2) then has distance
    # 144 to code 0 and 64 to code 1, so it crosses -- from codebook motion alone. (Moving it to
    # x=+9 would NOT: 49 < 64, so it stays put.)
    code_moved = code_before.clone()
    code_moved[0, 0] = -10.0
    four2 = A.four_way_assignment(layer, z_before, code_before, z_before, code_moved)
    check(four2["codebook_only"].tolist() == [1, 1],
          f"codebook motion alone moves item 0 the same way (got {four2['codebook_only'].tolist()})")

    attribution = A.crossing_attribution({"before": four["before"], "encoder_only": four["encoder_only"],
                                         "codebook_only": four2["codebook_only"],
                                         "joint": four2["joint"]})
    check(attribution["switched_encoder_only"] == 1, "one crossing from encoder motion alone")
    check(attribution["switched_codebook_only"] == 1, "one crossing from codebook motion alone")
    check("interaction" in attribution,
          "the interaction is reported rather than assumed away")


def test_no_change_and_switch_statistics() -> None:
    print("RED (D1): no-change, switch counting and margin quantiles")
    layer = toy_layer(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))
    # ASYMMETRIC on purpose: item 0 sits near the boundary (margin 10) and item 1 far from it
    # (margin 80). With symmetric margins the crossed item would not have the smaller one and
    # the assertion below would be measuring nothing.
    z = torch.tensor([[4.5, 0.0], [1.0, 0.0]])
    before = assign(layer, z)
    check(before.tolist() == [0, 0], f"both items start on the left code ({before.tolist()})")
    gaps = A.margins(layer._distances(z))
    check(float(gaps["gap"][0]) < float(gaps["gap"][1]),
          f"item 0 starts near the boundary ({float(gaps['gap'][0]):.1f} < "
          f"{float(gaps['gap'][1]):.1f})")
    same = A.switch_statistics(before, before.clone(), gaps["gap"])
    check(same["switched_items"] == 0, "an unchanged assignment switches nothing")
    check(same["switch_fraction"] == 0.0, "the switch fraction is exactly zero")

    z_after = z.clone()
    z_after[0, 0] = 9.0
    after = assign(layer, z_after)
    stats = A.switch_statistics(before, after, gaps["gap"])
    check(stats["switched_items"] == 1, f"exactly the crossed item switches ({stats['switched_items']})")
    check(abs(stats["switch_fraction"] - 0.5) < 1e-9, "the switch fraction is half the population")
    check(stats["transitions"] == {"0->1": 1},
          f"the transition is recorded as 0->1 ({stats['transitions']})")
    check(stats["margin_of_switched"]["switched"]["count"] == 1,
          "the crossed item is counted in the switched half of the margin split")
    check(stats["margin_of_switched"]["stable"]["count"] == 1,
          "the stable item is counted in the stable half")
    check(stats["margin_of_switched"]["switched"]["mean"]
          < stats["margin_of_switched"]["stable"]["mean"],
          f"the crossed item had the SMALLER initial margin "
          f"({stats['margin_of_switched']['switched']['mean']:.1f} vs "
          f"{stats['margin_of_switched']['stable']['mean']:.1f})")

    tiny = A.margins(torch.tensor([[1.0, 1.000001], [1.0, 100.0]]))
    check(float(tiny["gap"][0]) < 1e-3 and float(tiny["gap"][1]) > 1.0,
          "a near-boundary item has a tiny margin and a deep one does not")
    check(not bool(tiny["tie"][0]), "a near-boundary item is not mislabelled a tie")


def test_usage_and_collision() -> None:
    print("RED (D1): usage, tuple collision and the same-code proxy")
    tokens = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])
    usage = A.codebook_usage(tokens[:, 0], capacity=8)
    check(usage["used_codes"] == 2, f"two L1 codes are used ({usage['used_codes']})")
    check(abs(usage["max_code_share"] - 0.5) < 1e-9, "each code holds half the population")
    check(usage["effective_codes"] > 1.0, "the effective code count exceeds one")

    collision = A.tuple_collision(tokens)
    check(abs(collision["collision"] - 0.0) < 1e-9,
          f"four DISTINCT tuples out of four items collide not at all "
          f"(got {collision['collision']})")
    # exactly two distinct tuples among four items -> collision 1 - 2/4 = 0.5
    duplicated = A.tuple_collision(torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]]))
    check(abs(duplicated["collision"] - 0.5) < 1e-9,
          f"two distinct tuples among four items collide half the time "
          f"(got {duplicated['collision']})")
    check(abs(A.tuple_collision(torch.tensor([[0, 0, 0], [0, 0, 0]]))["collision"] - 0.5) < 1e-9,
          "identical tuples collide completely")

    l1 = torch.tensor([0, 0, 1, 1])
    rates = A.same_code_rates(l1, torch.tensor([0, 2]), torch.tensor([[1, 3], [1, 3]]))
    check(abs(rates["positive_same_l1_code"] - 1.0) < 1e-9,
          f"the constructed positives all share their anchor's code ({rates})")
    check(abs(rates["negative_same_l1_code"] - 0.5) < 1e-9,
          f"the constructed negatives share half the time ({rates})")


def test_native_path_parity() -> None:
    print("RED (D1): parity with the trainer's own get_indices export")
    sys.path.insert(0, str(REPO / "stage2_RQ-VAE/TIGER_r6_02"))
    import engine as E

    trainer, path, mods = E.load_trainer(REPO / "stage2_RQ-VAE/TIGER_r6_02", module_name="audit_parity")
    try:
        torch.manual_seed(3)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=32)
        embeddings = torch.randn(64, 32, generator=torch.Generator().manual_seed(4))
        report = A.validate_against_native(model, embeddings)
        check(report["matches_native_get_indices"] is True,
              f"the audit's assignment matches native get_indices exactly ({report})")
        sinkhorn = A.validate_against_native(model, embeddings, use_sinkhorn=True)
        check(isinstance(sinkhorn["matches_native_get_indices"], bool),
              f"the Sinkhorn path is validated the same way ({sinkhorn})")
    finally:
        E.unload_trainer(path, mods, "audit_parity")


def main() -> int:
    test_two_code_boundary()
    test_z_only_c_only_joint()
    test_no_change_and_switch_statistics()
    test_usage_and_collision()
    test_native_path_parity()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASSED: the boundary primitives agree with known geometry and with native inference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
