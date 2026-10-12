import numpy as np

from geometry import (
    auc_from_scores,
    ball_map,
    ordering_fractions,
    pairwise_poincare,
    score_variants,
)


def test_monotone_auc_invariance():
    pos = np.array([0.5, 2.0, 4.5, 7.0])
    neg = np.array([1.0, 3.0, 5.0, 8.0])
    transformed = {
        "raw": pos,
        "rms": pos / 2.3,
        "matched": pos * 4.7,
        "log1p": np.log1p(pos),
    }
    reference = auc_from_scores(pos, neg)
    assert all(auc_from_scores(p, {
        "raw": neg,
        "rms": neg / 2.3,
        "matched": neg * 4.7,
        "log1p": np.log1p(neg),
    }[name]) == reference for name, p in transformed.items())


def test_score_variants_bit_identical_auc():
    pos = np.array([0.5, 2.0, 4.5, 7.0])
    neg = np.array([1.0, 3.0, 5.0, 8.0])
    scores = score_variants(pos, neg, rms=2.3, global_scale=4.7)
    assert scores["raw"]["auc"] == scores["rms_normalized"]["auc"]
    assert scores["raw"]["auc"] == scores["matched_euclidean"]["auc"]
    assert scores["raw"]["auc"] == scores["log1p_euclidean"]["auc"]



def test_auc_positive_closer_direction_and_ties():
    assert auc_from_scores(np.array([1.0]), np.array([3.0])) == 1.0
    assert auc_from_scores(np.array([3.0]), np.array([1.0])) == 0.0
    assert auc_from_scores(np.array([2.0]), np.array([2.0])) == 0.5

def test_poincare_finite_symmetric_diagonal():
    points = np.array([[0.0, 0.0], [0.2, -0.1], [-0.15, 0.25]])
    ball, _ = ball_map(points, c=1.0, tangent_scale=1.0)
    distances = pairwise_poincare(ball, ball, c=1.0)
    assert np.isfinite(distances).all()
    assert np.allclose(distances, distances.T, rtol=0, atol=1e-7)
    assert np.allclose(np.diag(distances), 0.0, rtol=0, atol=1e-7)


def test_ball_map_strict_interior_and_reports_outside():
    points = np.array([[0.0, 0.0], [1.0e6, 0.0]])
    for c in (0.25, 0.5, 1.0, 2.0, 4.0):
        ball, report = ball_map(points, c=c, tangent_scale=1.0)
        assert np.all(np.sqrt(c) * np.linalg.norm(ball, axis=-1) < 1.0)
        assert report["would_leave_ball_count"] == 0
        raw_ball = np.array([[1.1 / np.sqrt(c), 0.0]])
        _, outside_report = ball_map(raw_ball, c=c, tangent_scale=1.0, already_mapped=True)
        assert outside_report["would_leave_ball_count"] == 1


def test_poincare_can_reorder_constructed_case():
    # The point at radius .99 is close to a positive on the far side of the
    # Euclidean chart, but its geodesic path is much longer than the central pair.
    points = np.array([[0.99, 0.0], [-0.99, 0.0], [0.0, 0.0], [0.1, 0.0]])
    ball, _ = ball_map(points, c=1.0, tangent_scale=1.0, already_mapped=True)
    e_pos = np.array([np.linalg.norm(points[0] - points[2])])
    e_neg = np.array([np.linalg.norm(points[0] - points[1])])
    h_pos = pairwise_poincare(ball[[0]], ball[[2]], 1.0)[0, 0]
    h_neg = pairwise_poincare(ball[[0]], ball[[1]], 1.0)[0, 0]
    assert e_pos < e_neg
    assert h_pos < h_neg
    # Non-vacuity: a deliberately misordered pair is corrected by hyperbolic distance.
    anchor = np.array([[0.9, 0.0]])
    positive = np.array([[0.7, 0.0]])
    negative = np.array([[0.9, 0.15]])
    euclidean_wrong = np.linalg.norm(anchor - positive) > np.linalg.norm(anchor - negative)
    mapped, _ = ball_map(np.concatenate([anchor, positive, negative]), 1.0, 1.0, already_mapped=True)
    hyperbolic_correct = pairwise_poincare(mapped[:1], mapped[1:2], 1.0)[0, 0] < pairwise_poincare(mapped[:1], mapped[2:], 1.0)[0, 0]
    assert euclidean_wrong and hyperbolic_correct


def test_ordering_fraction_complement():
    fractions = ordering_fractions(np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 3.0]))
    assert np.isclose(fractions["concordant_fraction"] + fractions["discordant_fraction"], 1.0 - fractions["tie_fraction"])


if __name__ == "__main__":
    tests = (
        ("T1 monotone AUC invariance", test_monotone_auc_invariance),
        ("T1 variant AUC identity", test_score_variants_bit_identical_auc),
        ("T1 AUC direction and ties", test_auc_positive_closer_direction_and_ties),
        ("T2 Poincare finite/symmetric/diagonal", test_poincare_finite_symmetric_diagonal),
        ("T3 strict ball interior and overflow report", test_ball_map_strict_interior_and_reports_outside),
        ("T4 constructed Poincare reordering", test_poincare_can_reorder_constructed_case),
        ("T5 ordering fractions", test_ordering_fraction_complement),
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"All {len(tests)} geometry tests passed.")
