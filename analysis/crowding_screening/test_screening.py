import numpy as np

from screening_metrics import classify_arm, gini_coefficient, local_purity

from screening import load_compatible_state


def test_extra_checkpoint_keys_do_not_change_screening_metrics():
    class SmallModel:
        def __init__(self):
            self.weights = {"weight": np.array([1.0, 2.0])}

        def state_dict(self):
            return self.weights

        def load_state_dict(self, state, strict):
            assert strict
            self.weights = {key: np.array(value, copy=True) for key, value in state.items()}

        def metric(self, values):
            return float(np.mean(np.asarray(values) @ self.weights["weight"]))

    values = np.array([[1.0, 0.0], [0.0, 1.0]])
    checkpoint = {"weight": np.array([3.0, 4.0]), "directional_heads.f_source.weight": np.array([99.0])}
    stripped = {"weight": checkpoint["weight"]}
    with_extra = SmallModel()
    without_extra = SmallModel()
    load_compatible_state(with_extra, checkpoint)
    load_compatible_state(without_extra, stripped)
    assert with_extra.metric(values) == without_extra.metric(values)


def test_purity_is_one_when_every_neighbour_is_related():
    vectors = np.array([[0.0], [1.0], [2.0], [3.0]])
    related = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(related, False)
    result = local_purity(vectors, related, np.ones(4), k=2, seed=42, null_draws=20)
    assert result["purity"] == 1.0

def test_random_neighbours_match_popularity_matched_null():
    vectors = np.arange(8, dtype=float)[:, None]
    related = np.zeros((8, 8), dtype=bool)
    popularity = np.ones(8)
    result = local_purity(vectors, related, popularity, k=2, seed=42, null_draws=40)
    assert result["purity"] == result["null_purity"] == 0.0


def test_codebook_entropy_and_gini():
    uniform = np.ones(256)
    concentrated = np.zeros(256)
    concentrated[0] = 100
    assert gini_coefficient(uniform, counts=True) == 0.0
    assert gini_coefficient(concentrated, counts=True) > 0.99
    from screening_metrics import occupancy_stats
    uniform_stats = occupancy_stats(np.repeat(np.arange(4), 5))
    concentrated_stats = occupancy_stats(np.zeros(20, dtype=int))
    assert uniform_stats["entropy"] == 2.0
    assert concentrated_stats["entropy"] == 0.0


def _comparison_metrics(**overrides):
    value = {
        "purity_lift": 0.1, "purity_lift_ci": [0.08, 0.12],
        "auc": 0.8, "auc_ci": [0.79, 0.81], "ordering_error": 0.2,
        "ordering_error_ci": [0.19, 0.21], "intrusion": 0.1, "intrusion_ci": [0.09, 0.11],
        "entropy": 4.0, "entropy_ci": [3.9, 4.1], "gini": 0.2, "gini_ci": [0.19, 0.21],
        "collision": 0.1, "collision_ci": [0.09, 0.11],
    }
    value.update(overrides)
    return value


def test_classifier_collapse_and_separation_loss():
    base = _comparison_metrics()
    collapse = classify_arm(_comparison_metrics(purity_lift=0.2, purity_lift_ci=[0.18, 0.22],
                                                 entropy=3.0, entropy_ci=[2.9, 3.1]),
                            base, base, "crowd_posattract_w1")
    loss = classify_arm(_comparison_metrics(auc=0.7, auc_ci=[0.69, 0.71]), base, base, "crowd_posattract_w1")
    assert collapse["classification"] == "COLLAPSE"
    assert loss["classification"] == "SEPARATION_LOSS"


def test_equal_metrics_are_nothing():
    base = _comparison_metrics()
    result = classify_arm(base, base, base, "crowd_posattract_w1")
    assert result["classification"] == "NOTHING"


def test_random_control_matching_lift_is_not_useful():
    base = _comparison_metrics(purity_lift=0.0, purity_lift_ci=[-0.01, 0.01])
    arm = _comparison_metrics(purity_lift=0.1, purity_lift_ci=[0.08, 0.12])
    control = _comparison_metrics(purity_lift=0.1, purity_lift_ci=[0.08, 0.12])
    result = classify_arm(arm, base, control, "crowd_posattract_w1")
    assert result["classification"] != "USEFUL_ORGANISATION"


def test_missing_metric_is_unclassified_with_name():
    base = _comparison_metrics()
    arm = _comparison_metrics()
    arm["auc_ci"] = None
    result = classify_arm(arm, base, base, "crowd_posattract_w1")
    assert result["classification"] == "UNCLASSIFIED"
    assert "auc_ci" in result["missing_metrics"]


def test_bootstrap_resamples_anchor_clusters():
    from screening_metrics import anchor_bootstrap_ci

    values = np.array([0.0, 0.0, 1.0, 1.0])
    anchors = np.array([10, 10, 20, 20])
    ci = anchor_bootstrap_ci(values, anchors, seed=42, replicates=200)
    assert ci[0] <= 0.5 <= ci[1]
    assert ci[0] == 0.0 and ci[1] == 1.0


def test_item_bootstrap_intervals_match_collision_and_gini_estimands():
    from screening_metrics import item_bootstrap_ci, occupancy_stats

    codes = np.array([0, 0, 1, 2, 2, 2])
    _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    collision_items = 1.0 - 1.0 / counts[inverse]
    collision_point = float(collision_items.mean())
    gini_point = occupancy_stats(codes, capacity=4)["gini"]
    assert collision_point == 0.5
    assert np.isclose(gini_point, 5 / 12)

    def gini(sample):
        return occupancy_stats(sample, capacity=4)["gini"]

    collision_short = item_bootstrap_ci(collision_items, np.mean, seed=17, replicates=50, bounds=(0, 1))
    collision_long = item_bootstrap_ci(collision_items, np.mean, seed=17, replicates=500, bounds=(0, 1))
    gini_short = item_bootstrap_ci(codes, gini, seed=17, replicates=50, bounds=(0, 1))
    gini_long = item_bootstrap_ci(codes, gini, seed=17, replicates=500, bounds=(0, 1))
    for point, short, long in ((collision_point, collision_short, collision_long),
                               (gini_point, gini_short, gini_long)):
        assert short["ci"][0] <= point <= short["ci"][1]
        assert long["ci"][0] <= point <= long["ci"][1]
        assert long["ci"][1] - long["ci"][0] < short["ci"][1] - short["ci"][0]
