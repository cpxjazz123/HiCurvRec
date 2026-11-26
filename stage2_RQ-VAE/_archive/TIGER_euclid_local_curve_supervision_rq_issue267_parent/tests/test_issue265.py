"""Issue #265 focused tests: Euclidean selection and center-local geometry."""
import torch
from model.geometry import euclidean_cost, mixed_cost


def test_euclidean_selection_cost_has_no_gate_channel():
    x = torch.tensor([[0.2, -1.0], [0.5, 0.3]])
    y = torch.tensor([[0.0, 0.0], [1.0, -0.5], [-0.2, 0.7]])
    direct = euclidean_cost(x, y, scale=1.0)
    expected = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    assert torch.allclose(direct, expected)
    # The same Euclidean table is independent of any expert weights.
    w1 = torch.tensor([1.0, 0.0, 0.0])
    w2 = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(mixed_cost(x, y, w1, curvature_on=False), expected)
    assert torch.allclose(mixed_cost(x, y, w2, curvature_on=False), expected)


def test_local_mixed_distance_uses_one_center_weight():
    z = torch.tensor([[0.2, -1.0], [0.5, 0.3], [0.0, 0.0]])
    w = torch.tensor([0.2, 0.5, 0.3])
    first = mixed_cost(z, z, w, curvature_on=True)
    second = mixed_cost(z, z, w, curvature_on=True)
    assert torch.equal(first, second)
    assert first.shape == (3, 3)
    assert torch.allclose(first, first.T)


def test_local_distance_is_not_a_gate_selected_codeword_table():
    x = torch.tensor([[0.3, 0.1]])
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    w_a = torch.tensor([1.0, 0.0, 0.0])
    w_b = torch.tensor([0.0, 0.0, 1.0])
    a = mixed_cost(x, y, w_a, curvature_on=True)
    b = mixed_cost(x, y, w_b, curvature_on=True)
    assert not torch.allclose(a, b), "expert weights must change the structural ruler"
