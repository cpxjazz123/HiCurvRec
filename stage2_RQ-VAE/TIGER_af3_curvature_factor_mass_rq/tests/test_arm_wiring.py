"""The arm's end-to-end wiring, written against the `math.json` statements.

Groups:

* `chart_head` reaches the quantiser: `RQVAE.forward` hands over the charted product
  point `(r^E, x_hi, x_lo)`, the hyperbolic blocks are REUSED at every level, and only
  the Euclidean block carries the residual;
* `p_i = u_i` per item: every entry point (`forward`, `get_indices`, `init_codebook`)
  indexes the frozen table with the item ids and nothing else;
* `init_codebook` takes its argmin from the SAME `D` as the assignment;
* `geometry_off` is elementwise the frozen Euclidean quantiser (the degeneracy test);
* `behaviour_off` pins `p` to `SHARE_REF` while keeping the chart (it must NOT equal the
  baseline: `math.json` `switches`);
* the trainer selects the mechanism distance by name, reads the frozen artefact and
  refuses a missing / incomplete table.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import factor_share_geometry as geometry
import train_rqvae
from distances import distance as distance_module
from distances import factor_share as adapter
from model.layers import RQLayer, VQLayer
from model.model import RQVAE


def tokenizer_config(**overrides):
    values = dict(
        hidden_sizes=[8, 4],
        codebook_num=2,
        codebook_size=(6, 6),
        codebook_dim=32,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=-1.0,   # argmin everywhere: the assignment must be readable
        sk_iters=-1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class Recording:
    """A metric that records its arguments and then runs the real registry entry."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def __call__(self, latent, codebook, p=None, mode="both_on"):
        self.calls.append({
            "latent": latent,
            "p": None if p is None else torch.as_tensor(p).detach().clone(),
            "mode": mode,
        })
        return self.inner(latent, codebook, p, mode)


SHARES = torch.tensor([0.05, 0.5, 0.95, 0.3, 0.7, 0.2], dtype=torch.float32)


# --------------------------------------------------------------------------------------
# chart_head: the model hands over the charted product point, once per level
# --------------------------------------------------------------------------------------

def test_forward_hands_the_quantiser_the_charted_product_point():
    torch.manual_seed(0)
    recorder = Recording(adapter.distances)
    model = RQVAE(tokenizer_config(), in_dim=32, distance=recorder, shares=SHARES)
    model.eval()
    embeddings = torch.randn(3, 32) * 0.1
    item_ids = torch.tensor([5, 0, 3])
    with torch.no_grad():
        model(embeddings, item_ids)

    assert len(recorder.calls) == model.rq.codebook_num
    encoded = model.encoder(embeddings)
    x_hi, x_lo = geometry.item_chart_head(encoded)
    for call in recorder.calls:
        rE, hi, lo = call["latent"]
        # the two hyperbolic blocks are the SAME charted product point at every level
        assert torch.equal(hi, x_hi) and torch.equal(lo, x_lo)
        assert (rE.shape, hi.shape, lo.shape) == ((3, 16), (3, 8), (3, 8))
        # p_i = u_i, indexed per item; nothing else feeds the share
        assert torch.allclose(call["p"], SHARES[item_ids])
        assert call["mode"] == "both_on"

    # r^{E,0} = h_E and only the Euclidean block recurses
    assert torch.equal(recorder.calls[0]["latent"][0], encoded[:, :16])
    assert not torch.allclose(recorder.calls[0]["latent"][0],
                             recorder.calls[1]["latent"][0])


def test_hyperbolic_blocks_are_not_part_of_the_residual_chain():
    """A quantiser that subtracted the codeword from all 32 dims would drift `x_hi`."""
    torch.manual_seed(1)
    recorder = Recording(adapter.distances)
    layer = RQLayer(tokenizer_config(codebook_num=3, codebook_size=(6, 6, 6)),
                    distance=recorder)
    layer.eval()
    latent = torch.randn(4, geometry.D_TOTAL) * 0.2
    p = torch.full((4,), 0.4)
    with torch.no_grad():
        layer(latent, p=p)
    hi = [call["latent"][1] for call in recorder.calls]
    lo = [call["latent"][2] for call in recorder.calls]
    assert all(torch.equal(block, hi[0]) for block in hi)
    assert all(torch.equal(block, lo[0]) for block in lo)


# --------------------------------------------------------------------------------------
# behaviour_off: p pinned to SHARE_REF, the geometry kept
# --------------------------------------------------------------------------------------

def test_behaviour_off_pins_the_share_and_keeps_the_chart():
    torch.manual_seed(0)
    recorder = Recording(adapter.distances)
    model = RQVAE(tokenizer_config(), in_dim=32, distance=recorder,
                  shares=SHARES, share_mode="behaviour_off")
    model.eval()
    embeddings = torch.randn(3, 32) * 0.1
    with torch.no_grad():
        model(embeddings, torch.tensor([5, 0, 3]))
    for call in recorder.calls:
        assert torch.allclose(call["p"], torch.full((3,), geometry.SHARE_REF))
        assert call["latent"][1] is not None and call["latent"][2] is not None
    # behaviour_off is NOT the baseline: the product cost still differs from the
    # Euclidean one, which is exactly what `math.json` says the switch keeps.
    latent = torch.randn(3, geometry.D_TOTAL) * 0.1
    x_hi, x_lo = geometry.item_chart_head(latent)
    blocks = (latent[:, :16], x_hi, x_lo)
    p = SHARES[:3]
    on = adapter.distances(blocks, torch.randn(6, 32) * 0.1, p, "both_on")
    off = adapter.distances(blocks, torch.randn(6, 32) * 0.1, p, "behaviour_off")
    assert not torch.allclose(on, off)


# --------------------------------------------------------------------------------------
# geometry_off: elementwise the frozen baseline quantiser
# --------------------------------------------------------------------------------------

def test_geometry_off_reproduces_the_baseline_layer():
    torch.manual_seed(0)
    config = tokenizer_config()
    baseline = RQLayer(config, distance=distance_module.distances)
    candidate = RQLayer(config, distance=adapter.distances)
    candidate.load_state_dict(baseline.state_dict())
    baseline.eval()
    candidate.eval()
    latent = torch.randn(8, geometry.D_TOTAL) * 0.3
    p = torch.linspace(0.1, 0.9, 8)
    with torch.no_grad():
        expected = baseline(latent)
        got = candidate(latent, p=p, mode="geometry_off")
    # same codewords ...
    assert torch.equal(expected[3], got[3])
    assert torch.allclose(expected[0], got[0], rtol=0, atol=1e-6)
    # ... and still the baseline's two mean squared errors, not the geometric ones
    for (expected_parts, got_parts) in zip(expected[1], got[1]):
        assert torch.allclose(expected_parts[0], got_parts[0], rtol=0, atol=1e-6)
        assert torch.allclose(expected_parts[1], got_parts[1], rtol=0, atol=1e-6)


# --------------------------------------------------------------------------------------
# init_codebook: same D for the argmin, and the same p
# --------------------------------------------------------------------------------------

def test_init_codebook_takes_its_argmin_from_the_same_distance():
    torch.manual_seed(0)
    recorder = Recording(adapter.distances)
    layer = VQLayer(codebook_size=4, codebook_dim=geometry.D_TOTAL, distance=recorder)
    latent = torch.randn(6, geometry.D_TOTAL) * 0.2
    p = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.9])
    residual = layer.init_codebook(latent, torch.device("cpu"), p=p, mode="both_on")
    assert recorder.calls, "init_codebook did not call the injected metric"
    assert recorder.calls[-1]["mode"] == "both_on"
    assert torch.allclose(recorder.calls[-1]["p"], p)
    # the residual is built from the PRODUCT argmin over the freshly initialized codebook
    x_hi, x_lo = geometry.item_chart_head(latent)
    distances = adapter.distances((latent[:, :16], x_hi, x_lo),
                                  layer.get_code_embs(), p, "both_on")
    expected = latent - layer.embed_code(distances.argmin(dim=-1))
    assert torch.allclose(residual, expected)


def test_init_codebook_refuses_a_missing_share():
    layer = VQLayer(codebook_size=4, codebook_dim=geometry.D_TOTAL, distance=adapter.distances)
    with pytest.raises(ValueError):
        layer.init_codebook(torch.randn(6, geometry.D_TOTAL), torch.device("cpu"))


# --------------------------------------------------------------------------------------
# the per-item share: every entry point, and it can never be defaulted
# --------------------------------------------------------------------------------------

def test_model_entry_points_refuse_a_missing_table_or_a_foreign_id():
    model = RQVAE(tokenizer_config(), in_dim=32, distance=adapter.distances, shares=SHARES)
    embeddings = torch.randn(4, 32)
    with pytest.raises(ValueError):
        model(embeddings)
    with pytest.raises(ValueError):
        model.get_indices(embeddings)
    with pytest.raises(ValueError):
        model.init_codebook(embeddings)
    with pytest.raises(ValueError):
        model(embeddings, torch.tensor([0, 1, 2, 6]))  # 6 is outside the table
    bare = RQVAE(tokenizer_config(), in_dim=32, distance=adapter.distances)
    with pytest.raises(ValueError):
        bare(embeddings, torch.tensor([0, 1, 2, 3]))


def test_item_ids_reach_the_share_for_get_indices_and_init_codebook():
    torch.manual_seed(0)
    recorder = Recording(adapter.distances)
    model = RQVAE(tokenizer_config(), in_dim=32, distance=recorder, shares=SHARES)
    model.eval()
    embeddings = torch.randn(8, 32) * 0.1
    item_ids = torch.tensor([4, 1, 0, 5, 2, 3, 5, 4])
    with torch.no_grad():
        model.init_codebook(embeddings, item_ids)
        recorder.calls.clear()
        model.get_indices(embeddings, item_ids)
    assert recorder.calls and torch.allclose(recorder.calls[0]["p"], SHARES[item_ids])


def test_unknown_switch_and_non_vq_type_are_refused():
    with pytest.raises(ValueError):
        RQVAE(tokenizer_config(), in_dim=32, distance=adapter.distances, share_mode="half_on")
    for vq_type in ("ema", "simvq", "nonsense"):
        with pytest.raises(ValueError):
            RQLayer(tokenizer_config(vq_type=vq_type), distance=adapter.distances)


# --------------------------------------------------------------------------------------
# the trainer: mechanism distance by name, frozen artefact read-only
# --------------------------------------------------------------------------------------

def test_train_selects_the_mechanism_distance_by_name():
    assert train_rqvae.DISTANCE is adapter.distances
    assert train_rqvae.DISTANCE_NAME == "factor_share"
    assert train_rqvae.SHARE_MODE == "both_on"
    assert train_rqvae.POTENTIAL_FILE.name == "af3_item_potential.npz"
    assert train_rqvae.POTENTIAL_FILE.parent.name == "eval"
    assert (train_rqvae.POTENTIAL_FILE.parent.parent
            == train_rqvae.REPO_ROOT / "stage2_RQ-VAE" / train_rqvae.ARM_NAME)


def test_train_never_recomputes_the_table_and_fails_loud_when_it_is_missing(tmp_path):
    source = inspect.getsource(train_rqvae)
    assert "build_table" not in source and "build_and_write" not in source
    with pytest.raises(RuntimeError):
        train_rqvae.load_frozen_table(tmp_path / "absent.npz")


def _table(items: int = 4) -> dict:
    rank = np.array([3.5, 1.5, 1.5, 3.5][:items], dtype=np.float64)
    return {
        "item_ids": np.arange(items, dtype=np.int64),
        "potential": np.zeros(items, dtype=np.float64),
        "u": (rank - 0.5) / float(items),
        "rank": rank,
        "is_missing": np.zeros(items, dtype=bool),
        "rank_of_graph_mean": np.asarray(2.5, dtype=np.float64),
    }


def _table_with_shares(u, potential=None) -> dict:
    """A table whose share column is exactly `u` (rank is only the inverse map)."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    items = int(u.size)
    return {
        "item_ids": np.arange(items, dtype=np.int64),
        "potential": (np.zeros(items, dtype=np.float64) if potential is None
                      else np.asarray(potential, dtype=np.float64).reshape(-1)),
        "u": u,
        "rank": u * float(items) + 0.5,
        "is_missing": np.zeros(items, dtype=bool),
        "rank_of_graph_mean": np.asarray(0.5 * (items + 1), dtype=np.float64),
    }


def test_shares_for_indexes_by_item_id_and_refuses_a_missing_id():
    table = _table()
    got = train_rqvae.shares_for([2, 0], table)
    expected = torch.from_numpy(table["u"])[torch.tensor([2, 0])]
    assert torch.equal(got, expected.to(torch.float32))
    with pytest.raises(RuntimeError):
        train_rqvae.shares_for([0, 9], table)


def test_potential_report_reads_the_same_vector_the_quantiser_gets():
    table = _table()
    share = train_rqvae.frozen_share(table)
    report = train_rqvae.potential_report(table, share)
    u = table["u"]
    assert report["items"] == 4
    # p is the vector the quantiser gets (share() applied once), not a re-sorted copy
    assert report["p_u_spearman"] == pytest.approx(1.0)
    # this table's potential column is constant, so there is nothing to correlate with
    assert report["p_potential_spearman"] is None
    assert "energy_rel_error_max" not in report
    assert report["rank_of_graph_mean"] == pytest.approx(2.5)
    assert report["p_miss"] == pytest.approx(0.5)
    assert report["p_miss_minus_share_ref"] == pytest.approx(0.0)
    assert report["u_min"] == pytest.approx(float(u.min()))
    assert report["p_percentiles"]["50"] == pytest.approx(float(np.percentile(u, 50)))


def test_p_miss_readout_follows_the_share_reference_constant(monkeypatch):
    """`|p_miss - SHARE_REF|` (math.json `metrics`) must read SHARE_REF, not a literal 0.5.

    On the old code (`abs(p_miss - 0.5)`) the second half of this test fails: moving the
    constant would leave the reading untouched.
    """
    table = _table()
    report = train_rqvae.potential_report(table, train_rqvae.frozen_share(table))
    assert geometry.SHARE_REF == 0.5
    assert report["p_miss_minus_share_ref"] == pytest.approx(
        abs(report["p_miss"] - geometry.SHARE_REF))
    monkeypatch.setattr(geometry, "SHARE_REF", 0.25)
    moved = train_rqvae.potential_report(table, train_rqvae.frozen_share(table))
    assert moved["p_miss_minus_share_ref"] == pytest.approx(abs(moved["p_miss"] - 0.25))
    assert moved["p_miss_minus_share_ref"] != pytest.approx(report["p_miss_minus_share_ref"])


def test_frozen_share_is_one_share_call_and_the_identity_u_to_p():
    table = _table()
    share = train_rqvae.frozen_share(table)
    # the whole table, in table order: the model indexes it by item id
    assert share.shape == (int(table["u"].size),)
    assert torch.allclose(share, torch.from_numpy(table["u"]).to(torch.float32))
    source = inspect.getsource(train_rqvae.frozen_share)
    assert "factor_share_geometry.share" in source


def test_potential_report_notices_a_share_that_is_not_the_mapped_one():
    table = _table_with_shares([0.1, 0.2, 0.7, 0.9], potential=[1.0, 2.0, 3.0, 4.0])
    good = train_rqvae.potential_report(table, train_rqvae.frozen_share(table))
    assert good["p_u_spearman"] == pytest.approx(1.0)
    assert good["p_potential_spearman"] == pytest.approx(1.0)
    # the statistic is not constant: a share that is not ordered like the table shows it
    scrambled = torch.tensor([0.1, 0.9, 0.2, 0.7], dtype=torch.float32)
    bad = train_rqvae.potential_report(table, scrambled)
    assert bad["p_u_spearman"] < 0.99
    assert bad["p_potential_spearman"] < 0.99


def test_energy_conservation_is_measured_not_assumed():
    torch.manual_seed(0)
    u = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    table = _table_with_shares(u)
    flipped = _table_with_shares(1.0 - u)
    ids = torch.arange(u.size, dtype=torch.long)

    hidden = torch.randn(u.size, geometry.D_TOTAL) * 0.05
    report = train_rqvae.energy_conservation_report(table, hidden, ids)
    assert report["items"] == int(u.size)
    assert report["energy_rel_error_max"] < 1e-5      # product side is not rescaled
    assert report["energy_clipped"] == 0
    assert isinstance(report["energy_p_corr"], float)

    # the same readings on the same rows detect a chart that left its ball: the
    # number is a measurement, it is not 0 by construction.
    far = hidden * 40.0
    assert train_rqvae.energy_conservation_report(table, far, ids)["energy_rel_error_max"] > 1e-6

    # Spearman(p, ||x_i||) is read, so p -> 1 - p must flip its sign
    other = train_rqvae.energy_conservation_report(flipped, hidden, ids)["energy_p_corr"]
    assert other == pytest.approx(-report["energy_p_corr"], abs=1e-12)


def test_energy_conservation_refuses_a_mismatched_batch():
    table = _table_with_shares([0.2, 0.4, 0.6])
    with pytest.raises(ValueError):
        train_rqvae.energy_conservation_report(
            table, torch.randn(3, geometry.D_TOTAL), torch.tensor([0, 1]))
    with pytest.raises(RuntimeError):
        train_rqvae.energy_conservation_report(
            table, torch.randn(2, geometry.D_TOTAL), torch.tensor([0, 3]))
