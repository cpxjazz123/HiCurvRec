"""The floor's guarantee, the patcher's safety, and a run against both real quantisers.

The last two tests are the ones that matter for "works with any mechanism": they copy an
actual tree -- the frozen baseline (Euclidean VQLayer) and the F0 mixed quantiser -- patch
it, and run the patched class in a fresh interpreter on the cost matrix that collapsed every
arm in this investigation (three cheap codewords, five unreachable).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anticollapse import (CodebookCollapse, InfeasibleFloor, ReadOnlyBaseline,  # noqa: E402
                          audit_sid,
                          UnknownTree,
                          apply_floor, constrained_assignment, verify_assignment,
                          verify_codebook, verify_tree)

BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
MIXED = REPO / "results" / "research" / "qda" / "trees" / "F0_seed42"
SCRATCH = REPO / "results" / "research" / "qda" / "anticollapse_check"


def _collapsed(items=4096, codewords=256, live=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand(items, codewords, generator=generator) + 10.0
    cost[:, :live] = torch.rand(items, live, generator=generator)
    return cost


def test_floor_rescues_a_collapsed_cost():
    cost = _collapsed()
    assignment, report = constrained_assignment(cost, 1)
    counts = torch.bincount(assignment, minlength=cost.shape[1])
    assert int(counts.min()) == 1
    assert report["moves"] == 248


def test_floor_is_free_when_the_assignment_already_satisfies_it():
    generator = torch.Generator().manual_seed(1)
    cost = torch.rand(4096, 256, generator=generator)
    assignment, report = constrained_assignment(cost, 1)
    assert report["moves"] == 0 and torch.equal(assignment, torch.argmin(cost, dim=-1))


def test_a_codebook_larger_than_the_batch_falls_back_to_coverage():
    """A mechanism may use more codewords than a batch has items; that must not crash.

    No assignment can cover 256 codewords with 100 items, so the guarantee becomes the
    strongest available one: no two items share a codeword.
    """
    cost = torch.rand(100, 256) + 10.0
    cost[:, :5] = torch.rand(100, 5)
    index, report = constrained_assignment(cost, 1)
    assert report["mode"] == "coverage"
    assert int(torch.unique(index).numel()) == 100
    assert verify_assignment(index, 256, 1)["mode"] == "coverage"
    with pytest.raises(CodebookCollapse):
        verify_assignment(torch.zeros(100, dtype=torch.long), 256, 1)


@pytest.fixture()
def scratch():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    made = []
    def make(name: str, source: Path) -> Path:
        target = SCRATCH / name
        if target.exists():
            shutil.rmtree(target)
        # the frozen baseline ships a dangling dataset symlink; it is not part of the patch
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"),
                        symlinks=True, ignore_dangling_symlinks=True)
        made.append(target)
        return target
    yield make
    for path in made:
        shutil.rmtree(path, ignore_errors=True)


def test_patched_baseline_survives_a_collapsed_cost(scratch):
    tree = scratch("euclid", BASELINE)
    report = apply_floor(tree)
    assert report["changed"] is True and report["family"] == "euclid"
    verdict = verify_tree(tree)
    assert verdict["ok"] is True, verdict
    assert verdict["used"] == verdict["codewords"] == 8


def test_patched_mixed_tree_survives_a_collapsed_cost(scratch):
    tree = scratch("mixed", MIXED)
    report = apply_floor(tree)
    assert report["changed"] is True and report["family"] == "mixed"
    verdict = verify_tree(tree)
    assert verdict["ok"] is True, verdict


def test_patching_is_idempotent(scratch):
    tree = scratch("twice", BASELINE)
    first = apply_floor(tree)
    body = (tree / first["module"]).read_text(encoding="utf-8")
    second = apply_floor(tree)
    assert first["changed"] is True and second["changed"] is False
    assert (tree / first["module"]).read_text(encoding="utf-8") == body


def test_an_unrecognised_quantiser_fails_closed(scratch):
    tree = scratch("unknown", MIXED)
    original = (tree / "model" / "mixed_rq.py").read_text(encoding="utf-8")
    (tree / "model" / "mixed_rq.py").write_text(
        original.replace("return torch.argmin(distances, dim=-1)",
                         "return torch.argmin(distances, dim=-2)"),
        encoding="utf-8")
    with pytest.raises(UnknownTree):
        apply_floor(tree)


def test_copy_baseline_installs_the_floor_and_records_it(scratch):
    """The framework's own candidate-creation path must produce a protected tree."""
    sys.path.insert(0, str(REPO / "tools" / "research"))
    from rounds.baseline import copy_baseline

    destination = SCRATCH / "created_by_framework"
    if destination.exists():
        shutil.rmtree(destination)
    try:
        provenance = copy_baseline(destination)
        assert provenance["assignment_floor"]["changed"] is True
        assert (destination / "assignment_floor.py").is_file()
        assert provenance["candidate_digest"] != provenance["candidate_digest_after_floor"]
        assert verify_tree(destination)["ok"] is True
    finally:
        shutil.rmtree(destination, ignore_errors=True)


def test_the_admission_check_passes_a_protected_tree_and_fails_the_bare_baseline(scratch):
    sys.path.insert(0, str(REPO / "tools" / "research"))
    from stage2 import checks

    protected = scratch("admission", BASELINE)
    apply_floor(protected)
    assert checks.check_assignment_floor(protected)["status"] == "PASS"
    assert checks.check_assignment_floor(BASELINE)["status"] == "FAIL"


def test_a_tree_without_a_quantiser_fails_closed(scratch):
    empty = SCRATCH / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    (empty / "model").mkdir(exist_ok=True)
    try:
        with pytest.raises(UnknownTree):
            apply_floor(empty)
    finally:
        shutil.rmtree(empty, ignore_errors=True)


def test_the_floor_is_checked_on_what_the_quantiser_actually_returned():
    """A mechanism can return indices from anywhere; the claim is checked, not trusted."""
    good = torch.cat([torch.zeros(4, dtype=torch.long), torch.arange(1, 5)])
    assert verify_assignment(good, 5, 1)["min_count"] == 1
    bad = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    with pytest.raises(CodebookCollapse) as error:
        verify_assignment(bad, 4, 1)
    assert "2 of 4 codewords" in str(error.value)


def test_duplicate_codewords_are_refused_and_close_ones_are_only_reported():
    distinct = torch.eye(8)
    report = verify_codebook(distinct)
    assert report["distinct"] == 8 and report["min_distance"] == pytest.approx(2 ** 0.5)
    duplicated = torch.eye(8)
    duplicated[3] = duplicated[0]
    with pytest.raises(CodebookCollapse) as error:
        verify_codebook(duplicated)
    assert "duplicate rows" in str(error.value)
    close = torch.eye(8)
    close[3] = close[0] + 1e-6            # near-duplicates are legitimate in this metric
    assert verify_codebook(close)["distinct"] == 8


def test_the_admission_runs_only_the_real_entry():
    """Section 88: the floor is not a gate -- it is structural, installed before the run."""
    sys.path.insert(0, str(REPO / "tools" / "research"))
    from stage2 import checks
    import inspect

    source = inspect.getsource(checks.run_all)
    assert "checks = [check_entry(tree)]" in source


def test_launch_installs_the_floor_before_the_run(scratch, monkeypatch):
    """`launch` is the only permitted start path, so it is the choke point for the floor."""
    sys.path.insert(0, str(REPO / "tools" / "research"))
    from stage2 import launcher

    tree = scratch("launched", BASELINE)
    assert not (tree / "assignment_floor.py").is_file()      # raw baseline: unprotected

    calls = {}

    class Done:
        returncode = 0

    def fake_run(command, **kwargs):
        calls["patched"] = (tree / "assignment_floor.py").is_file()
        calls["has_floor"] = "constrained_assignment" in (
            tree / "model" / "layers.py").read_text(encoding="utf-8")
        return Done()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher.launch(tree / "train_rqvae.py", tree / "result",
                           python=sys.executable) == 0
    assert calls == {"patched": True, "has_floor": True}
    assert (tree / "result" / "assignment_floor.json").is_file()


DUPLICATE_SCRIPT = """
import torch
from model.layers import VQLayer
layer = VQLayer(8, 4)
layer.eval()
codes = layer.get_code_embs()
with torch.no_grad():
    codes[1] = codes[0]                     # the degenerate table: two codewords equal
x = torch.randn(64, 4)
try:
    layer(x)
except Exception as error:
    print(type(error).__name__ + ": " + str(error)[:80])
else:
    print("NO_ERROR")
"""


def test_a_duplicated_codebook_is_refused_by_the_patched_layer(scratch):
    """Distinctness is checked in the layer, at export time, on the table it used."""
    import subprocess

    tree = scratch("duplicated", BASELINE)
    apply_floor(tree)
    done = subprocess.run([sys.executable, "-c", DUPLICATE_SCRIPT], cwd=str(tree),
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-1500:]
    assert "CodebookCollapse" in done.stdout, done.stdout[-1500:]
    assert "duplicate rows" in done.stdout


def test_the_frozen_baseline_is_refused_and_stays_byte_identical():
    """The baseline is the copy source for every candidate; patching it destroys provenance."""
    import hashlib

    module = BASELINE / "model" / "layers.py"
    before = hashlib.sha256(module.read_bytes()).hexdigest()
    with pytest.raises(ReadOnlyBaseline):
        apply_floor(BASELINE)
    assert hashlib.sha256(module.read_bytes()).hexdigest() == before
    assert not (BASELINE / "assignment_floor.py").exists()


def test_the_artifact_audit_refuses_a_level_that_uses_one_codeword(tmp_path):
    import json

    live = {str(i): [i % 64, (i // 64) % 64, i % 7] for i in range(4096)}
    path = tmp_path / "live.json"
    path.write_text(json.dumps(live))
    report = audit_sid(path)
    assert report["status"] == "PASS", report
    assert [entry["distinct"] for entry in report["levels"]] == [64, 64, 7]

    # the fourth column is the collision-disambiguation suffix, not a level: a constant
    # suffix must not fail a healthy identifier
    suffixed = {key: list(value) + [0] for key, value in live.items()}
    path.write_text(json.dumps(suffixed))
    assert audit_sid(path)["status"] == "PASS"

    for level in range(3):
        dead = {key: list(value) for key, value in live.items()}
        for value in dead.values():
            value[level] = 1
        path.write_text(json.dumps(dead))
        report = audit_sid(path)
        assert report["status"] == "FAIL"
        assert "level(s) " + str(level + 1) in report["detail"]


BYPASS_EDIT = ("        index = self._indices(cost, infer_use_sk)",
               "        index = torch.argmin(cost, dim=-1)")


def test_a_mechanism_that_assigns_on_its_own_is_refused(scratch):
    """The escape from round 11: replace `forward`'s assignment, keep the rest."""
    source = REPO / "stage2_RQ-VAE" / "TIGER_graphlet_codebook_rq"
    if not (source / "model" / "mixed_rq.py").is_file():
        pytest.skip("the mixed arm is not present")
    tree = scratch("bypass", source)
    apply_floor(tree)
    healthy = verify_tree(tree)
    assert healthy["ok"] is True
    assert healthy["used"] == healthy["used_forward"] == healthy["codewords"]

    module = tree / "model" / "mixed_rq.py"
    text = module.read_text(encoding="utf-8")
    assert BYPASS_EDIT[0] in text
    module.write_text(text.replace(*BYPASS_EDIT), encoding="utf-8")
    verdict = verify_tree(tree)
    assert verdict["ok"] is False
    # The static scan is the first line of defence and names the site; the runtime check
    # catches it only if the scan is bypassed. Either is a refusal.
    assert ("CodebookCollapse" in verdict["reason"]
            or "not guarded" in verdict["reason"])


def test_a_tree_whose_check_was_deleted_is_refused(scratch):
    tree = scratch("deleted", BASELINE)
    apply_floor(tree)
    module = tree / "model" / "layers.py"
    text = module.read_text(encoding="utf-8")
    text = text.replace("        verify_assignment(embed_ind, self.n_embed, ASSIGNMENT_FLOOR)\n", "")
    module.write_text(text, encoding="utf-8")
    with pytest.raises(UnknownTree) as error:
        apply_floor(tree)
    assert "assignment check was removed" in str(error.value)


def test_an_installed_floor_older_than_the_source_is_refreshed(scratch):
    tree = scratch("stale", BASELINE)
    apply_floor(tree)
    installed = tree / "assignment_floor.py"
    installed.write_text("def constrained_assignment(): pass\n", encoding="utf-8")
    report = apply_floor(tree)
    assert report["changed"] == "module"
    assert "def _coverage_assignment" in installed.read_text(encoding="utf-8")


def test_the_balancing_rule_is_not_switched_off(scratch):
    """CLAUDE.md 19h: the baseline's Sinkhorn stays; the floor applies to its output."""
    tree = scratch("balancing", BASELINE)
    apply_floor(tree)
    text = (tree / "model" / "layers.py").read_text(encoding="utf-8")
    assert "self.use_sk = sk_epsilon > 0 and sk_iters > 0" in text
    assert "not FORCE_CONSTRAINED_ALL" not in text
    index, _ = constrained_assignment(-torch.rand(4096, 8), 1)
    assert int(torch.bincount(index, minlength=8).min()) >= 1


def test_the_selfcheck_command_separates_a_protected_tree_from_a_broken_one(scratch):
    """The one command an author runs before training; it must not be optimistic."""
    import subprocess

    script = Path(__file__).resolve().parent / "selfcheck.py"
    good = scratch("selfcheck_good", BASELINE)
    done = subprocess.run([sys.executable, str(script), str(good)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-1200:]
    assert "PROTECTED" in done.stdout

    bad = scratch("selfcheck_bad", BASELINE)
    apply_floor(bad)
    module = bad / "model" / "layers.py"
    text = module.read_text(encoding="utf-8")
    text = text.replace("        embed_ind = self._indices(cost, infer_use_sk)",
                        "        embed_ind = torch.argmin(cost, dim=-1)")
    module.write_text(text, encoding="utf-8")
    done = subprocess.run([sys.executable, str(script), str(bad)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 1
    assert "NOT PROTECTED" in done.stdout


def test_identical_encoder_output_is_refused_only_when_it_makes_the_costs_tie(scratch):
    """What is refused is a tie-decided assignment, not an input that happens to repeat."""
    tree = scratch("tie", BASELINE)
    apply_floor(tree)
    probe = """
import math, torch
from model.layers import VQLayer
layer = VQLayer(8, 4)
codes = layer.get_code_embs()
with torch.no_grad():
    for k in range(codes.shape[0]):
        angle = 2 * math.pi * k / codes.shape[0]
        codes[k] = torch.tensor([math.cos(angle), math.sin(angle), 0.0, 0.0])
try:
    with torch.no_grad():
        layer(torch.zeros(4096, 4))
except Exception as error:
    print(type(error).__name__ + ": " + str(error)[:60])
else:
    print("NO_ERROR")
"""
    import subprocess
    done = subprocess.run([sys.executable, "-c", probe], cwd=str(tree),
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-1200:]
    assert "CodebookCollapse" in done.stdout
    assert "tie-decided" in done.stdout


EMA_FORWARD = ("        embed_ind = self._indices(cost, infer_use_sk)\n"
               "        verify_assignment(embed_ind, self.n_embed, ASSIGNMENT_FLOOR)\n"
               "        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)")


def test_the_ema_quantiser_is_guarded_too(scratch):
    """VQ_TYPE="ema" trains through a second forward; leaving it unpatched would be a hole."""
    tree = scratch("ema", BASELINE)
    apply_floor(tree)
    text = (tree / "model" / "layers.py").read_text(encoding="utf-8")
    assert EMA_FORWARD in text
    assert text.count("verify_margins(cost)") == 2        # VQLayer and EMAVQLayer
    assert text.count("verify_codebook(self.get_code_embs(), distance=False)") == 2
    verdict = verify_tree(tree)
    assert verdict["ok"] is True
    assert set(verdict["layers"]) == {"VQLayer", "EMAVQLayer"}
    assert all(entry["used"] == entry["used_forward"] == 8
               for entry in verdict["layers"].values())

    # and a tree that replaces the EMA assignment is refused, like the default one
    module = tree / "model" / "layers.py"
    module.write_text(text.replace(EMA_FORWARD,
                                   "        embed_ind = torch.argmin(cost, dim=-1)\n"
                                   "        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)"),
                      encoding="utf-8")
    refused = verify_tree(tree)
    assert refused["ok"] is False
    # Removing the checks removes the exception too: the probe then reports the coverage
    # it measured (1 of 8 through the EMA forward) instead of a raised error.
    assert ("EMAVQLayer" in refused["reason"] or "CodebookCollapse" in refused["reason"]
            or "not guarded" in refused["reason"])


SNEAKY = [
    "",
    "",
    "class SneakyLayer(VQLayer):",
    "    def forward(self, x, infer_use_sk=False):",
    "        latent = x.view(-1, self.dim)",
    "        return torch.argmin(self._distances(latent), dim=-1)",
    "",
]


def test_a_new_assignment_site_in_the_quantiser_is_refused(scratch):
    """The generic form of the EMA hole: any raw argmin left in the patched module."""
    tree = scratch("new_site", BASELINE)
    module = tree / "model" / "layers.py"
    module.write_text(module.read_text(encoding="utf-8") + "\n".join(SNEAKY),
                      encoding="utf-8")
    with pytest.raises(UnknownTree) as error:
        apply_floor(tree)
    assert "still assigns without the guard" in str(error.value)
    assert "model/layers.py" in str(error.value)


def test_a_declared_non_assignment_use_is_allowed(scratch):
    """The exception has to be written down with a reason, not inferred."""
    # The same module object `apply_floor` comes from: the package is imported under two
    # names in this suite (`anticollapse` from the test's sys.path, `stage2.anticollapse`
    # from the repository root), and a dict set on the other one would be invisible.
    import anticollapse.patch_tree as patch_tree

    tree = scratch("declared", BASELINE)
    module = tree / "model" / "layers.py"
    module.write_text(module.read_text(encoding="utf-8")
                      + "\n# reporting only: torch.argmax(counts, dim=-1)\n",
                      encoding="utf-8")
    patch_tree.ALLOWED_SITES["model/layers.py"] = {
        "torch.argmax(counts, dim=-1)": "a histogram maximum for logging"}
    try:
        apply_floor(tree)
        assert patch_tree.unguarded_sites(tree, "euclid") == []
    finally:
        patch_tree.ALLOWED_SITES.pop("model/layers.py", None)


def test_a_zeroed_floor_constant_is_refused(scratch):
    """`ASSIGNMENT_FLOOR = 0` leaves every guard call in place and disables the floor."""
    tree = scratch("zero_floor", BASELINE)
    apply_floor(tree)
    module = tree / "model" / "layers.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace("ASSIGNMENT_FLOOR = 1",
                                                   "ASSIGNMENT_FLOOR = 0"),
        encoding="utf-8")
    with pytest.raises(UnknownTree) as error:
        apply_floor(tree)
    assert "the floor was disabled" in str(error.value)


def test_an_assignment_through_min_dim_is_refused(scratch):
    """The same collapse through a different spelling: `cost.min(dim=-1)[1]`."""
    tree = scratch("min_dim", BASELINE)
    apply_floor(tree)
    module = tree / "model" / "layers.py"
    source = module.read_text(encoding="utf-8")
    source += "\n".join([
        "",
        "class Alternate(VQLayer):",
        "    def pick(self, cost):",
        "        return cost.min(dim=-1)[1]",
        "",
    ])
    module.write_text(source, encoding="utf-8")
    with pytest.raises(UnknownTree) as error:
        apply_floor(tree)
    assert "still assigns without the guard" in str(error.value)
    assert "min(dim=" in str(error.value)
