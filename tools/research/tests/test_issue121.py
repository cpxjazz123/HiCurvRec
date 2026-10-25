"""Issue #121 regressions: the joint scope gate, the one permitted baseline and
the distinction between an infrastructure error and a defect.

The negative cases are the ones the issue names: a behaviour-only suffix, a
hyperbolic-only distance swap, two unrelated losses and inherited hyperbolic
code that nothing calls. All four have to be out of scope even when they score
well.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.research.rounds import baseline as baseline_module
from tools.research.rounds import ideas as ideas_module
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import scope as scope_module
from tools.research.rounds import scoring as scoring_module


def _coupled(**overrides):
    """A proposal that meets the joint behaviour x hyperbolic contract."""
    idea = {
        "idea_id": "coupled", "title": "t", "motivation": "m", "novelty": "n",
        "key_change": "c", "specification": "s", "cost_estimate": "low",
        "falsifiable_hypothesis": "h", "risks": ["r"],
        "behavior_signal": "train-only directed item transitions between consecutive "
                           "items of the training sequences",
        "behavior_computation": "counted on the train split interactions only",
        "leakage_boundary": "train-only, no validation or test interaction is read",
        "hyperbolic_operator": "Poincare distance with negative curvature",
        "hyperbolic_location": "inside the residual quantization of the codebook",
        "coupling_path": "the behaviour weight multiplies the curved residual distance "
                         "inside the assignment, so both sides act in one pathway",
        "coupling_equation": "L = sum_b w_behaviour(x_b) * d_poincare(v_b, c_b)",
        "observables": "the codebook assignment changes when the weight changes",
        "ablation_behavior_off": "set the behaviour weight to one everywhere",
        "ablation_geometry_off": "replace the curved distance by the Euclidean one",
        "ablation_both_on": "the full coupled assignment",
        "spec_equations": "z = exp_0^poincare(log_0(v) + w * delta)",
        "spec_inputs_outputs": "takes item vectors and transitions, returns codes",
        "safeguards": "clamp the weight and check the loss is finite",
    }
    idea.update(overrides)
    return idea


class ScopeGateTests(unittest.TestCase):
    """The four out-of-scope shapes the issue names."""

    def test_a_coupled_proposal_is_in_scope(self):
        verdict = scope_module.check_scope(_coupled())
        self.assertTrue(verdict["in_scope"], verdict["reasons"])

    def test_a_behaviour_only_suffix_is_out_of_scope(self):
        idea = _coupled(
            hyperbolic_operator="none: the behaviour is appended to the SID suffix",
            hyperbolic_location="after the Euclidean quantizer",
            coupling_path="the behaviour suffix is appended after the quantizer",
            coupling_equation="L = L_rec + lambda_b * L_behaviour")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])
        self.assertTrue(any("curved operation" in reason for reason in verdict["reasons"]))

    def test_a_hyperbolic_only_distance_swap_is_out_of_scope(self):
        idea = _coupled(
            behavior_signal="none", behavior_computation="none", leakage_boundary="none",
            coupling_path="the distance is swapped for a curved one in the quantizer",
            coupling_equation="L = d_poincare(v, c)")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])
        self.assertTrue(any("behaviour" in reason for reason in verdict["reasons"]))

    def test_two_unrelated_losses_are_out_of_scope(self):
        idea = _coupled(
            coupling_path="a behaviour loss and a hyperbolic loss are added to the "
                          "reconstruction loss",
            coupling_equation="L = L_rec + lambda_b * L_behaviour + lambda_h * L_hyperbolic")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])
        self.assertTrue(any("side by side" in reason for reason in verdict["reasons"]))

    def test_inherited_unused_hyperbolic_code_is_out_of_scope(self):
        idea = _coupled(
            coupling_path="the existing hyperbolic module is reused as it is and a "
                          "behaviour term is added to the loss",
            coupling_equation="L = L_rec + lambda_b * L_behaviour")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])

    def test_a_high_score_does_not_compensate_an_out_of_scope_idea(self):
        """The schema refuses it, and the selection would refuse it as well."""
        with self.assertRaises(ideas_module.IdeaError):
            ideas_module.validate_idea(
                _coupled(idea_id="b",
                         coupling_path="a behaviour loss and a hyperbolic loss are added "
                                       "to the loss",
                         coupling_equation="L = L_rec + lambda_b * L_behaviour + "
                                           "lambda_h * L_hyperbolic"),
                agent="agent1")
        # And a row that reaches the selection out of scope is rejected there,
        # whatever it scored.
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a")]}),
            ("agent2", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
            ("agent3", {"ideas": [_coupled(idea_id="d", title="delta")]}),
        ])
        reports = []
        for scorer in ("s1", "s2", "s3"):
            scores = [{"idea_id": idea["idea_id"],
                       "scores": {dimension: 5 for dimension in scoring_module.DIMENSIONS},
                       "rationale": "it looks strong"} for idea in ideas]
            reports.append({"scorer": scorer, "scores": scores})
        matrix = scoring_module.build_matrix(ideas, reports)
        for row in matrix["rows"]:
            if row["idea_id"] == "a":
                row["scope_verdict"] = "OUT_OF_SCOPE"
                row["scope_reasons"] = ["the coupling is not connected"]
        selection = scoring_module.select(matrix)
        rejected = [entry for entry in selection["rejected"] if entry["idea_id"] == "a"]
        self.assertTrue(rejected)
        self.assertTrue(any("out of scope" in reason for reason in rejected[0]["reasons"]))
        self.assertNotIn("a", [entry["idea_id"] for entry in selection["selected"]])

    def test_the_scope_decision_is_audited_against_the_proposal(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a")]}),
            ("agent2", {"ideas": [_coupled(idea_id="b", title="beta")]}),
            ("agent3", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
        ])
        reports = []
        for scorer in ("s1", "s2", "s3"):
            scores = [{"idea_id": idea["idea_id"],
                       "scores": {dimension: 4 for dimension in scoring_module.DIMENSIONS},
                       "rationale": "fine",
                       "scope_verdict": "IN_SCOPE" if scorer != "s2" else "OUT_OF_SCOPE",
                       "scope_reason": "the coupling looks weak" if scorer == "s2" else ""}
                      for idea in ideas]
            reports.append({"scorer": scorer, "scores": scores})
        matrix = scoring_module.build_matrix(ideas, reports)
        row = [entry for entry in matrix["rows"] if entry["idea_id"] == "a"][0]
        # One reviewer's label does not decide the idea: the proposal's own
        # content does, and the disagreement is recorded.
        self.assertEqual(row["scope_verdict"], "IN_SCOPE")
        self.assertEqual([entry["scorer"] for entry in row["scope_disagreements"]], ["s2"])

    def test_an_out_of_scope_idea_is_never_the_fallback(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a")]}),
            ("agent2", {"ideas": [_coupled(idea_id="b", title="beta")]}),
            ("agent3", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
        ])
        reports = []
        for scorer in ("s1", "s2", "s3"):
            scores = []
            for idea in ideas:
                if idea["idea_id"] == "a":
                    values = {dimension: 5 for dimension in scoring_module.DIMENSIONS}
                else:
                    values = {dimension: 1 for dimension in scoring_module.DIMENSIONS}
                scores.append({"idea_id": idea["idea_id"], "scores": values,
                               "rationale": "r"})
            reports.append({"scorer": scorer, "scores": scores})
        matrix = scoring_module.build_matrix(ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertEqual([entry["idea_id"] for entry in selection["selected"]], ["a"])

    def test_an_idea_missing_the_scope_fields_is_rejected(self):
        idea = _coupled()
        del idea["coupling_path"]
        with self.assertRaises(ideas_module.IdeaError):
            ideas_module.validate_idea(idea, agent="agent1")


class BaselineTests(unittest.TestCase):
    """One permitted source tree, one fresh copy per candidate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = self.root / "TIGER_RQ-VAE"
        (self.baseline / "model").mkdir(parents=True)
        (self.baseline / "train_rqvae.py").write_text(
            'OUTPUT_JSON = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/'
            'TIGER_RQ-VAE/item_sids.json")\n', encoding="utf-8")
        (self.baseline / "model" / "model.py").write_text("x = 1\n", encoding="utf-8")
        (self.baseline / "logs").mkdir()
        (self.baseline / "logs" / "old.log").write_text("stale\n", encoding="utf-8")

    def test_a_missing_baseline_fails_closed(self):
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.copy_baseline(self.root / "candidate",
                                          baseline=self.root / "does_not_exist")

    def test_a_candidate_is_a_fresh_copy_without_the_run_artifacts(self):
        candidate = self.root / "candidate"
        provenance = baseline_module.copy_baseline(candidate, baseline=self.baseline)
        self.assertTrue((candidate / "train_rqvae.py").is_file())
        self.assertFalse((candidate / "logs").exists())
        self.assertEqual(provenance["baseline_digest"], provenance["candidate_digest"])

    def test_two_candidates_get_two_independent_copies(self):
        first = self.root / "one"
        second = self.root / "two"
        baseline_module.copy_baseline(first, baseline=self.baseline)
        baseline_module.copy_baseline(second, baseline=self.baseline)
        (first / "train_rqvae.py").write_text("changed = True\n", encoding="utf-8")
        self.assertNotEqual(baseline_module.source_digest(first),
                            baseline_module.source_digest(second))
        self.assertEqual(baseline_module.source_digest(second),
                         baseline_module.source_digest(self.baseline))

    def test_a_changed_baseline_is_detected(self):
        before = baseline_module.source_digest(self.baseline)
        (self.baseline / "model" / "model.py").write_text("x = 2\n", encoding="utf-8")
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.guard_baseline_untouched(self.baseline, before)

    def test_a_candidate_that_would_write_into_the_baseline_is_refused(self):
        candidate = self.root / "candidate"
        baseline_module.copy_baseline(candidate, baseline=self.baseline)
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.guard_output_paths(candidate, baseline=self.baseline)

    def test_a_candidate_that_would_write_into_another_candidate_is_refused(self):
        candidate = self.root / "candidate"
        other = self.root / "TIGER_other"
        baseline_module.copy_baseline(candidate, baseline=self.baseline)
        baseline_module.copy_baseline(other, baseline=self.baseline)
        (candidate / "train_rqvae.py").write_text(
            'OUTPUT_JSON = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/'
            'TIGER_other/item_sids.json")\n', encoding="utf-8")
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.guard_output_paths(candidate, baseline=self.baseline,
                                               other_candidates=[other])

    def test_an_isolated_candidate_passes_the_guard(self):
        candidate = self.root / "candidate"
        baseline_module.copy_baseline(candidate, baseline=self.baseline)
        (candidate / "train_rqvae.py").write_text(
            'OUTPUT_JSON = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/'
            'TIGER_candidate/item_sids.json")\n', encoding="utf-8")
        resolved = baseline_module.guard_output_paths(candidate, baseline=self.baseline)
        self.assertIn("OUTPUT_JSON", resolved)


class StateDistinctionTests(unittest.TestCase):
    """An infrastructure error is not a verdict on the mechanism."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = pipeline_module.new_round(self.root, "round_test", objective="o")
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
        self.state = pipeline_module.load_state(self.directory)
        self.state["implementations"] = [pipeline_module.implementation_record(
            _coupled(), {"changed_files": ["model.py"]}, tree)]

    def test_an_agent_error_does_not_block_the_mechanism(self):
        state = pipeline_module.mark_agent_error(self.state, "coupled", "the agent timed out")
        record = state["implementations"][0]
        self.assertEqual(record["status"], "agent_error")
        self.assertNotIn("coupled", [entry["idea_id"] for entry in state["blocked"]])
        self.assertEqual(state["agent_errors"][0]["kind"], pipeline_module.AGENT_INFRA_ERROR)

    def test_an_agent_error_is_counted_for_the_retry(self):
        state = pipeline_module.mark_agent_error(self.state, "coupled", "boom")
        state = pipeline_module.mark_agent_error(state, "coupled", "boom again")
        self.assertEqual(state["implementations"][0]["agent_error_attempts"], 2)

    def test_a_defect_blocks_with_its_kind(self):
        state = pipeline_module.mark_blocked(self.state, "coupled", "the export path is wrong")
        self.assertEqual(state["blocked"][0]["kind"], pipeline_module.IMPLEMENTATION_NEEDS_FIX)

    def test_a_measured_infeasibility_is_its_own_kind(self):
        state = pipeline_module.mark_resource_blocked(self.state, "coupled",
                                                      "one epoch takes 40 minutes")
        self.assertEqual(state["resource_blocked"][0]["kind"], pipeline_module.RESOURCE_BLOCKED)
        self.assertEqual(state["implementations"][0]["status"], "resource_blocked")

    def test_a_missing_baseline_is_its_own_kind(self):
        state = pipeline_module.mark_blocked(self.state, "coupled", "baseline unavailable",
                                              kind=pipeline_module.BASELINE_UNAVAILABLE)
        self.assertEqual(state["blocked"][0]["kind"], pipeline_module.BASELINE_UNAVAILABLE)

    def test_the_summary_counts_the_kinds(self):
        state = pipeline_module.mark_agent_error(self.state, "coupled", "boom")
        summary = pipeline_module.summary(state)
        self.assertEqual(summary["agent_errors"], 1)
        self.assertEqual(summary["blocked"], 0)


if __name__ == "__main__":
    unittest.main()


class DigestLocationTests(unittest.TestCase):
    """A tree's digest does not depend on where the tree is filed."""

    def test_a_tree_under_results_still_has_a_digest(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            tree = root / "results" / "research" / "issue121" / "trees" / "r" / "idea"
            (tree / "model").mkdir(parents=True)
            (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            (tree / "model" / "model.py").write_text("y = 2\n", encoding="utf-8")
            digest = baseline_module.source_digest(tree)
            self.assertTrue(digest)
            other = root / "elsewhere" / "idea"
            (other / "model").mkdir(parents=True)
            (other / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            (other / "model" / "model.py").write_text("y = 2\n", encoding="utf-8")
            self.assertEqual(digest, baseline_module.source_digest(other))
