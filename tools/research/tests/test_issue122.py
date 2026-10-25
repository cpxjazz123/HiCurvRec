"""Issue #122 regressions: weighted rubric, pairwise debate, immutable decisions.

The policy these tests pin is the one the issue asks for: seven weighted
dimensions that must agree across the prompt, the parser and the matrix; a
strict schema that refuses an incomplete review instead of averaging over it;
pairwise comparisons that are deterministic and expose order bias, ties and
cycles; no automatic fallback; and a decision record that a later recalculation
cannot rewrite.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tools.research.rounds import decision as decision_module
from tools.research.rounds import evidence as evidence_module
from tools.research.rounds import ideas as ideas_module
from tools.research.rounds import plaintext as plaintext_module
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import rubric as rubric_module
from tools.research.rounds import scoring as scoring_module
from tools.research.rounds import scope as scope_module


def _coupled(identifier="coupled", **overrides):
    idea = {
        "idea_id": identifier, "title": "t", "motivation": "m", "novelty": "n",
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


def _round_ideas(count=3):
    second = {"ideas": [_coupled("b", title="beta")]}
    if count >= 2:
        second["ideas"].append(_coupled("d", title="delta"))
    return ideas_module.collect_ideas([("agent1", {"ideas": [_coupled("a")]}),
                                       ("agent2", second),
                                       ("agent3", {"ideas": [_coupled("c", title="gamma")]})])


def _reports(ideas, *, value=4, per_idea=None):
    per_idea = per_idea or {}
    reports = []
    for scorer in ("s1", "s2", "s3"):
        scores = []
        for idea in ideas:
            score = per_idea.get(idea["idea_id"], value)
            scores.append({"idea_id": idea["idea_id"],
                           "scores": {dimension: score for dimension in scoring_module.DIMENSIONS},
                           "rationale": "r", "evidence_ids": ["E1"],
                           "objection": "", "counterexample": ""})
        reports.append({"scorer": scorer, "scores": scores})
    return reports


class RubricPolicyTests(unittest.TestCase):
    def test_the_seven_dimensions_sum_to_one_hundred(self):
        rubric = rubric_module.load_rubric()
        self.assertEqual(len(rubric["dimension_names"]), 7)
        self.assertEqual(sum(rubric["weight_by_dimension"].values()), 100.0)
        self.assertEqual(rubric["weight_by_dimension"]["behavior_evidence"], 20.0)
        self.assertEqual(rubric["weight_by_dimension"]["exploration_value"], 5.0)

    def test_the_policy_carries_a_version_and_a_digest(self):
        rubric = rubric_module.load_rubric()
        self.assertTrue(rubric["version"])
        self.assertEqual(len(rubric["sha256"]), 64)

    def test_a_rubric_with_weights_that_do_not_sum_to_one_hundred_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "rubric.json"
            payload = json.loads(Path(rubric_module.RUBRIC_FILE).read_text())
            payload["dimensions"][0]["weight"] = 21
            path.write_text(json.dumps(payload))
            with self.assertRaises(rubric_module.RubricError):
                rubric_module.load_rubric(path)

    def test_the_weighted_score_uses_the_weights(self):
        rubric = rubric_module.load_rubric()
        medians = {dimension: (5 if dimension == "exploration_value" else 0)
                   for dimension in rubric["dimension_names"]}
        self.assertEqual(rubric_module.weighted_score(medians, rubric), 0.25)

    def test_prompt_parser_and_matrix_agree(self):
        self.assertEqual(tuple(plaintext_module._SCORE_DIMENSIONS),
                         tuple(scoring_module.DIMENSIONS))
        prompt = __import__("tools.research.rounds.agents", fromlist=["SCORE_PROMPT"]).SCORE_PROMPT
        for dimension in scoring_module.DIMENSIONS:
            self.assertIn(dimension, prompt)


class StrictSchemaTests(unittest.TestCase):
    def test_a_renamed_dimension_is_a_schema_error(self):
        ideas = _round_ideas(1)
        reports = _reports(ideas)
        reports[0]["scores"][0]["scores"]["novelty_"] = reports[0]["scores"][0]["scores"].pop("novelty")
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)

    def test_a_missing_dimension_is_a_schema_error(self):
        ideas = _round_ideas(1)
        reports = _reports(ideas)
        del reports[0]["scores"][0]["scores"]["falsifiability"]
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)

    def test_a_reviewer_that_skips_a_proposal_is_a_schema_error(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas)
        reports[2]["scores"] = reports[2]["scores"][:-1]
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)


class PairwiseTests(unittest.TestCase):
    def test_a_majority_win_is_awarded_deterministically(self):
        rows = [{"idea_id": "a"}, {"idea_id": "b"}]
        comparisons = [
            {"first": "a", "second": "b", "winner": "a", "reviewer": "s1", "first_listed": "a"},
            {"first": "a", "second": "b", "winner": "a", "reviewer": "s2", "first_listed": "a"},
            {"first": "b", "second": "a", "winner": "b", "reviewer": "s3", "first_listed": "b"},
        ]
        result = scoring_module.pairwise(rows, comparisons)
        self.assertEqual(result["copeland"]["a"], 1)
        self.assertEqual(result["copeland"]["b"], -1)
        self.assertFalse(result["ties"])

    def test_a_tie_is_recorded(self):
        rows = [{"idea_id": "a"}, {"idea_id": "b"}]
        comparisons = [
            {"first": "a", "second": "b", "winner": "a", "reviewer": "s1", "first_listed": "a"},
            {"first": "b", "second": "a", "winner": "b", "reviewer": "s2", "first_listed": "b"},
        ]
        result = scoring_module.pairwise(rows, comparisons)
        self.assertEqual(result["ties"], [["a", "b"]])
        self.assertEqual(result["copeland"]["a"], 0)

    def test_a_cycle_is_detected(self):
        rows = [{"idea_id": idea} for idea in ("a", "b", "c")]
        comparisons = [
            {"first": "a", "second": "b", "winner": "a", "reviewer": "s1"},
            {"first": "b", "second": "c", "winner": "b", "reviewer": "s1"},
            {"first": "c", "second": "a", "winner": "c", "reviewer": "s1"},
        ]
        result = scoring_module.pairwise(rows, comparisons)
        self.assertEqual(result["cycles"], [["a", "b", "c"]])

    def test_the_position_bias_is_measured(self):
        rows = [{"idea_id": "a"}, {"idea_id": "b"}]
        comparisons = [
            {"first": "a", "second": "b", "winner": "a", "reviewer": "s1", "first_listed": "a"},
            {"first": "b", "second": "a", "winner": "b", "reviewer": "s2", "first_listed": "b"},
        ]
        result = scoring_module.pairwise(rows, comparisons)
        self.assertEqual(result["position_bias"],
                         {"first_listed_wins": 2, "second_listed_wins": 0})

    def test_a_malformed_comparison_is_refused(self):
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.pairwise([{"idea_id": "a"}, {"idea_id": "b"}],
                                    [{"first": "a", "second": "b", "winner": "c"}])


class DecisionRecordTests(unittest.TestCase):
    def _evaluation(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas)
        return ideas, reports, scoring_module.evaluate(ideas, reports)

    def test_the_decision_is_written_once_and_never_rewritten(self):
        ideas, reports, evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            snapshot = decision_module.build_snapshot(
                round_id="r1", evaluation=evaluation, reports=reports, bundles={},
                proposals=ideas)
            path = decision_module.write_snapshot(directory, snapshot)
            self.assertTrue(path.is_file())
            with self.assertRaises(RuntimeError):
                decision_module.write_snapshot(directory, dict(snapshot, decision_hash="other"))

    def test_a_recalculation_is_an_additive_audit(self):
        ideas, reports, evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            snapshot = decision_module.build_snapshot(
                round_id="r1", evaluation=evaluation, reports=reports, bundles={},
                proposals=ideas)
            path = decision_module.write_snapshot(directory, snapshot)
            before = path.read_bytes()
            audit = decision_module.write_audit(directory, {"note": "recomputed"},
                                                reason="a scoring bug was fixed")
            self.assertTrue(audit.name.startswith("rescored_audit_"))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(decision_module.load_snapshot(directory)["decision_hash"],
                             snapshot["decision_hash"])

    def test_the_decision_hash_changes_with_the_decision(self):
        ideas, reports, evaluation = self._evaluation()
        first = decision_module.build_snapshot(round_id="r1", evaluation=evaluation,
                                               reports=reports, bundles={}, proposals=ideas)
        changed = dict(evaluation)
        changed["selection"] = dict(evaluation["selection"], selected=[])
        second = decision_module.build_snapshot(round_id="r1", evaluation=changed,
                                                reports=reports, bundles={}, proposals=ideas)
        self.assertNotEqual(first["decision_hash"], second["decision_hash"])


class EvidenceBundleTests(unittest.TestCase):
    def test_an_unknown_section_says_unknown(self):
        bundle = evidence_module.build_bundle({"idea_id": "x"}, root=Path("."))
        states = {entry["status"] for entry in bundle["behavioral_evidence"]}
        self.assertIn(rubric_module.UNKNOWN, states)
        self.assertTrue(any(entry["status"] == rubric_module.UNKNOWN
                            for entry in bundle["behavioral_evidence"]))

    def test_the_literature_is_not_checked_rather_than_claimed(self):
        bundle = evidence_module.build_bundle(_coupled(), root=Path("."))
        statuses = {entry["status"] for entry in bundle["prior_work"]}
        self.assertIn(rubric_module.NOT_CHECKED, statuses)

    def test_every_evidence_entry_has_an_id_and_the_bundle_a_digest(self):
        bundle = evidence_module.build_bundle(_coupled(), root=Path("."))
        self.assertEqual(len(bundle["evidence_ids"]), sum(
            len(bundle[section]) for section in
            ("behavioral_evidence", "geometric_necessity", "prior_work",
             "experiment_ledger", "implementation_cost")))
        self.assertEqual(len(bundle["sha256"]), 64)

    def test_a_measured_outcome_carries_its_comparability(self):
        bundle = evidence_module.build_bundle(_coupled(), root=Path("."))
        ledger = bundle["experiment_ledger"]
        self.assertTrue(ledger)
        for entry in ledger:
            if isinstance(entry.get("value"), dict) and "test_recall_at_10" in entry["value"]:
                self.assertIn("comparable_to_frozen", entry["value"])


class NoFallbackTests(unittest.TestCase):
    def test_nobody_qualified_means_no_qualified_idea(self):
        ideas = _round_ideas(1)
        reports = _reports(ideas, value=1)
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertEqual(selection["status"], scoring_module.STATUS_NO_QUALIFIED)
        self.assertEqual(selection["selected"], [])

    def test_the_round_goes_back_to_ideation(self):
        ideas = _round_ideas(1)
        reports = _reports(ideas, value=1)
        with tempfile.TemporaryDirectory() as work:
            directory = pipeline_module.new_round(Path(work), "round_nf", objective="o")
            state = pipeline_module.load_state(directory)
            state["ideas"] = ideas
            state["step"] = "scoring"
            state = pipeline_module.apply_scoring(state, reports, directory=directory)
            self.assertEqual(state["step"], "ideation")
            self.assertIn("no_qualified_idea", [event["event"] for event in state["events"]])

    def test_an_out_of_scope_proposal_with_perfect_scores_is_not_selected(self):
        ideas = _round_ideas(2)
        # b is out of scope: a behaviour suffix on an unchanged quantiser
        ideas[1]["coupling_path"] = "a behaviour suffix is appended after the quantiser"
        ideas[1]["coupling_equation"] = "L = L_rec + lambda_b * L_behaviour"
        idea_index = {idea["idea_id"]: idea for idea in ideas}
        verdict = scope_module.check_scope(idea_index[ideas[1]["idea_id"]])
        self.assertFalse(verdict["in_scope"])
        reports = _reports(ideas, value=5)
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertNotIn(ideas[1]["idea_id"], [entry["idea_id"] for entry in selection["selected"]])
        rejected = [entry for entry in selection["rejected"]
                    if entry["idea_id"] == ideas[1]["idea_id"]][0]
        self.assertEqual(rejected["status"], scoring_module.REASON_OUT_OF_SCOPE)


class BackupsAndCapTests(unittest.TestCase):
    def test_one_is_selected_and_the_rest_become_ordered_backups(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas, per_idea={ideas[0]["idea_id"]: 5,
                                            ideas[1]["idea_id"]: 4,
                                            ideas[2]["idea_id"]: 3})
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertEqual(len(selection["selected"]), 1)
        self.assertTrue(selection["backups"])
        self.assertLessEqual(len(selection["backups"]),
                             scoring_module.DEFAULT_SELECTION["pairwise_finalists"])
        self.assertIn(ideas[0]["idea_id"], [entry["idea_id"] for entry in selection["selected"]])

    def test_a_resource_blocked_proposal_keeps_its_science(self):
        ideas = _round_ideas(1)
        ideas[0]["cost_estimate_hours"] = 900
        reports = _reports(ideas, value=5)
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        rejected = selection["rejected"][0]
        self.assertEqual(rejected["status"], scoring_module.REASON_RESOURCE)
        self.assertNotIn(rejected["status"], ("OUT_OF_SCOPE", "REFUTED"))


if __name__ == "__main__":
    unittest.main()
