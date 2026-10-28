"""Issue #122 regressions: the ranking policy, the aggregate order, immutable decisions.

The policy these tests pin is the one the issue asks for: the reviewers RANK
the proposals, the aggregate order is auditable, a strict schema that refuses an
incomplete ranking instead of averaging over it, no automatic fallback, and a
decision record that a later recalculation cannot rewrite.
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


def _ranking(ideas, order=None, reason="the evidence puts it here"):
    """One reviewer's ranking of the proposals, best first."""
    order = order or [idea["idea_id"] for idea in ideas]
    return [{"idea_id": identifier, "position": position, "reason": reason}
            for position, identifier in enumerate(order, start=1)]


def _order(ideas, best):
    """The ranking of every proposal, with one of them first."""
    identifiers = [idea["idea_id"] for idea in ideas]
    return [best] + [identifier for identifier in identifiers if identifier != best]


def _reports(ideas, *, order=None, reviewers=("s1", "s2", "s3")):
    """The reviewers' rankings of the same proposals."""
    return [{"reviewer": reviewer, "ranking": _ranking(ideas, order),
             "rubric_version": scoring_module.RUBRIC["version"],
             "rubric_sha256": scoring_module.RUBRIC["sha256"]}
            for reviewer in reviewers]


class PolicyTests(unittest.TestCase):
    def test_the_policy_names_the_selection_rule(self):
        rubric = rubric_module.load_rubric()
        self.assertEqual(rubric["selection"]["max_selected"], 1)
        self.assertGreaterEqual(rubric["selection"]["min_reviewers"], 1)
        self.assertIn("total_rank", rubric["selection"]["tie_break_order"])
        self.assertIn("idea_id", rubric["selection"]["tie_break_order"])

    def test_the_policy_carries_a_version_and_a_digest(self):
        rubric = rubric_module.load_rubric()
        self.assertTrue(rubric["version"])
        self.assertEqual(len(rubric["sha256"]), 64)

    def test_a_policy_without_a_selection_rule_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "rubric.json"
            payload = json.loads(Path(rubric_module.RUBRIC_FILE).read_text())
            del payload["selection"]
            path.write_text(json.dumps(payload))
            with self.assertRaises(rubric_module.RubricError):
                rubric_module.load_rubric(path)

    def test_a_policy_without_a_version_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "rubric.json"
            payload = json.loads(Path(rubric_module.RUBRIC_FILE).read_text())
            payload["version"] = "  "
            path.write_text(json.dumps(payload))
            with self.assertRaises(rubric_module.RubricError):
                rubric_module.load_rubric(path)

    def test_the_prompt_and_the_parser_agree_on_the_ranking(self):
        prompt = __import__("tools.research.rounds.agents",
                            fromlist=["RANK_PROMPT"]).RANK_PROMPT
        self.assertIn("RANK:", prompt)
        self.assertIn("position=", prompt)
        parsed = plaintext_module.parse_ranks(
            "RANK: a | position=1 | reason=first\nRANK: b | position=2 | reason=second\n")
        self.assertEqual([entry["idea_id"] for entry in parsed["ranking"]], ["a", "b"])


class StrictSchemaTests(unittest.TestCase):
    def test_a_reviewer_that_skips_a_proposal_is_a_schema_error(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas)
        reports[2]["ranking"] = reports[2]["ranking"][:-1]
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)

    def test_a_reviewer_that_repeats_a_proposal_is_a_schema_error(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas)
        reports[0]["ranking"][1]["idea_id"] = reports[0]["ranking"][0]["idea_id"]
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)

    def test_a_tied_position_is_a_schema_error(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas)
        reports[0]["ranking"][1]["position"] = 1
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)

    def test_a_position_outside_the_range_is_a_schema_error(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas)
        reports[0]["ranking"][0]["position"] = 7
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)

    def test_a_ranking_without_a_reason_is_a_schema_error(self):
        ideas = _round_ideas(1)
        reports = _reports(ideas)
        reports[0]["ranking"][0]["reason"] = "  "
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)

    def test_a_missing_proposal_is_never_a_partial_average(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas, reviewers=("s1", "s2"))
        reports[1]["ranking"] = [entry for entry in reports[1]["ranking"]
                                 if entry["idea_id"] != ideas[0]["idea_id"]]
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, reports)


class AggregateOrderTests(unittest.TestCase):
    def test_the_best_placed_proposal_is_the_one_kept(self):
        ideas = _round_ideas(3)
        order = _order(ideas, ideas[0]["idea_id"])
        reports = _reports(ideas, order=order)
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertEqual(len(selection["selected"]), 1)
        self.assertEqual(selection["selected"][0]["idea_id"], order[0])
        self.assertEqual(selection["status"], scoring_module.STATUS_SELECTED)

    def test_the_aggregate_order_is_auditable(self):
        ideas = _round_ideas(3)
        order = _order(ideas, ideas[0]["idea_id"])
        reports = _reports(ideas, order=order)
        matrix = scoring_module.evaluate(ideas, reports)["matrix"]
        row = next(entry for entry in matrix["rows"]
                   if entry["idea_id"] == ideas[1]["idea_id"])
        self.assertEqual(row["total_rank"], 2 * 3)
        self.assertEqual(row["first_places"], 0)
        self.assertEqual(row["per_reviewer_position"]["s2"], 2)
        self.assertIn("evidence", row["per_reviewer_reason"]["s1"])

    def test_the_reviewers_disagreeing_changes_the_winner(self):
        ideas = _round_ideas(3)
        best = ideas[0]["idea_id"]
        second = ideas[1]["idea_id"]
        third = ideas[2]["idea_id"]
        reports = [
            {"reviewer": "s1", "ranking": _ranking(ideas, _order(ideas, best))},
            {"reviewer": "s2", "ranking": _ranking(ideas, _order(ideas, best))},
            {"reviewer": "s3", "ranking": _ranking(ideas, _order(ideas, second))},
        ]
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertEqual(selection["selected"][0]["idea_id"], best)
        # second is placed first once and second twice: 1 + 2 + 2 = 5.
        self.assertEqual(selection["aggregate"][second]["total_rank"], 5)
        self.assertEqual(selection["aggregate"][best]["first_places"], 2)

    def test_a_tie_in_the_aggregate_is_broken_deterministically(self):
        ideas = _round_ideas(2)
        first, second = ideas[0]["idea_id"], ideas[1]["idea_id"]
        reports = [
            {"reviewer": "s1", "ranking": _ranking(ideas, _order(ideas, first))},
            {"reviewer": "s2", "ranking": _ranking(ideas, _order(ideas, second))},
        ]
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertEqual(len(selection["selected"]), 1)
        self.assertEqual(selection["selected"][0]["idea_id"], min(first, second))


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

    def test_the_decision_records_the_rankings(self):
        ideas, reports, evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as work:
            snapshot = decision_module.build_snapshot(
                round_id="r1", evaluation=evaluation, reports=reports, bundles={},
                proposals=ideas)
            self.assertTrue(snapshot["reviews"])
            self.assertTrue(snapshot["reviews"][0]["ranking"])
            self.assertNotIn("pairwise", snapshot)

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
    def test_an_out_of_scope_proposal_is_never_selected(self):
        ideas = _round_ideas(2)
        # b is out of scope: a behaviour suffix on an unchanged quantiser
        ideas[1]["coupling_path"] = "a behaviour suffix is appended after the quantiser"
        ideas[1]["coupling_equation"] = "L = L_rec + lambda_b * L_behaviour"
        idea_index = {idea["idea_id"]: idea for idea in ideas}
        verdict = scope_module.check_scope(idea_index[ideas[1]["idea_id"]])
        self.assertFalse(verdict["in_scope"])
        order = _order(ideas, ideas[1]["idea_id"])
        reports = _reports(ideas, order=order)
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertNotIn(ideas[1]["idea_id"],
                         [entry["idea_id"] for entry in selection["selected"]])
        rejected = [entry for entry in selection["rejected"]
                    if entry["idea_id"] == ideas[1]["idea_id"]][0]
        self.assertEqual(rejected["status"], scoring_module.REASON_OUT_OF_SCOPE)

    def test_a_proposal_nobody_ranked_is_not_qualified(self):
        ideas = _round_ideas(2)
        reports = _reports(ideas, reviewers=("s1",))
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertEqual(selection["status"], scoring_module.STATUS_NO_QUALIFIED)
        self.assertEqual(selection["selected"], [])

    def test_the_round_goes_back_to_ideation(self):
        ideas = _round_ideas(1)
        reports = _reports(ideas, reviewers=("s1",))
        with tempfile.TemporaryDirectory() as work:
            directory = pipeline_module.new_round(Path(work), "round_nf", objective="o")
            state = pipeline_module.load_state(directory)
            state["ideas"] = ideas
            state["step"] = "scoring"
            state = pipeline_module.apply_scoring(state, reports, directory=directory)
            self.assertEqual(state["step"], "ideation")
            self.assertIn("no_qualified_idea", [event["event"] for event in state["events"]])


class BackupsAndCapTests(unittest.TestCase):
    def test_one_is_selected_and_the_rest_become_ordered_backups(self):
        ideas = _round_ideas(3)
        order = _order(ideas, ideas[0]["idea_id"])
        reports = _reports(ideas, order=order)
        selection = scoring_module.evaluate(ideas, reports)["selection"]
        self.assertEqual(len(selection["selected"]), 1)
        self.assertTrue(selection["backups"])
        self.assertEqual([entry["idea_id"] for entry in selection["backups"]],
                         [entry["idea_id"] for entry in selection["backups"]])
        self.assertEqual(len(selection["backups"]), len(ideas) - 1)
        totals = [entry["total_rank"] for entry in selection["backups"]]
        self.assertEqual(totals, sorted(totals))


if __name__ == "__main__":
    unittest.main()
