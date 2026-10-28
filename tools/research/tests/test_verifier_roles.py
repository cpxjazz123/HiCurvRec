"""The five reviewers are five roles, not five copies of one review.

Coverage is the product of what each reviewer is told to own; five identical
prompts make it a lottery. These checks pin the division of labour and the
brief every reviewer reads: the same mechanism description and the previous
review's findings, so a repair is judged against what it claimed to fix.
"""
import unittest
from pathlib import Path

from tools.research.rounds import agents as agent_contracts
from tools.research.rounds import worker as worker_module


class RoleContractTests(unittest.TestCase):

    def test_there_are_five_roles_with_unique_ids(self):
        from tools.research.rounds import pipeline as pipeline_module
        roles = agent_contracts.VERIFIER_ROLES
        self.assertEqual(len(roles), pipeline_module.VERIFIER_AGENTS)
        ids = [role["id"] for role in roles]
        self.assertEqual(len(set(ids)), len(ids))

    def test_every_role_states_what_it_must_cover(self):
        for role in agent_contracts.VERIFIER_ROLES:
            self.assertGreaterEqual(len(role["must"]), 3, role["id"])
            self.assertTrue(role["scope"].strip(), role["id"])
            self.assertGreaterEqual(len(role["checks"]), 2, role["id"])
            for name, guidance in role["checks"]:
                self.assertTrue(name.strip() and guidance.strip(), role["id"])

    def test_the_roles_cover_the_five_failure_classes(self):
        expected = {"math-geometry", "gradient-update", "distributed-state",
                    "behaviour-sid", "integration-resources"}
        self.assertEqual({role["id"] for role in agent_contracts.VERIFIER_ROLES}, expected)

    def test_the_scopes_together_cover_the_mechanism(self):
        scopes = " ".join(role["scope"] for role in agent_contracts.VERIFIER_ROLES)
        for path in ("model/frechet_poincare.py", "model/layers.py", "train_rqvae.py",
                     "tests/test_geometry.py", "tests/test_ddp.py", "tests/test_paths.py"):
            self.assertIn(path, scopes)

    def test_no_two_roles_share_their_check_names(self):
        names = [name for role in agent_contracts.VERIFIER_ROLES
                 for name, _ in role["checks"]]
        self.assertEqual(len(set(names)), len(names))


class PromptTests(unittest.TestCase):

    def _prompts(self, record=None):
        idea = {"idea_id": "idea-x", "title": "t", "claim": "c", "mechanism": "m",
                "geometry": "g", "behaviour_signal": "b", "coupling": "x",
                "specification": "s", "switches": "w", "falsification": "f",
                "novelty": "n"}
        prompts = {}
        for index, role in enumerate(agent_contracts.VERIFIER_ROLES, 1):
            prompts[index] = agent_contracts.VERIFY_PROMPT.format(
                role_index=index, role_total=len(agent_contracts.VERIFIER_ROLES),
                role_name=role["name"],
                role_must="\n".join("  - " + item for item in role["must"]),
                role_scope=role["scope"],
                role_checks=worker_module._format_role_checks(role),
                idea=worker_module.plaintext.render_idea(idea), tree="/tmp/tree",
                change_brief=worker_module._format_change_brief(record or {}),
                change_diff="diff --git a/model/frechet_poincare.py b/model/frechet_poincare.py\n",
                findings_file="/tmp/findings.json",
                manifest_file="/tmp/manifest.txt",
                previous_findings=worker_module._format_findings(
                    (record or {}).get("blocking_findings")),
                objective=agent_contracts.RESEARCH_OBJECTIVE,
                scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
                agent_contracts.EXPERIMENT_POLICY)
        return prompts

    def test_every_prompt_is_filled_and_distinct(self):
        prompts = self._prompts()
        self.assertEqual(len(set(prompts.values())), len(prompts))
        for prompt in prompts.values():
            self.assertNotIn("{", prompt.replace("{\"", ""))
            self.assertIn("YOUR AREA", prompt)
            self.assertIn("VERDICT: PASS or NEEDS_FIX", prompt)

    def test_each_prompt_carries_its_own_checks_only(self):
        prompts = self._prompts()
        for index, role in enumerate(agent_contracts.VERIFIER_ROLES, 1):
            prompt = prompts[index]
            for name, _ in role["checks"]:
                self.assertIn(name, prompt)
            for other in agent_contracts.VERIFIER_ROLES:
                if other["id"] == role["id"]:
                    continue
                for name, _ in other["checks"]:
                    self.assertNotIn(name, prompt, "role %s leaked %s" % (role["id"], other["id"]))

    def test_the_brief_and_the_previous_findings_reach_every_reviewer(self):
        record = {
            "changed_files": ["model/frechet_poincare.py", "train_rqvae.py"],
            "changes": "the Mobius sum now carries a plus",
            "smoke": "two ranks, three steps, no reducer error",
            "blocking_findings": [{
                "severity": "blocking", "location": "train_rqvae.py:344-345",
                "detail": "the codebook parameter never receives a gradient"}],
        }
        prompts = self._prompts(record)
        for index in prompts:
            prompt = prompts[index]
            self.assertIn("model/frechet_poincare.py", prompt)
            self.assertIn("the Mobius sum now carries a plus", prompt)
            self.assertIn("two ranks, three steps", prompt)
            self.assertIn("train_rqvae.py:344-345", prompt)
            self.assertIn("never receives a gradient", prompt)

    def test_a_first_review_says_so(self):
        self.assertIn("first review", worker_module._format_findings([]))


class AnswerShapeTests(unittest.TestCase):
    """The reviewers' answer format did not change with the roles."""

    def test_the_parser_reads_a_role_shaped_answer(self):
        from tools.research.rounds.plaintext import parse_verification
        text = "\n".join([
            "VERDICT: NEEDS_FIX",
            "CHECK: mathematics and geometry: formulas, domains and identities",
            "CHECK VERDICT: NEEDS_FIX",
            "EXERCISED PATH: model/frechet_poincare.py:31 | hand evaluation at c=1",
            "FINDING SEVERITY: blocking",
            "FINDING LOCATION: model/frechet_poincare.py:31",
            "FINDING DETAIL: the second term carries the wrong sign",
            "FINDING REPRODUCTION: 0 (+) 0.3 = -0.3",
            "FINDING CORRECTION: use plus",
        ])
        report = parse_verification(text)
        self.assertEqual(report["verdict"], "NEEDS_FIX")
        self.assertEqual(report["checks"][0]["findings"][0]["severity"], "blocking")


if __name__ == "__main__":
    unittest.main()
