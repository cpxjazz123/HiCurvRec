"""The incremental review: a diff, a targeted dispatch, and a per-role read scope.

Three things were missing and each one made a revision review weaker than it
looked: the reviewers got the previous findings but not the code that changed,
every revision re-ran all five roles whatever it touched, and each role read the
whole tree although the division of labour is what makes five reviewers useful.
"""
import tempfile
import unittest
from pathlib import Path

from tools.research.rounds import agents as agent_contracts
from tools.research.rounds import worker as worker_module


class RoleReadScopeTests(unittest.TestCase):
    """Every role declares the files its area is judged on."""

    def test_every_role_declares_a_read_scope(self):
        for role in agent_contracts.VERIFIER_ROLES:
            self.assertTrue(role.get("reads"), role["id"])
            for path in role["reads"]:
                self.assertFalse(str(path).startswith("/"), role["id"])
                self.assertNotIn("..", str(path), role["id"])

    def test_the_declared_scopes_cover_the_mechanism_files(self):
        scopes = [path for role in agent_contracts.VERIFIER_ROLES for path in role["reads"]]
        for path in ("train_rqvae.py", "model/layers.py", "model/model.py",
                     "model/frechet_poincare.py"):
            self.assertIn(path, scopes)

    def test_the_geometry_role_does_not_get_the_trainer(self):
        geometry = [role for role in agent_contracts.VERIFIER_ROLES
                    if role["id"] == "math-geometry"][0]
        self.assertNotIn("train_rqvae.py", geometry["reads"])

    def test_no_role_reads_the_whole_tree(self):
        """A directory listing is allowed, an unconditional tree read is not."""
        for role in agent_contracts.VERIFIER_ROLES:
            self.assertNotIn("*", role["reads"], role["id"])
            for path in role["reads"]:
                if path == ".":
                    continue
                self.assertTrue(path.endswith((".py", ".md", ".json")), (role["id"], path))

    def test_the_root_listing_does_not_make_a_role_own_every_change(self):
        """integration-resources may list the tree but must not be dispatched
        for a change that no file in its scope touches."""
        roles = worker_module.affected_roles(
            "diff --git a/model/frechet_poincare.py b/model/frechet_poincare.py\n")
        self.assertNotIn("integration-resources", roles)


class SnapshotAndDiffTests(unittest.TestCase):

    def test_a_registration_snapshots_the_tree_and_the_next_one_diffs_it(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            tree = root / "trees" / "idea"
            (tree / "model").mkdir(parents=True)
            (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            (tree / "model" / "m.py").write_text("y = 1\n", encoding="utf-8")
            first = worker_module.snapshot_tree(root, "idea", 0, tree)
            self.assertTrue((Path(first) / "train_rqvae.py").is_file())

            (tree / "model" / "m.py").write_text("y = 2\n", encoding="utf-8")
            patch = worker_module.tree_diff(root, "idea", 0, tree)
            self.assertIn("train_rqvae.py".replace("train_rqvae.py", "model/m.py"), patch)
            self.assertIn("-y = 1", patch)
            self.assertIn("+y = 2", patch)

    def test_a_run_artifact_is_not_part_of_the_diff(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            tree = root / "trees" / "idea"
            (tree / "logs").mkdir(parents=True)
            (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            worker_module.snapshot_tree(root, "idea", 0, tree)
            (tree / "logs" / "train.log").write_text("noise\n", encoding="utf-8")
            self.assertEqual(worker_module.tree_diff(root, "idea", 0, tree).strip(), "")

    def test_the_diff_is_bounded(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            tree = root / "trees" / "idea"
            tree.mkdir(parents=True)
            (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            worker_module.snapshot_tree(root, "idea", 0, tree)
            (tree / "train_rqvae.py").write_text("y = 2\n" * 20000, encoding="utf-8")
            patch = worker_module.tree_diff(root, "idea", 0, tree)
            self.assertLessEqual(len(patch), worker_module.DIFF_LIMIT + 200)
            self.assertIn("truncated", patch)


class TargetedDispatchTests(unittest.TestCase):

    def test_only_the_roles_that_own_a_changed_file_are_dispatched(self):
        patch = "diff --git a/model/frechet_poincare.py b/model/frechet_poincare.py\n"
        roles = worker_module.affected_roles(patch)
        self.assertIn("math-geometry", roles)
        self.assertNotIn("integration-resources", roles)

    def test_a_role_with_an_open_finding_is_always_dispatched(self):
        patch = "diff --git a/model/frechet_poincare.py b/model/frechet_poincare.py\n"
        findings = [{"verifier": "r0_3", "role": "distributed-state",
                     "location": "train_rqvae.py:344"}]
        roles = worker_module.affected_roles(patch, findings, reported=["r0_3"])
        self.assertIn("distributed-state", roles)

    def test_a_change_to_the_trainer_reaches_the_roles_that_read_it(self):
        patch = "diff --git a/train_rqvae.py b/train_rqvae.py\n"
        roles = worker_module.affected_roles(patch)
        self.assertIn("gradient-update", roles)
        self.assertIn("distributed-state", roles)
        self.assertIn("integration-resources", roles)

    def test_a_review_always_dispatches_at_least_one_role(self):
        self.assertTrue(worker_module.affected_roles(""))


class FindingsFileTests(unittest.TestCase):

    def test_the_full_findings_list_is_written_for_the_reviewers(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            findings = [{"verifier": "r0_%d" % index, "location": "f.py:%d" % index,
                         "detail": "d" * 500, "severity": "blocking"}
                        for index in range(30)]
            path = worker_module.write_findings_file(root, "idea", 1, findings)
            self.assertTrue(Path(path).is_file())
            import json
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(len(payload["findings"]), 30)


if __name__ == "__main__":
    unittest.main()
