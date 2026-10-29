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


class ManifestTests(unittest.TestCase):
    """A reviewer must know what exists even when its scope cannot read it.

    The alternative produced a blocking finding that a module was "absent from the
    delivered tree" when the module existed and simply was not in that role's read
    scope.
    """

    def test_the_manifest_lists_every_source_file_with_its_size(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            tree = root / "trees" / "idea"
            (tree / "model").mkdir(parents=True)
            (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            (tree / "model" / "model.py").write_text("y = 2\nz = 3\n", encoding="utf-8")
            path = worker_module.write_manifest(root, "idea", 2, tree)
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("train_rqvae.py", text)
            self.assertIn("model/model.py", text)
            self.assertIn("lines", text)

    def test_every_reviewer_receives_the_manifest(self):
        source = (Path(__file__).resolve().parents[1] / "rounds" /
                  "worker.py").read_text(encoding="utf-8")
        self.assertIn("write_manifest(", source)
        block = source[source.index("def run_verification"):]
        block = block[:block.index("def verify(")]
        self.assertIn("manifest", block)
        self.assertIn("role_read_paths(tree, role)", block)
        self.assertIn("findings_path", block)
        self.assertIn("manifest", block)


class FullChangeListTests(unittest.TestCase):
    """Role selection must not depend on the truncated prompt text."""

    def _trees(self, work):
        root = Path(work)
        tree = root / "trees" / "idea"
        (tree / "model").mkdir(parents=True)
        (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
        (tree / "model" / "m.py").write_text("y = 1\n", encoding="utf-8")
        worker_module.snapshot_tree(root, "idea", 0, tree)
        return root, tree

    def test_a_change_past_the_truncation_is_still_dispatched(self):
        with tempfile.TemporaryDirectory() as work:
            root, tree = self._trees(work)
            (tree / "model" / "m.py").write_text("# long\n" * 40000, encoding="utf-8")
            (tree / "train_rqvae.py").write_text("x = 2\n", encoding="utf-8")
            patch = worker_module.tree_diff(root, "idea", 0, tree)
            self.assertIn("truncated", patch)
            self.assertNotIn("train_rqvae.py", worker_module.changed_files(patch))
            files = worker_module.full_change_list(root, "idea", 0, tree)
            self.assertIn("train_rqvae.py", files)
            roles = worker_module.affected_roles("", files=files)
            self.assertIn("integration-resources", roles)

    def test_a_deletion_changes_the_file_list(self):
        with tempfile.TemporaryDirectory() as work:
            root, tree = self._trees(work)
            (tree / "train_rqvae.py").unlink()
            files = worker_module.full_change_list(root, "idea", 0, tree)
            self.assertIn("train_rqvae.py", files)

    def test_the_reviewer_number_is_the_full_list_position(self):
        source = (Path(__file__).resolve().parents[1] / "rounds" /
                  "worker.py").read_text(encoding="utf-8")
        self.assertIn("role_index[roles[index - 1][\"id\"]]", source)


class FindingOwnershipTests(unittest.TestCase):
    """An unresolved blocking finding stays until its own role re-reviews."""

    def test_a_finding_keeps_its_role_across_passes(self):
        from tools.research.rounds import pipeline as pipeline_module
        with tempfile.TemporaryDirectory() as work:
            directory = pipeline_module.new_round(Path(work), "round_own", objective="o")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [{
                "implementation_id": "i", "idea_id": "i", "tree": str(directory),
                "status": "implemented", "revision": 0}]
            state = pipeline_module.apply_verification(state, "i", [{
                "verifier": "verify_i_r0_3", "role": "distributed-state",
                "verdict": "NEEDS_FIX", "checks": [{"name": "c", "verdict": "NEEDS_FIX",
                                                    "findings": [{"severity": "blocking",
                                                                  "location": "f.py:1",
                                                                  "detail": "d"}]}],
                "boundary_version": "b", "boundary_sha256": "0" * 64,
                "missing_reviewers": []}])
            record = state["implementations"][0]
            self.assertEqual(record["blocking_findings"][0]["role"], "distributed-state")

    def test_an_old_finding_survives_a_pass_by_another_role(self):
        from tools.research.rounds import pipeline as pipeline_module
        with tempfile.TemporaryDirectory() as work:
            directory = pipeline_module.new_round(Path(work), "round_own2", objective="o")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [{
                "implementation_id": "i", "idea_id": "i", "tree": str(directory),
                "status": "implemented", "revision": 0}]
            state = pipeline_module.apply_verification(state, "i", [{
                "verifier": "verify_i_r0_3", "role": "distributed-state",
                "verdict": "NEEDS_FIX", "checks": [{"name": "c", "verdict": "NEEDS_FIX",
                                                    "findings": [{"severity": "blocking",
                                                                  "location": "f.py:1",
                                                                  "detail": "d"}]}],
                "boundary_version": "b", "boundary_sha256": "0" * 64,
                "missing_reviewers": []}])
            state = pipeline_module.apply_verification(state, "i", [{
                "verifier": "verify_i_r1_1", "role": "math-geometry",
                "verdict": "PASS", "checks": [],
                "boundary_version": "b", "boundary_sha256": "0" * 64,
                "missing_reviewers": []}])
            record = state["implementations"][0]
            self.assertTrue(record.get("blocking_findings"),
                            "a PASS from another role must not close an open blocker")


class ReadCoverageTests(unittest.TestCase):
    """The allowlist must match the alias the agent uses, and gaps are recorded."""

    def test_the_scope_is_listed_under_both_aliases(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "tree"
            (tree / "model").mkdir(parents=True)
            (tree / "model" / "m.py").write_text("x = 1\n", encoding="utf-8")
            paths = worker_module.role_read_paths(tree, {"reads": ["model/m.py"]})
            self.assertGreaterEqual(len(paths), 1)
            self.assertIn(str(tree / "model" / "m.py"), paths)
            self.assertIn(str((tree / "model" / "m.py").resolve()), paths)

    def test_an_unreadable_entry_is_reported_as_a_gap(self):
        with tempfile.TemporaryDirectory() as work:
            missing = Path(work) / "nope.py"
            gaps = worker_module.read_gaps([str(missing)])
            self.assertEqual(len(gaps), 1)
            readable = Path(work) / "ok.py"
            readable.write_text("x = 1\n", encoding="utf-8")
            self.assertEqual(worker_module.read_gaps([str(readable)]), [])

    def test_a_role_without_existing_files_gets_no_file_at_all(self):
        """The fallback widened the scope and hid the gap it should have reported.

        The tree's directory is still declared - the agent has to be able to LIST
        its scope - but no FILE outside the scope is granted, and in particular no
        file of a scope that does not exist.
        """
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "tree"
            tree.mkdir()
            (tree / "model").mkdir()
            (tree / "model" / "secret.py").write_text("x = 1\n", encoding="utf-8")
            paths = worker_module.role_read_paths(tree, {"reads": ["model/absent.py"]})
            self.assertNotIn(str(tree / "model" / "secret.py"), paths)
            self.assertEqual([path for path in paths if path.endswith(".py")], [])


class CoverageGapTests(unittest.TestCase):
    """A report whose reads were refused is not a source review."""

    def test_the_report_marks_refused_reads(self):
        source = (Path(__file__).resolve().parents[1] / "rounds" /
                  "worker.py").read_text(encoding="utf-8")
        self.assertIn('report["source_review"] = False', source)
        self.assertIn('"kind": "read_denied"', source)


class ListingTests(unittest.TestCase):
    """An agent must be able to LIST its scope, not only open the files in it."""

    def test_the_tree_directory_is_in_the_scope(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "tree"
            (tree / "model").mkdir(parents=True)
            (tree / "model" / "m.py").write_text("x = 1\n", encoding="utf-8")
            paths = worker_module.role_read_paths(tree, {"reads": ["model/m.py"]})
            self.assertIn(str(tree), paths)
            self.assertIn(str(tree.resolve()), paths)


class CachedReportRoleTests(unittest.TestCase):
    """A reused report still says which role it belongs to."""

    def test_the_role_is_read_from_the_task_number(self):
        from tools.research.rounds import agents as agent_contracts
        roles = agent_contracts.VERIFIER_ROLES
        for index, role in enumerate(roles, 1):
            task = "verify_idea_r2_%d" % index
            self.assertEqual(worker_module.role_of_task(task), role["id"])


class PromptSizeTests(unittest.TestCase):
    """The prompt points at the long documents instead of repeating them."""

    def test_the_objective_is_one_line_in_the_prompt(self):
        from tools.research.rounds import agents as agent_contracts
        self.assertIn("{objective_summary}", agent_contracts.VERIFY_PROMPT)
        self.assertNotIn("THE RESEARCH OBJECTIVE:\\n{objective}",
                         agent_contracts.VERIFY_PROMPT)

    def test_the_scope_contract_is_a_file(self):
        with tempfile.TemporaryDirectory() as work:
            path = worker_module.scope_file(Path(work))
            self.assertTrue(path.is_file())
            self.assertIn("scope", path.read_text(encoding="utf-8").lower())
