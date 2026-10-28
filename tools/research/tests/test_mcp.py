"""MCP control-plane regression; no OpenCode, GPU or dataset needed."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.research import settings
from tools.research.mcp import handlers, verification_handlers
from tools.research.workflows import pipeline


class ResearchMCPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patches = [
            patch.object(settings, "STATE_ROOT", self.root),
            patch.object(settings, "LIVE_AGENTS_ALLOWED", False),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_start_uses_one_root_dag_and_safe_data_only_worker(self):
        with patch.object(handlers, "_spawn_worker", return_value=12345) as spawn:
            res = handlers.research_start()
        self.assertEqual(len(res["run_id"]), 20)
        self.assertEqual(res["workers_started"], [12345])
        spawn.assert_called_once()
        snap = pipeline.store().snapshot()["tasks"]
        prefix = res["run_id"] + "__"
        self.assertEqual(set(snap), {prefix + name for name in handlers.ROLES})
        self.assertEqual(snap[prefix + "geometry"]["dependencies"],
                         [prefix + "behavior", prefix + "literature"])
        self.assertFalse(res["gpu_launched"])
        self.assertEqual(handlers.research_status(res["run_id"])["status"], "queued")
        self.assertEqual(handlers.research_results(res["run_id"])["artifacts"], {})

    def test_live_mode_fans_out_and_cancel_is_fail_closed(self):
        with patch.object(settings, "LIVE_AGENTS_ALLOWED", True):
            with patch.object(handlers, "_spawn_worker", side_effect=[101, 102]) as launch:
                result = handlers.research_start()
            self.assertEqual(launch.call_count, 2)
            rid = result["run_id"]
            cancel = handlers.research_cancel(rid)
            self.assertTrue(cancel["cancel_requested"])
            self.assertTrue((handlers._root(rid) / "cancel.json").is_file())
            snap = pipeline.store().snapshot()["tasks"]
            self.assertTrue(all(snap[rid + "__" + name]["status"] == "cancelled"
                                for name in handlers.ROLES))
            with patch.object(handlers, "_spawn_worker") as launch_again:
                out = handlers.research_resume(rid)
                self.assertEqual(out["reason"], "cancel_requested")
                launch_again.assert_not_called()
            self.assertEqual(handlers.research_status(rid)["status"], "cancelled")

    def test_real_evidence_and_mutation_of_bytes(self):
        with patch.object(handlers, "_spawn_worker", return_value=333):
            rid = handlers.research_start()["run_id"]
        root = handlers._root(rid)
        claim = pipeline.store().claim("test-owner", roles=("DATA",), prefix=rid + "__")
        self.assertEqual(claim, (rid + "__behavior", "DATA"))
        path = root / "behavior.json"
        path.write_text('{"population":"train_only","transitions":3}', encoding="utf-8")
        pipeline.store().complete(claim[0], "test-owner", path)
        summary = handlers.research_results(rid)["artifacts"]["behavior"]
        self.assertTrue(summary["verified"])
        self.assertIn("transitions", summary["preview"])
        path.write_text('{"forged":true}', encoding="utf-8")
        bad = handlers.research_results(rid)["artifacts"]["behavior"]
        self.assertFalse(bad["verified"])

    def test_finished_research_is_not_relabelled_cancelled(self):
        with patch.object(handlers, "_spawn_worker", return_value=333):
            rid = handlers.research_start()["run_id"]
        root = handlers._root(rid)
        for name in handlers.ROLES:
            task, role = pipeline.store().claim("test", prefix=rid + "__")
            self.assertEqual(task, rid + "__" + name)
            path = root / (name + ".json")
            path.write_text('{"finished":true}', encoding="utf-8")
            pipeline.store().complete(task, "test", path)
        result = handlers.research_cancel(rid)
        self.assertFalse(result["cancel_requested"])
        self.assertFalse((root / "cancel.json").exists())
        self.assertEqual(handlers.research_status(rid)["status"], "completed")

    def test_mutating_agent_receipt_invalidates_returned_result(self):
        with patch.object(handlers, "_spawn_worker", return_value=333):
            rid = handlers.research_start()["run_id"]
        root = handlers._root(rid)
        task, _ = pipeline.store().claim("test", roles=("LITERATURE",), prefix=rid + "__")
        folder = root / "agents" / "literature"
        folder.mkdir(parents=True)
        output = folder / "literature.json"
        output.write_text('{"sources":["paper"]}', encoding="utf-8")
        receipt = folder / "execution.json"
        from tools.research.core.evidence import digest_file
        receipt.write_text(json.dumps({"output_sha256": digest_file(output)}), encoding="utf-8")
        pipeline.store().complete(task, "test", receipt)
        self.assertTrue(handlers.research_results(rid)["artifacts"]["literature"]["verified"])
        receipt.write_text(json.dumps({"output_sha256": digest_file(output), "changed": True}))
        self.assertFalse(handlers.research_results(rid)["artifacts"]["literature"]["verified"])

    def test_running_cancel_requires_cooperative_completion(self):
        with patch.object(handlers, "_spawn_worker", return_value=333):
            rid = handlers.research_start()["run_id"]
        claimed = pipeline.store().claim("owner", prefix=rid + "__")
        self.assertEqual(claimed[1], "DATA")
        handlers.research_cancel(rid)
        self.assertEqual(pipeline.store().snapshot()["tasks"][claimed[0]]["status"], "cancelling")
        pipeline.store().finish_cancel(claimed[0], "owner")
        self.assertEqual(pipeline.store().snapshot()["tasks"][claimed[0]]["status"], "cancelled")

    def test_unknown_and_unsafe_run_ids(self):
        for val in ("../escape", "run_fake", "run_" + "0" * 16):
            with self.assertRaises(ValueError):
                handlers.research_results(val)

    def test_verifier_authoring_refuses_disabled_agent(self):
        response = verification_handlers.research_verify_start(
            "run_0123456789abcdef", "some-source")
        self.assertFalse(response["started"])
        self.assertEqual(response["reason"], "live_opencode_disabled_in_settings")

    def test_mcp_registry_and_dispatch_are_real(self):
        src = settings.ROOT / "tools" / "genrec-mcp" / "server.py"
        spec = importlib.util.spec_from_file_location("genrec_mcp_test", src)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        names = ("research_start", "research_resume", "research_status",
                 "research_results", "research_cancel", "research_verify_start",
                 "research_verify_review", "research_verify_status")
        self.assertTrue(set(names).issubset(server.TOOLS))
        self.assertEqual(server.tool_schema("research_start")["inputSchema"]["required"], [])
        self.assertEqual(server.tool_schema("research_cancel")["inputSchema"]["required"], ["run_id"])
        with patch.object(server, "_research_endpoints",
                          return_value=(handlers, verification_handlers)):
            with patch.object(handlers, "_spawn_worker", return_value=31337):
                raw, error = server.dispatch_tool("research_start", {})
            self.assertFalse(error)
            self.assertIn("run_id", json.loads(raw))


if __name__ == "__main__":
    unittest.main()


class ConformanceCacheRootTests(unittest.TestCase):
    """The conformance cache writes under the repository, not into a NameError."""

    def test_the_report_runs_without_an_undefined_root(self):
        import importlib.util
        from pathlib import Path
        import tempfile
        server_path = Path(__file__).resolve().parents[2] / "genrec-mcp" / "server.py"
        spec = importlib.util.spec_from_file_location("genrec_server_conformance", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "TIGER_probe"
            tree.mkdir()
            (tree / "train_rqvae.py").write_text("ARM_NAME = 'TIGER_probe'\n", encoding="utf-8")
            report = module._conformance_report(tree, "digestprobe")
        self.assertIsNotNone(report.get("passed"))
        self.assertTrue((module.REPO / "results" / "research" / "_conformance").is_dir())
