import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.research.verification import workflow


class WorkflowSafetyTests(unittest.TestCase):
    def test_agents_disabled_by_default(self):
        with patch.object(workflow.settings, "LIVE_AGENTS_ALLOWED", False):
            with self.assertRaises(PermissionError):
                workflow.draft_test_suites("new_candidate", Path("/tmp/absent"))

    def test_reviewed_files_absent_refuse_freeze(self):
        with tempfile.TemporaryDirectory() as work:
            repo = Path(work)
            with patch.object(workflow, "_inputs", return_value={
                "source_sha256": "a" * 64, "proposal_sha256": "b" * 64}):
                with patch.object(workflow, "CASES_ROOT", repo):
                    with self.assertRaises(workflow.AdmissionError):
                        workflow.freeze_reviewed_manifest("new_candidate", repo)


if __name__ == "__main__":
    unittest.main()
