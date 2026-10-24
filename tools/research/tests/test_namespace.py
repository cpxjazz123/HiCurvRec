"""The active research namespace cannot silently reuse the retired issue's queue."""
import unittest
from pathlib import Path

from tools.research import settings
from tools.research.verification import admission


class CurrentExperimentNamespaceTests(unittest.TestCase):
    def test_new_issue_has_separate_root_and_verifier(self):
        self.assertEqual(settings.STATE_ROOT.name, "issue117")
        self.assertEqual(admission.EVIDENCE_ROOT, settings.STATE_ROOT)
        self.assertNotEqual(settings.STATE_ROOT, settings.ROOT / "results" / "research" / "issue116")

    def test_registered_mcp_and_dag_refer_to_current_issue(self):
        root = settings.ROOT
        for relative in ("tools/research/workflows/pipeline.py",
                         "tools/research/mcp/handlers.py",
                         "tools/genrec-mcp/server.py"):
            text = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("Issue116", text)
            self.assertNotIn("issue116", text)
        self.assertIn('issue117', (root / "tools/research/settings.py").read_text())


if __name__ == "__main__":
    unittest.main()
