import json
import os
import tempfile
import unittest
from pathlib import Path
from tools.research.agents.runner import run_agent
from tools.research.core.evidence import read_json


class AgentTests(unittest.TestCase):
    def test_fake_opencode_is_real_subprocess_not_a_marker(self):
        with tempfile.TemporaryDirectory() as work:
            binary = Path(work) / "fake-opencode"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:4] == ['run','--standalone','--format']\n"
                "Path('answer.json').write_text(json.dumps({'ok': True}))\n"
                "print(json.dumps({'type': 'mock', 'tool': 'none'}))\n",
                encoding="utf-8",
            )
            binary.chmod(0o700)
            receipt = run_agent("test_agent", "Write answer", "answer.json",
                                Path(work) / "out", binary=str(binary), timeout=10)
            evidence = read_json(receipt)
            self.assertTrue(evidence["verified"])
            self.assertGreater(evidence["pid"], 0)
            self.assertEqual(evidence["exit_code"], 0)
            self.assertTrue((receipt.parent / "session.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
