"""Regression: starting Stage3 performs no admission check (owner 2026-11-09).

The launch used to refuse a SID whose technical validity, collision gate or
layer record failed. That is now forbidden: the layer record is still written,
but a failure is reported and the launch proceeds.

The test imports the real server module, forces the layer record to fail, mocks
the GPU probe, the harness writer and the process spawn, and asserts the launch
is started. It also asserts the admission wrapper no longer exists.

    python3 -m tools.research.stage2.tests.test_stage3_no_admission
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[4]
SERVER = REPO / "tools" / "genrec-mcp" / "server.py"
sys.path.insert(0, str(REPO))


def _server():
    spec = importlib.util.spec_from_file_location("genrec_server_no_admission", SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoAdmissionTest(unittest.TestCase):
    def test_a_failed_layer_record_does_not_refuse_the_launch(self):
        server = _server()
        sid = server.RESULT2_ROOT / "TIGER_RQ-VAE" / "item_sids.json"
        self.assertTrue(sid.is_file(), "the frozen baseline SID is the test fixture")

        class FakeChild:
            pid = 424242

        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(server, "RESULT3_ROOT", Path(raw) / "stage3"), \
                    mock.patch.object(server, "record_sid_layers",
                                      lambda _sid: {"passed": False, "reason": "forced"}), \
                    mock.patch.object(server, "frozen_protocol_check",
                                      lambda: {"ok": True}), \
                    mock.patch.object(server, "probe_gpu_layout",
                                      lambda: {"mode": "single-gpu"}), \
                    mock.patch.object(server, "stage3_supports_single_gpu", lambda: True), \
                    mock.patch.object(server, "stage3_launch_decision", lambda _probe: None), \
                    mock.patch.object(server, "stage3_processes", lambda: []), \
                    mock.patch.object(server, "write_stage3_harness",
                                      lambda *_a, **_k: Path(raw) / "harness.py"), \
                    mock.patch.object(server, "save_state", lambda *_a, **_k: None), \
                    mock.patch.object(server.subprocess, "Popen",
                                      lambda *_a, **_k: FakeChild()):
                result = server.tool_stage3_start(str(sid))

        self.assertTrue(result.get("started"), result)
        self.assertFalse(result["layer_record"]["passed"])
        self.assertNotIn("validity", result)
        self.assertNotIn("refused", result)

    def test_the_admission_wrapper_is_gone(self):
        server = _server()
        self.assertFalse(hasattr(server, "stage2_validity_admission"),
                         "no admission wrapper may remain on the launch path")


def main() -> int:
    failures = 0
    for test in (NoAdmissionTest.test_a_failed_layer_record_does_not_refuse_the_launch,
                 NoAdmissionTest.test_the_admission_wrapper_is_gone):
        try:
            instance = NoAdmissionTest()
            getattr(instance, test.__name__)()
            print("PASS " + test.__name__, flush=True)
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc), flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR " + test.__name__ + ": " + repr(exc), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
