"""The report keeps the three gradient states, and importing it writes nothing."""
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import probe_q8


def test_report_keeps_none_zero_and_nonzero_per_block():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    c = torch.nn.Parameter(torch.zeros(2))
    out = probe_q8.report_blocks({"blk": [a, b, c]},
                                 {"blk": [None, torch.zeros(2), torch.ones(2)]})
    assert out["blk"]["none_tensors"] == 1
    assert out["blk"]["zero_tensors"] == 1
    assert out["blk"]["nonzero_tensors"] == 1
    assert out["blk"]["tensors"] == 3


def test_importing_the_report_module_writes_nothing():
    with tempfile.TemporaryDirectory() as work:
        code = ("import sys; sys.path.insert(0, %r); import probe_q8; "
                "print('imported')" % str(ROOT))
        done = subprocess.run([sys.executable, "-c", code], cwd=work,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert list(Path(work).iterdir()) == [], "the import created files"


if __name__ == "__main__":
    test_report_keeps_none_zero_and_nonzero_per_block()
    test_importing_the_report_module_writes_nothing()
    print("ok")
