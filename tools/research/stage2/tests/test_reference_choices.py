"""An argument declared as a set of choices must be sampled from that set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.research.stage2.reference import _inputs


def test_a_choice_argument_stays_on_its_choices():
    call = {"args": [{"name": "h", "width": 3, "choices": [1.0, 2.0]}]}
    arguments, _ = _inputs(call, seed=17)
    values = arguments["h"].detach()
    assert set(values.flatten().tolist()) <= {1.0, 2.0}, values
