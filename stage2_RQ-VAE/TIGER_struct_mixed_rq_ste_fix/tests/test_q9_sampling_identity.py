"""The reported identity must describe the triples the loss actually used."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import probe_q9

Q8 = Path("/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q8/factors.json")


def consistent(report):
    ident = report["identity"]
    rows = ident.get("z_struct_rows")
    if rows is None:                       # q8 kept it under shapes
        rows = report.get("shapes", {}).get("z_struct_rows")
    return rows == 3 * ident["triples_accepted"]


def test_the_historical_q8_identity_does_not_match_its_rows():
    report = json.loads(Q8.read_text())
    assert consistent(report) is False, (
        "q8 reported z_struct_rows=99 as if it belonged to 45 triples; "
        "this case records that mismatch instead of hiding it")


def test_the_q9_report_describes_the_triples_it_used():
    with tempfile.TemporaryDirectory() as work:
        report = probe_q9.run(Path(work))
        assert consistent(report) is True
        assert report["identity"]["triples_requested"] >= report["identity"]["triples_accepted"]
        assert report["identity"]["z_struct_rows"] == 3 * report["identity"]["triples_accepted"]
        assert report["diffs"]["base_vs_total_z_base"] == 0.0
        assert report["diffs"]["struct_vs_total_z_struct"] == 0.0


if __name__ == "__main__":
    test_the_historical_q8_identity_does_not_match_its_rows()
    test_the_q9_report_describes_the_triples_it_used()
    print("ok")
