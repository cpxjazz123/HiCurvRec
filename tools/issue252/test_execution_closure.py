"""The real generated entry's imported repository code belongs to the archive."""
from pathlib import Path
import tempfile
import unittest
from tools.issue252.archive_audit import imported_repo_sources,sha256

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())


class ExecutionClosureTests(unittest.TestCase):
    def test_actual_adapter_dependency_is_bound_even_when_trainer_does_not_import_it(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'candidate';tree.mkdir();(repo/'tools').mkdir()
            (tree/'train_rqvae.py').write_text('import math\n')
            (tree/'execution_entry.py').write_text('from tools.adapter_dependency import VALUE\n')
            dependency=repo/'tools/adapter_dependency.py';dependency.write_text('VALUE=42\n')
            sources=imported_repo_sources(tree,repo)
            self.assertEqual(sources.get('tools/adapter_dependency.py'),sha256(dependency),
                             'actual framework adapter imports must be included in executed-source provenance')


if __name__=='__main__':unittest.main()
