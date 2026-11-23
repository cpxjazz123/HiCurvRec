"""External graph loaders must not silently receive NOT_APPLICABLE."""
from pathlib import Path
import tempfile
import unittest
from tools.research.stage2.graph_contract import inspect_tree

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())


class ExternalGraphReaderTests(unittest.TestCase):
    def inspect(self,source):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            tree=Path(directory);(tree/'model.py').write_text(source)
            return inspect_tree(tree,folder=tree/'canonical_not_built')

    def test_shared_ego_reader_is_foreign_not_inapplicable(self):
        result=self.inspect('from tools.graph_infra.ego_gnn import EgoStore\nstore=EgoStore(ROOT,ROOT/"assets")\n')
        self.assertEqual(result['status'],'FAIL')
        self.assertIn('model.py',result['foreign'])

    def test_reader_import_alias_cannot_hide_graph_use(self):
        result=self.inspect('from tools.graph_infra.ego_gnn import EgoStore as LocalStore\nstore=LocalStore(ROOT)\n')
        self.assertEqual(result['status'],'FAIL')
        self.assertIn('model.py',result['foreign'])

    def test_module_import_alias_cannot_hide_graph_use(self):
        result=self.inspect('import tools.graph_infra.ego_gnn as local\nstore=local.EgoStore(ROOT)\n')
        self.assertEqual(result['status'],'FAIL')

    def test_comments_and_strings_do_not_create_a_graph_reader(self):
        source='# from tools.graph_infra.ego_gnn import EgoStore\nlabel="EgoStore"\n'
        self.assertEqual(self.inspect(source)['status'],'NOT_APPLICABLE')

    def test_unrelated_semantic_model_remains_inapplicable(self):
        self.assertEqual(self.inspect('class Encoder:\n    pass\n')['status'],'NOT_APPLICABLE')

    def test_broken_source_fails_closed(self):
        result=self.inspect('from tools.graph_infra.ego_gnn import EgoStore\ndef broken(:\n')
        self.assertEqual(result['status'],'FAIL')


if __name__=='__main__':unittest.main()
