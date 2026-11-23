"""Named registration remains fail-closed at the candidate consumer boundary."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.research.stage2 import graph_contract as gc


class NamedGraphTests(unittest.TestCase):
    def link_repository(self, root, isolated, contract):
        manifest = json.loads((root / 'assets/manifest.json').read_text())
        paths = set(contract['bindings']) | {str(gc.NAMED_CONTRACT), contract['audit']}
        for entry in manifest['assets']:
            if entry.get('use') == 'training' or 'graph_name' in entry:
                paths.add(entry['path'])
                paths.update(entry.get('inputs', {}))
                paths.update(entry.get('source_hashes', {}))
        for relative in paths:
            destination = isolated / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(root / relative, destination)
        return manifest

    def inspect(self, source, verdict=None):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[4] / 'results/research/issue252') as directory:
            tree = Path(directory)
            (tree / 'model.py').write_text(source)
            with patch.object(gc, 'verify_named_graph', return_value=verdict or {'status': 'PASS', 'identity': 'one', 'detail': 'verified named graph'}):
                return gc.inspect_tree(tree)

    def test_registered_ego_consumer(self):
        source = "BEHAVIOR_GRAPH_NAME='infra246-visible-v2'\nfrom tools.graph_infra.ego_gnn import EgoStore\nstore=EgoStore(repo)\n"
        self.assertEqual(self.inspect(source)['status'], 'PASS')
        self.assertEqual(self.inspect(source, {'status':'FAIL','detail':'source drift'})['status'], 'FAIL')

    def test_unknown_and_nonliteral_names(self):
        for value in ("'unknown'", 'get_name()'):
            self.assertEqual(self.inspect('BEHAVIOR_GRAPH_NAME='+value)['status'], 'FAIL')

    def test_mixed_and_evaluation_graphs(self):
        prefix = "BEHAVIOR_GRAPH_NAME='infra246-visible-v2'\nfrom tools.graph_infra.ego_gnn import EgoStore\n"
        for text in ("load('behavior_graph_directed/graph.npz')", "load('strict_struct_test_direct')", "EgoStore(repo, assets_root=other)", "load('alien_graph.npz')", "# train.parquet adjacency"):
            self.assertEqual(self.inspect(prefix+text)['status'], 'FAIL', text)

    def test_alias_cannot_redirect_registered_loader(self):
        prefix = "BEHAVIOR_GRAPH_NAME='infra246-visible-v2'\n"
        source = prefix + "from tools.graph_infra.ego_gnn import EgoStore as Local\nLocal(repo, other)\n"
        self.assertEqual(self.inspect(source)['status'], 'FAIL')

    def test_visible_filename_outside_registered_root(self):
        source = "BEHAVIOR_GRAPH_NAME='infra246-visible-v2'\nfrom tools.graph_infra.ego_gnn import EgoStore\nload('alien/visible_forward.npz')\n"
        self.assertEqual(self.inspect(source)['status'], 'FAIL')

    def test_candidate_cannot_import_graph_builder(self):
        source = "BEHAVIOR_GRAPH_NAME='infra246-visible-v2'\nfrom tools.graph_infra.ego_gnn import EgoStore\nfrom tools.graph_infra.graph_build import count_edges\n"
        self.assertEqual(self.inspect(source)['status'], 'FAIL')

    def test_contract_and_receipt_mutations_fail(self):
        original = gc._sha256
        for target in (gc.NAMED_CONTRACT, Path('assets/v7_graph_contract/audit.json')):
            def mutated(path):
                return 'changed' if Path(path) == gc.repo_root() / target else original(path)
            with patch.object(gc, '_sha256', side_effect=mutated):
                self.assertEqual(gc.verify_named_graph(gc.NAMED_GRAPH)['status'], 'FAIL')

    def test_mutated_source_and_manifest_roles_fail_but_additions_do_not(self):
        root = gc.repo_root()
        contract = json.loads((root / gc.NAMED_CONTRACT).read_text())
        with tempfile.TemporaryDirectory(dir=root / 'results/research/issue252') as directory:
            isolated = Path(directory)
            manifest = self.link_repository(root, isolated, contract)
            manifest['assets'].append({'name':'unrelated_report','path':'assets/unrelated.json','sha256':'new','use':'evaluation'})
            (isolated / 'assets/manifest.json').write_text(json.dumps(manifest))
            with patch.object(gc, 'repo_root', return_value=isolated):
                self.assertEqual(gc.verify_named_graph(gc.NAMED_GRAPH)['status'], 'PASS')
                entry = next(x for x in manifest['assets'] if x['name']=='visible_forward')
                entry['use'] = 'evaluation'
                (isolated / 'assets/manifest.json').write_text(json.dumps(manifest))
                self.assertEqual(gc.verify_named_graph(gc.NAMED_GRAPH)['status'], 'FAIL')
                entry['use'] = contract['manifest_leaves']['visible_forward']['use']
                (isolated / 'assets/manifest.json').write_text(json.dumps(manifest))
                source = isolated / 'tools/issue252/graph_audit.py'
                source.unlink()
                source.write_text('changed audit implementation')
                self.assertEqual(gc.verify_named_graph(gc.NAMED_GRAPH)['status'], 'FAIL')

    def test_new_cache_requires_graph_and_training_ancestry(self):
        root = gc.repo_root()
        metadata = json.loads((root / gc.NAMED_CONTRACT).read_text())
        with tempfile.TemporaryDirectory(dir=root / 'results/research/issue252') as directory:
            isolated = Path(directory)
            manifest = self.link_repository(root, isolated, metadata)
            cache = isolated / 'assets/cache.json'
            cache.write_text('{}')
            entry = {'name':'cache','path':'assets/cache.json','use':'training',
                     'sha256':gc._sha256(cache),'bytes':cache.stat().st_size}
            manifest['assets'].append(entry)
            manifest_path = isolated / 'assets/manifest.json'
            def verdict():
                manifest_path.write_text(json.dumps(manifest))
                with patch.object(gc, 'repo_root', return_value=isolated):
                    return gc.verify_named_graph(gc.NAMED_GRAPH)['status']
            self.assertEqual(verdict(), 'FAIL')
            entry.update(graph_name=gc.NAMED_GRAPH,
                         graph_sha256=metadata['manifest_leaves']['visible_forward']['sha256'],
                         inputs={'assets/train/visible_forward.npz':metadata['manifest_leaves']['visible_forward']['sha256']},
                         source_hashes={'tools/issue252/graph_audit.py':metadata['bindings']['tools/issue252/graph_audit.py']['sha256']})
            self.assertEqual(verdict(), 'PASS')
            with patch.object(gc, 'repo_root', return_value=isolated):
                original_identity = gc.verify_named_graph(gc.NAMED_GRAPH)['identity']
            cache.write_text('{"changed":true}')
            entry['sha256'], entry['bytes'] = gc._sha256(cache), cache.stat().st_size
            self.assertEqual(verdict(), 'PASS')
            with patch.object(gc, 'repo_root', return_value=isolated):
                self.assertNotEqual(original_identity, gc.verify_named_graph(gc.NAMED_GRAPH)['identity'])
            inputs = entry['inputs']
            evaluation = next(e for e in manifest['assets'] if e['name']=='strict_struct_test_direct')
            entry['inputs'] = {evaluation['path']: evaluation['sha256']}
            self.assertEqual(verdict(), 'FAIL')
            entry['inputs'] = inputs
            entry['graph_sha256'] = 'wrong'
            self.assertEqual(verdict(), 'FAIL')

    def test_registration_receipt_is_pinned(self):
        root = gc.repo_root()
        contract = root / gc.NAMED_CONTRACT
        self.assertEqual(gc._sha256(contract), gc.NAMED_CONTRACT_SHA256)
        data = json.loads(contract.read_text())
        self.assertIn('tools/issue252/graph_audit.py', data['bindings'])
        self.assertIn('results/stage0_build_parquet/train.parquet', data['bindings'])
        self.assertNotIn('assets/manifest.json', data['bindings'])

    def test_identity_exposes_contract_and_source(self):
        result = gc.verify_named_graph('infra246-visible-v2')
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['contract_sha256'], gc.NAMED_CONTRACT_SHA256)


if __name__ == '__main__':
    unittest.main()
