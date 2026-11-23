"""Source-bound MCP results survive reconnects without repeating GPU checks."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
COUNTER=itertools.count()


def server():
    name='issue251_cache_server_'+str(next(COUNTER))
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools/genrec-mcp/server.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def fixture(repo):
    tree=repo/'stage2_RQ-VAE'/'Candidate';(tree/'config').mkdir(parents=True)
    (tree/'train_rqvae.py').write_text('VALUE=1\n')
    (tree/'config/train_config.json').write_text('{"beta":0.25}\n')
    checks=repo/'tools/research/stage2';checks.mkdir(parents=True)
    for name in ('checks.py','ddp_probe.py','graph_contract.py'):
        (checks/name).write_text('POLICY=1\n')
    return tree


def verdict(tree,status='PASS'):
    from tools.research.stage2.checks import check_behavior_graph
    return {'tree':str(tree),'passed':status=='PASS','admissible':status=='PASS',
            'checks':[{'name':'entry','status':status,'detail':'executed verdict'},
                      check_behavior_graph(tree)]}


class MachineCacheTests(unittest.TestCase):
    def test_public_machine_calls_reuse_the_result_after_server_restart(self):
        first,second=server(),server()
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);tree=fixture(repo)
            with patch.object(first,'REPO',repo),patch.object(second,'REPO',repo),patch(
                    'tools.research.stage2.checks.run_all',return_value=verdict(tree,'FAIL')) as runner:
                a=first.machine_admission(tree);b=first.machine_admission(tree)
                c=second.machine_admission(tree)
                self.assertFalse(a['passed']);self.assertEqual(a,b);self.assertEqual(b,c)
                self.assertEqual(runner.call_count,1,'cached machine calls must not repeat the expensive entry')

    def test_conformance_result_publication_survives_reconnect(self):
        first,second=server(),server()
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);tree=fixture(repo)
            with patch.object(first,'REPO',repo),patch.object(second,'REPO',repo),patch(
                    'tools.research.stage2.checks.run_all',return_value=verdict(tree)) as runner:
                a=first._conformance_report(tree,'bound_source')
                b=second._conformance_report(tree,'bound_source')
                self.assertEqual(a,b)
                self.assertEqual(runner.call_count,1,'completed verdict must be atomically published, not silently discarded')

    def test_configuration_change_cannot_reuse_old_admission(self):
        module=server()
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);tree=fixture(repo)
            with patch.object(module,'REPO',repo),patch('tools.research.stage2.checks.run_all',
                    side_effect=[verdict(tree),verdict(tree,'FAIL')]) as runner:
                self.assertTrue(module.machine_admission(tree)['passed'])
                (tree/'config/train_config.json').write_text('{"beta":0.5}\n')
                self.assertFalse(module.machine_admission(tree)['passed'])
                self.assertEqual(runner.call_count,2)

    def test_graph_checker_change_invalidates_cache_policy_identity(self):
        module=server()
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);fixture(repo)
            with patch.object(module,'REPO',repo):
                before=module._checks_fingerprint()
                (repo/'tools/research/stage2/graph_contract.py').write_text('POLICY=2\n')
                self.assertNotEqual(before,module._checks_fingerprint(),
                                    'changed graph admission policy must invalidate cached verdicts')

    def test_cached_entry_cannot_hide_a_changed_graph_verdict(self):
        module=server()
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);tree=fixture(repo);report=verdict(tree)
            with patch.object(module,'REPO',repo),patch('tools.research.stage2.checks.run_all',
                    return_value=report) as runner:
                self.assertTrue(module.machine_admission(tree)['passed'])
                changed={'name':'behavior_graph','status':'FAIL','detail':'graph bytes changed'}
                with patch('tools.research.stage2.checks.check_behavior_graph',return_value=changed):
                    result=module.machine_admission(tree)
                self.assertFalse(result['passed'],'a cached entry must not bypass current graph integrity')
                self.assertEqual(result['checks'][-1],changed)
                self.assertEqual(runner.call_count,1)

    def test_shared_runtime_change_invalidates_entry_identity(self):
        module=server()
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);fixture(repo)
            runtime=repo/'tools/graph_infra';runtime.mkdir(parents=True)
            source=runtime/'ego_gnn.py';source.write_text('VERSION=1\n')
            with patch.object(module,'REPO',repo):
                before=module._checks_fingerprint()
                source.write_text('VERSION=2\n')
                self.assertNotEqual(before,module._checks_fingerprint(),
                                    'shared GNN runtime changes must invalidate the entry verdict')

    def test_not_run_is_not_cached_as_an_executed_verdict(self):
        module=server()
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);tree=fixture(repo)
            with patch.object(module,'REPO',repo),patch('tools.research.stage2.checks.run_all',
                    side_effect=[verdict(tree,'NOT_RUN'),verdict(tree)]) as runner:
                self.assertFalse(module.machine_admission(tree)['passed'])
                self.assertTrue(module.machine_admission(tree)['passed'])
                self.assertEqual(runner.call_count,2)


if __name__=='__main__':unittest.main()
