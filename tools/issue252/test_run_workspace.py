"""The formal run must execute an immutable source generation, not the live tree."""
from pathlib import Path
import tempfile
import unittest

import importlib.util
import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())


class RunWorkspaceTests(unittest.TestCase):
    def test_source_bytes_are_frozen_with_independent_results(self):
        from tools.issue252.run_workspace import freeze_run
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'stage2_RQ-VAE'/'Arm'
            (tree/'model').mkdir(parents=True);(tree/'config').mkdir()
            (tree/'train_rqvae.py').write_text('VALUE=1\n')
            (tree/'model/__init__.py').write_text('VALUE=2\n')
            (tree/'config/train_config.json').write_text('{"frozen":true}\n')
            run=freeze_run(repo,tree,'run_one','Arm')
            source=Path(run['source']);result=Path(run['result']);entry=Path(run['entry'])
            self.assertEqual(source,repo/'results/stage2_RQ-VAE/_runs/run_one/Arm/src')
            self.assertEqual(result,repo/'results/stage2_RQ-VAE/_runs/run_one/Arm/result')
            self.assertEqual((source/'train_rqvae.py').read_text(),'VALUE=1\n')
            (tree/'train_rqvae.py').write_text('VALUE=99\n')
            self.assertEqual((source/'train_rqvae.py').read_text(),'VALUE=1\n')
            self.assertEqual(entry.parent,source)
            self.assertTrue(entry.is_file())
            with self.assertRaises(FileExistsError):freeze_run(repo,tree,'run_one','Arm')

    def test_run_identity_cannot_escape_the_repository(self):
        from tools.issue252.run_workspace import freeze_run
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'stage2_RQ-VAE'/'Arm';tree.mkdir(parents=True)
            for identity in ('../outside','/outside',''):
                with self.assertRaises(ValueError):freeze_run(repo,tree,identity,'Arm')

    def test_inventory_matches_archive_and_omits_only_its_exclusions(self):
        from tools.issue252.archive_audit import source_hashes
        from tools.issue252.run_workspace import freeze_run, load_run
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'stage2_RQ-VAE/Arm'
            (tree/'model').mkdir(parents=True);(tree/'logs').mkdir()
            (tree/'train_rqvae.py').write_text('EPOCHS=3000\n')
            (tree/'model/__init__.py').write_text('VALUE=2\n')
            (tree/'logs/live.json').write_text('{}')
            (tree/'weights.pth').write_bytes(b'not source')
            (tree/'dataset').mkdir();(tree/'dataset/config.json').write_text('{"keep":1}')
            admitted=source_hashes(tree)
            run=freeze_run(repo,tree,'generation','Arm')
            frozen=source_hashes(run['source'])
            self.assertEqual(set(frozen),set(admitted)|{'execution_entry.py'})
            self.assertEqual({k:frozen[k] for k in admitted},admitted)
            self.assertEqual(load_run(repo,'generation','Arm')['source'],run['source'])
            (Path(run['source'])/'model/__init__.py').write_text('VALUE=99\n')
            with self.assertRaisesRegex(ValueError,'source'):
                load_run(repo,'generation','Arm')

    def test_adapter_changes_paths_only_and_preserves_recipe(self):
        from tools.issue252.run_workspace import freeze_run
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'stage2_RQ-VAE/Arm'
            (tree/'model').mkdir(parents=True)
            (tree/'train_rqvae.py').write_text('EPOCHS=3000\n')
            (tree/'model/__init__.py').write_text('VALUE=1\n')
            marker=repo/'tools/research/stage2/contract.py'
            marker.parent.mkdir(parents=True);marker.write_text('')
            run=freeze_run(repo,tree,'run_adapter','Arm')
            called=[]
            trainer=SimpleNamespace(EPOCHS=3000,SEED=42,DATA='unchanged',
                                    main=lambda:called.append(True))
            with patch.dict(sys.modules,{'train_rqvae':trainer}), \
                    patch.object(sys,'path',list(sys.path)), \
                    patch('tools.research.stage2.launcher.is_child_rank',return_value=True):
                runpy.run_path(run['entry'],run_name='__main__')
            self.assertEqual(called,[True])
            self.assertEqual((trainer.EPOCHS,trainer.SEED,trainer.DATA),(3000,42,'unchanged'))
            self.assertEqual(trainer.RUN_ID,'run_adapter')
            self.assertEqual(trainer.RESULT_DIR,Path(run['result']))
            self.assertEqual(trainer.CHECKPOINT,Path(run['result'])/'rqvae_best.pth')
            self.assertEqual(trainer.METRICS_PATH,Path(run['result'])/'logs/training_metrics.jsonl')

    def test_adapter_parent_uses_only_the_standard_launcher(self):
        from tools.issue252.run_workspace import freeze_run
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'stage2_RQ-VAE/Arm'
            (tree/'model').mkdir(parents=True)
            (tree/'train_rqvae.py').write_text('EPOCHS=3000\n')
            (tree/'model/__init__.py').write_text('')
            marker=repo/'tools/research/stage2/contract.py'
            marker.parent.mkdir(parents=True);marker.write_text('')
            run=freeze_run(repo,tree,'parent_adapter','Arm')
            with patch.object(sys,'path',list(sys.path)), \
                    patch('tools.research.stage2.launcher.is_child_rank',return_value=False), \
                    patch('tools.research.stage2.launcher.launch',return_value=0) as launch:
                with self.assertRaises(SystemExit) as done:
                    runpy.run_path(run['entry'],run_name='__main__')
            self.assertEqual(done.exception.code,0)
            launch.assert_called_once_with(Path(run['entry']),Path(run['result']),
                                           python=sys.executable)

    def test_same_run_id_cannot_be_reused_for_another_arm(self):
        from tools.issue252.run_workspace import freeze_run
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory)
            for arm in ('Arm','Other'):
                tree=repo/'stage2_RQ-VAE'/arm;(tree/'model').mkdir(parents=True)
                (tree/'train_rqvae.py').write_text('VALUE=1\n')
                (tree/'model/__init__.py').write_text('')
            freeze_run(repo,repo/'stage2_RQ-VAE/Arm','same_run','Arm')
            with self.assertRaises(FileExistsError):
                freeze_run(repo,repo/'stage2_RQ-VAE/Other','same_run','Other')

    def test_arm_traversal_and_source_symlinks_are_refused(self):
        from tools.issue252.run_workspace import freeze_run
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'stage2_RQ-VAE/Arm';(tree/'model').mkdir(parents=True)
            (tree/'train_rqvae.py').write_text('VALUE=1\n')
            (tree/'model/__init__.py').write_text('')
            for arm in ('../escape','/escape','..',''):
                with self.assertRaises(ValueError):freeze_run(repo,tree,'safe',arm)
            (tree/'linked.json').symlink_to(tree/'train_rqvae.py')
            with self.assertRaisesRegex(ValueError,'symlink'):
                freeze_run(repo,tree,'safe','Arm')


def server():
    spec=importlib.util.spec_from_file_location('issue252_workspace_server',
                                               ROOT/'tools/genrec-mcp/server.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class PersistedRunTests(unittest.TestCase):
    def setUp(self):
        from tools.issue252.run_workspace import freeze_run
        self.temporary=tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252')
        self.addCleanup(self.temporary.cleanup)
        self.repo=Path(self.temporary.name);self.tree=self.repo/'stage2_RQ-VAE/Arm'
        (self.tree/'model').mkdir(parents=True)
        (self.tree/'train_rqvae.py').write_text('VALUE=1\n')
        (self.tree/'model/__init__.py').write_text('')
        self.run=freeze_run(self.repo,self.tree,'current_run','Arm')
        self.module=server()
        self.state={'run_id':self.run['run_id'],'arm':'Arm',
                    'source_tree':self.run['source'],'entrypoint':self.run['entry'],
                    'result_dir':self.run['result'],'logs':self.run['logs']}

    def test_status_uses_exact_persisted_result_not_static_tree(self):
        result=Path(self.run['result'])
        (result/'item_sids.json').write_text('{"0":[1,2,3,0]}')
        with patch.object(self.module,'REPO',self.repo), \
                patch.object(self.module,'resolve_stage2_tree',return_value=(self.tree,None)), \
                patch.object(self.module,'load_state',return_value=self.state), \
                patch.object(self.module,'tree_entrypoint',return_value=(self.tree/'train_rqvae.py',[])), \
                patch.object(self.module,'stage_processes',return_value=[]), \
                patch.object(self.module,'state_process_live',return_value=False), \
                patch.object(self.module,'probe_gpu_layout',return_value={}), \
                patch.object(self.module,'_gpu_processes',return_value=[]), \
                patch.object(self.module,'card_conflicts',return_value={}):
            status=self.module.tool_stage2_status(str(self.tree))
        self.assertEqual(status['result_dir'],str(result))
        self.assertEqual(status['source_tree'],self.run['source'])
        self.assertEqual(status['run_id'],'current_run')
        self.assertIn('item_sids.json',status['artifacts'])
        self.assertEqual(status['metrics'],self.run['logs']['metrics'])

    def test_nested_sid_keeps_arm_identity_and_rejects_crossed_generation(self):
        from tools.issue252.run_workspace import freeze_run
        sid=Path(self.run['result'])/'item_sids.json'
        root=self.repo/'results/stage2_RQ-VAE'
        with patch.object(self.module,'REPO',self.repo), \
                patch.object(self.module,'RESULT2_ROOT',root), \
                patch.object(self.module,'load_state',return_value=self.state):
            self.assertEqual(self.module.derive_stage3_variant(sid),'Arm')
            self.assertEqual(self.module.resolve_stage2_sid(root/'Arm/item_sids.json'),sid)
            old=freeze_run(self.repo,self.tree,'old_run','Arm')
            with self.assertRaisesRegex(ValueError,'current'):
                self.module.resolve_stage2_sid(Path(old['result'])/'item_sids.json')

    def test_runtime_helper_change_invalidates_admission_fingerprint(self):
        helper=self.repo/'tools/issue252/run_workspace.py'
        helper.parent.mkdir(parents=True);helper.write_text('POLICY=1\n')
        with patch.object(self.module,'REPO',self.repo):
            first=self.module._checks_fingerprint()
            helper.write_text('POLICY=2\n')
            self.assertNotEqual(first,self.module._checks_fingerprint())

    def test_stage3_evidence_is_selected_by_exact_sid_not_newest_arm_file(self):
        root=self.repo/'results/stage3_T5Train';sid=Path(self.run['result'])/'item_sids.json'
        correct=root/'Arm/logs/old_timestamp';wrong=root/'Arm/logs/new_timestamp'
        for directory,path,run_id in ((correct,sid,'current_run'),
                                       (wrong,sid.parent.parent.parent/'other_run/Arm/result/item_sids.json','other_run')):
            directory.mkdir(parents=True)
            (directory/'training_metrics.jsonl').write_text(json.dumps({
                'event':'train_start','code_path':str(path)})+'\n')
            (directory/'test_final.json').write_text(json.dumps({
                'sid_path':str(path),'stage2_run_id':run_id}))
        with patch.object(self.module,'RESULT3_ROOT',root):
            metrics,final=self.module.stage3_bound_files('Arm',sid,'current_run')
        self.assertEqual(metrics,correct/'training_metrics.jsonl')
        self.assertEqual(final,correct/'test_final.json')

    def test_stage3_export_cannot_cross_source_or_checkpoint_generations(self):
        from tools.issue252.archive_audit import sha256
        sid=Path(self.run['result'])/'item_sids.json';sid.write_text('{"0":[1,2,3,0]}')
        checkpoint=sid.parent/'rqvae_best.pth';checkpoint.write_bytes(b'checkpoint')
        block={'schema':'genrec.stage2.export_manifest/1','tokens_match':True,
               'tree':self.run['source'],'checkpoint':str(checkpoint),
               'checkpoint_sha256':sha256(checkpoint),
               'sid':str(sid),'sid_sha256':sha256(sid)}
        manifest=sid.parent/'export_manifest.json'
        with patch.object(self.module,'REPO',self.repo), \
                patch.object(self.module,'stage3_archive_binding',return_value={'allow':True}) as binding:
            manifest.write_text(json.dumps(block))
            self.assertTrue(self.module.stage3_export_gate(sid)['allow'])
            binding.return_value={'allow':False,'reason':'missing exact archive binding'}
            self.assertFalse(self.module.stage3_export_gate(sid)['allow'])
            binding.return_value={'allow':True}
            for field in ('tree','checkpoint'):
                crossed=dict(block);crossed[field]=str(self.repo/'other-generation')
                manifest.write_text(json.dumps(crossed))
                verdict=self.module.stage3_export_gate(sid)
                self.assertFalse(verdict['allow'])
                self.assertIn('generation',verdict['reason'])

    def test_public_start_preserves_four_cards_and_never_archives_old_results(self):
        module=self.module
        layout={'mode':'ddp','visible_count':4,'visible':['0','1','2','3']}
        old=self.repo/'results/stage2_RQ-VAE/Arm/item_sids.json'
        old.parent.mkdir(parents=True);old.write_text('old generation')
        harness=self.repo/'results/research/_mcp_runtime';harness.mkdir(parents=True)
        with patch.object(module,'REPO',self.repo), \
                patch.object(module,'HARNESS_DIR',harness), \
                patch.object(module,'resolve_stage2_tree',return_value=(self.tree,None)), \
                patch.object(module,'tree_entrypoint',return_value=(self.tree/'train_rqvae.py',[])), \
                patch.object(module,'tree_result_dir',return_value=old.parent), \
                patch.object(module,'tree_python',return_value=sys.executable), \
                patch.object(module,'tree_arm',return_value='Arm'), \
                patch.object(module,'probe_gpu_layout',return_value=layout), \
                patch.object(module,'allocate_stage2_card') as single_card, \
                patch.object(module,'load_state',return_value={}), \
                patch.object(module,'save_state'), \
                patch.object(module,'stage_processes',return_value=[]), \
                patch.object(module,'state_process_live',return_value=False), \
                patch.object(module,'write_stage2_worker',return_value=harness/'stage2_worker.py'), \
                patch.object(module.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as spawn:
            result=module.tool_stage2_start(str(self.tree))
        single_card.assert_not_called()
        self.assertEqual(result['gpu_probe']['visible_count'],4)
        self.assertEqual(old.read_text(),'old generation')
        spec=json.loads(spawn.call_args.args[0][-1])
        self.assertEqual(spec['gpu_probe']['mode'],'ddp')
        self.assertIn('run_id',spec)
        self.assertIsNotNone(module.stage2_launch_decision(
            {'mode':'single-gpu','visible_count':1,'entrypoint_supports_single_gpu':True}))

    def test_worker_admits_then_runs_frozen_adapter_without_deleted_bootstrap(self):
        from tools.research.verification.admission import source_digest
        module=self.module;saved=[]
        layout={'mode':'ddp','visible_count':4,'visible':['0','1','2','3']}
        with patch.object(module,'REPO',self.repo), \
                patch.object(module,'load_state',return_value={}), \
                patch.object(module,'save_state',side_effect=lambda key,value:saved.append(dict(value))), \
                patch.object(module,'machine_admission',return_value={
                    'passed':True,'source_digest':source_digest(self.tree),
                    'graph_identity':None,'checks_fingerprint':module._checks_fingerprint(),
                    'checks':[]}) as admission, \
                patch.object(module,'_checks_fingerprint',return_value=module._checks_fingerprint()), \
                patch('tools.research.stage2.checks.check_behavior_graph',
                      return_value={'status':'NOT_APPLICABLE','identity':None}), \
                patch.object(module,'_wait_for_the_ranks',return_value=True), \
                patch.object(module.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as spawn:
            rc=module.stage2_worker_main(str(self.tree),{
                'state_key':'stage2__Arm','run_id':'worker_run','arm':'Arm',
                'python':sys.executable,'gpu_probe':layout})
        self.assertEqual(rc,0)
        admission.assert_called_once_with(self.tree)
        cmd=spawn.call_args.args[0]
        self.assertEqual(Path(cmd[1]).name,'execution_entry.py')
        self.assertNotIn('tools.research.stage2.bootstrap',' '.join(cmd))
        self.assertEqual(spawn.call_args.kwargs['cwd'],saved[-1]['result_dir'])
        self.assertEqual(saved[-1]['run_id'],'worker_run')
        self.assertEqual(saved[-1]['graph_identity'],None)
        self.assertEqual(saved[-1]['source_tree'],
                         str(self.repo/'results/stage2_RQ-VAE/_runs/worker_run/Arm/src'))

    def test_source_drift_after_admission_refuses_without_formal_spawn(self):
        from tools.research.verification.admission import source_digest
        digest=source_digest(self.tree)
        def drift(tree):
            (tree/'train_rqvae.py').write_text('VALUE=99\n')
            return {'passed':True,'source_digest':digest,'graph_identity':None,
                    'checks_fingerprint':'fixed','checks':[]}
        saved=[]
        with patch.object(self.module,'REPO',self.repo), \
                patch.object(self.module,'load_state',return_value={}), \
                patch.object(self.module,'save_state',side_effect=lambda key,value:saved.append(dict(value))), \
                patch.object(self.module,'machine_admission',side_effect=drift), \
                patch.object(self.module.subprocess,'Popen') as spawn:
            rc=self.module.stage2_worker_main(str(self.tree),{
                'state_key':'stage2__Arm','run_id':'drift_run','arm':'Arm',
                'python':sys.executable,'gpu_probe':{'mode':'ddp','visible_count':4}})
        self.assertEqual(rc,1)
        spawn.assert_not_called()
        self.assertIn('source digest changed',saved[-1]['refused'])

if __name__=='__main__':unittest.main()
