"""Fail-closed formal credentials; fixtures describe evidence, never GPU work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.issue252.test_run_workspace import ROOT, server

ARM = 'TIGER_gnn_attention_sectional_rq'


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        from tools.issue252 import training_admission as admission
        self.a = admission
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.tree = self.repo/'stage2_RQ-VAE'/ARM
        for name, text in {
            'train_rqvae.py': 'EPOCHS=3000\n', 'model/__init__.py': '',
            'config/train_config.json': '{}', 'tests/test_reference.py': '',
        }.items():
            p=self.tree/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        for name in ('tools/research/stage2/launcher.py','tools/issue252/performance.py',
                     'tools/issue252/launch_validation.py','tools/issue252/launch_benchmark.py',
                     'results/research/issue252/CONTRACT.md',
                     'results/research/issue252/EXECUTION_CONTRACT.md'):
            p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
        p=self.repo/'assets/manifest.json';p.parent.mkdir();p.write_text('{"assets":[]}')
        self.recipe={'epochs':3000,'per_rank_batch':1024,'global_batch':4096,
                     'world_size':4,'seed':42,
                     'config_sha256':self.a.sha256(self.tree/'config/train_config.json')}
        groups=['encoder','decoder','GNN.input','GNN.layer','sharedKey','sharedValue','sharedScore']
        groups += ['Query%d'%i for i in range(5)]+['codebook%d'%i for i in range(3)]
        stats={name:{'norm':1.,'finite':True,'absent':False,'tensors':1,'none':0,'zero':0}
               for name in groups}
        self.validation=[];self.benchmark=[]
        for kind in ('validation','benchmark'):
            directory=self.repo/'results/research/issue252'/kind/'actual_run';directory.mkdir(parents=True)
            for rank in range(4):
                row={'status':'pass','rank':rank,'run_id':kind+'_run','worker_pid':100+rank,
                     'controller_pid':99,'formal_training':False,'recipe':self.recipe,
                     'source_hashes':self.a.source_binding(self.repo,self.tree,kind),
                     'diagnostic_epochs':1 if kind=='validation' else 60}
                if kind=='validation':
                    structural=copy.deepcopy(stats);structural['decoder']['norm']=0.
                    row.update(actual_batch_items=1024,completed_epoch=1,
                        completed_local_losses=[1.,2.],first_step_losses={
                            'original_loss':1.,'structural_loss':1.,'combined_loss':2.},
                        actual_structural_hard_q_decode_exact=True,
                        first_structural_sample={'centre':[1]},
                        normal_backward=stats,actual_optimizer_update=stats,
                        normal_forward={'encoder':{'calls':1,'finite':True}},
                        raw_real_batch_VJP={'original':stats,'structural':structural},
                        completed_parameter_hashes=['d'*64]*4,completed_state_finite=True,
                        switches={label:{'hard_q_decode_exact':True,'cost_delta_norm':1.,
                            'weights_sha256':'a'*64,'cost_sha256':'b'*64,'tokens_sha256':'c'*64}
                            for label in ('both_on','behavior_off','curvature_off','both_off')})
                    row['switches']['query_intervention']={'weight_delta_norm':1.,'cost_delta_norm':1.}
                    if rank==0:
                        cp=directory/'diagnostic_epoch1.pth';cp.write_bytes(b'checkpoint')
                        sid=directory/'diagnostic_item_sids.json';sid.write_text('{"0":[1,2,3,0]}')
                        row['export']={'items':1,'checkpoint_sha256':self.a.sha256(cp),
                            'sid_sha256':self.a.sha256(sid),'same_checkpoint_exact':True,'formal_training':False}
                elif rank==0:
                    row['epoch_boundaries']=[{'epoch':i,'start':float(i),'end':i+.25,
                        'structural_coverage':[{'rank':r,'epoch_updates':1,
                            'epoch_target_occurrences':1,'epoch_reference_draws':1} for r in range(4)]}
                        for i in range(1,61)]
                    # Continuous boundaries include no hidden interepoch time.
                    for i,boundary in enumerate(row['epoch_boundaries']):
                        boundary['start']=i*.25;boundary['end']=(i+1)*.25
                    row['periodic_evaluation_epochs']=[50]
                    row['performance']={'passed':True,'mean_seconds':.25}
                p=directory/('rank_%d.json'%rank);p.write_text(json.dumps(row))
                getattr(self,kind).append(p)
        self.tests={}
        for kind in ('deterministic','reference','mutation'):
            p=self.repo/('results/research/issue252/'+kind+'.json')
            output=p.with_suffix('.out');output.write_text('Ran 3 tests\nOK\n')
            p.write_text(json.dumps({'schema':'genrec.issue252.tests/1','kind':kind,'status':'pass',
                'command':['python','-m','unittest'],'exit_code':0,
                'output':{'path':str(output),'sha256':self.a.sha256(output)},
                'tests_run':3,'failures':0,'errors':0,'skipped':0,
                'source_hashes':self.a.source_binding(self.repo,self.tree,'validation'),
                'mutations':[{'category':c,'caught':True} for c in ('numeric','structural','routing')]}))
            self.tests[kind]=p

    def create(self):
        return self.a.create_receipt(self.repo,self.tree,validation_reports=self.validation,
            benchmark_reports=self.benchmark,test_reports=self.tests)

    def mutate(self,path,change):
        row=json.loads(path.read_text());change(row);path.write_text(json.dumps(row))

    def test_round_trip_and_hash_bound_reports(self):
        descriptor=self.create()
        self.assertEqual(Path(descriptor['path']),self.repo/self.a.RECEIPT)
        self.assertEqual(self.a.check_receipt(self.repo,self.tree),descriptor)
        self.validation[0].write_text('{}')
        with self.assertRaises(ValueError):self.a.check_receipt(self.repo,self.tree)

    def test_missing_receipt_refuses_before_workspace_reservation(self):
        from tools.issue252.run_workspace import freeze_run
        with self.assertRaises((ValueError,FileNotFoundError)):
            freeze_run(self.repo,self.tree,'formal_run',ARM)
        self.assertFalse((self.repo/'results/stage2_RQ-VAE/_runs/formal_run').exists())

    def test_failing_or_missing_rank_and_recipe_refused(self):
        original=list(self.validation);self.validation=self.validation[:3]
        with self.assertRaises(ValueError):self.create()
        self.validation=original
        for field,value in (('status','fail'),('rank',2),('run_id','another_run'),
                            ('recipe',dict(self.recipe,epochs=60)),('actual_batch_items',32)):
            with self.subTest(field=field):
                old=self.validation[0].read_text()
                self.mutate(self.validation[0],lambda row:row.update({field:value}))
                with self.assertRaises(ValueError):self.create()
                self.validation[0].write_text(old)

    def test_missing_math_gradient_ddp_switch_and_export_evidence_refused(self):
        for field in ('normal_backward','actual_optimizer_update','normal_forward','raw_real_batch_VJP',
                      'completed_parameter_hashes','switches','export','first_step_losses'):
            with self.subTest(field=field):
                old=self.validation[0].read_text()
                self.mutate(self.validation[0],lambda row:row.pop(field))
                with self.assertRaises(ValueError):self.create()
                self.validation[0].write_text(old)
        for kind in self.tests:
            old=self.tests[kind].read_text()
            self.mutate(self.tests[kind],lambda row:row.update(status='fail'))
            with self.assertRaises(ValueError):self.create()
            self.tests[kind].write_text(old)

    def test_forged_performance_pass_missing_rank_and_coverage_refused(self):
        for change in (
            lambda row:row['epoch_boundaries'][40].update(end=1000.),
            lambda row:row['epoch_boundaries'].pop(),
            lambda row:row.update(periodic_evaluation_epochs=[]),
            lambda row:row['epoch_boundaries'][49].update(structural_coverage=[{}]),
            lambda row:row['epoch_boundaries'][49]['structural_coverage'][0].update(epoch_updates=0),
        ):
            old=self.benchmark[0].read_text();self.mutate(self.benchmark[0],change)
            with self.assertRaises(ValueError):self.create()
            self.benchmark[0].write_text(old)
        self.benchmark.pop()
        with self.assertRaises(ValueError):self.create()

    def test_current_source_inventory_policy_contract_and_asset_drift_refused(self):
        self.create()
        for path in (self.tree/'model/new_production.py',self.tree/'model/__init__.py',
                     self.repo/'tools/research/stage2/new_production.py',
                     self.repo/'results/research/issue252/CONTRACT.md',self.repo/'assets/manifest.json'):
            with self.subTest(path=path):
                previous=path.read_bytes() if path.exists() else None
                path.write_text('changed')
                with self.assertRaises(ValueError):self.a.check_receipt(self.repo,self.tree)
                if previous is None:path.unlink()
                else:path.write_bytes(previous)

    def test_final_credential_is_stored_and_rechecked_before_spawn(self):
        from tools.issue252.run_workspace import check_run_admission, freeze_run
        credential=self.create()
        run=freeze_run(self.repo,self.tree,'credential_run',ARM)
        self.assertEqual(run['training_admission'],credential)
        self.assertEqual(check_run_admission(self.repo,run),credential)
        receipt=self.repo/self.a.RECEIPT
        self.mutate(receipt,lambda row:row.update(extra='changed after freeze'))
        with self.assertRaises(ValueError):check_run_admission(self.repo,run)

    def test_source_bound_test_report_and_captured_output_drift_refused(self):
        for kind in self.tests:
            original=self.tests[kind].read_text()
            self.mutate(self.tests[kind],lambda row:row.update(source_hashes={}))
            with self.assertRaises(ValueError):self.create()
            self.tests[kind].write_text(original)
        self.create()
        output=Path(json.loads(self.tests['reference'].read_text())['output']['path'])
        output.write_text('changed command output')
        with self.assertRaises(ValueError):self.a.check_receipt(self.repo,self.tree)

    def test_freeze_enforces_receipt_even_if_machine_result_is_forged(self):
        from tools.research.verification.admission import source_digest
        module=server()
        with patch.object(module,'REPO',self.repo), \
             patch.object(module,'load_state',return_value={}), \
             patch.object(module,'save_state'), \
             patch.object(module,'machine_admission',return_value={
                 'passed':True,'checks':[],'source_digest':source_digest(self.tree),
                 'graph_identity':None,'checks_fingerprint':'forged'}), \
             patch.object(module.subprocess,'Popen') as spawn:
            rc=module.stage2_worker_main(str(self.tree),{
                'state_key':'stage2__'+ARM,'run_id':'forged_run','arm':ARM,
                'python':'unused','gpu_probe':{'mode':'ddp','visible_count':4}})
        self.assertEqual(rc,1);spawn.assert_not_called()
        self.assertFalse((self.repo/'results/stage2_RQ-VAE/_runs/forged_run').exists())

    def test_machine_gate_and_worker_cannot_bypass_missing_credential(self):
        module=server();saved=[]
        with patch.object(module,'REPO',self.repo), \
             patch.object(module,'_conformance_report',return_value={'passed':True,'checks':[]}), \
             patch.object(module,'load_state',return_value={}), \
             patch.object(module,'save_state',side_effect=lambda key,row:saved.append(dict(row))), \
             patch('tools.research.stage2.anticollapse.patch_tree.apply_floor'), \
             patch.object(module.subprocess,'Popen') as spawn:
            self.assertFalse(module.machine_admission(self.tree)['passed'])
            rc=module.stage2_worker_main(str(self.tree),{'state_key':'stage2__'+ARM,
                'run_id':'formal_run','arm':ARM,'python':'unused',
                'gpu_probe':{'mode':'ddp','visible_count':4}})
        self.assertEqual(rc,1);spawn.assert_not_called()
        self.assertFalse((self.repo/'results/stage2_RQ-VAE/_runs/formal_run').exists())


class FrozenFloorTests(unittest.TestCase):
    def test_frozen_stale_floor_refuses_without_mutating_or_spawning(self):
        from tools.research.stage2 import launcher
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            tree=Path(directory);(tree/'model').mkdir()
            quantizer=tree/'model/layers.py'
            quantizer.write_text('constrained_assignment(\nverify_assignment(\n')
            installed=tree/'assignment_floor.py';installed.write_text('stale')
            before={p:p.read_bytes() for p in (quantizer,installed)}
            with patch.object(launcher.patch_tree,'floor_constant',return_value={}), \
                 patch.object(launcher.patch_tree,'unguarded_sites',return_value=[]), \
                 patch.object(launcher.patch_tree,'_migrate',side_effect=lambda text:text), \
                 patch.object(launcher.patch_tree,'apply_floor') as apply:
                with self.assertRaises(launcher.LaunchRefused):
                    launcher.prepare_floor(tree,frozen=True)
            apply.assert_not_called()
            self.assertEqual({p:p.read_bytes() for p in before},before)

    def test_current_frozen_floor_is_verify_only(self):
        from tools.research.stage2 import launcher
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            tree=Path(directory);(tree/'model').mkdir()
            (tree/'model/layers.py').write_text('constrained_assignment(\nverify_assignment(\n')
            (tree/'assignment_floor.py').write_bytes(launcher.patch_tree.FLOOR_MODULE.read_bytes())
            with patch.object(launcher.patch_tree,'floor_constant',return_value={}), \
                 patch.object(launcher.patch_tree,'unguarded_sites',return_value=[]), \
                 patch.object(launcher.patch_tree,'_migrate',side_effect=lambda text:text), \
                 patch.object(launcher.patch_tree,'apply_floor') as apply:
                self.assertFalse(launcher.prepare_floor(tree,frozen=True)['changed'])
            apply.assert_not_called()


    def test_frozen_launcher_refuses_patch_before_spawn_and_preserves_workspace(self):
        from tools.issue252.run_workspace import freeze_run
        from tools.research.stage2 import launcher
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            repo=Path(directory);tree=repo/'stage2_RQ-VAE/Arm'
            (tree/'model').mkdir(parents=True);(repo/'stage1_GeneEmbedding').mkdir()
            (tree/'train_rqvae.py').write_text('EPOCHS=3000\n')
            (tree/'model/layers.py').write_text('constrained_assignment(\nverify_assignment(\n')
            (tree/'assignment_floor.py').write_text('stale')
            run=freeze_run(repo,tree,'stale_floor_run','Arm')
            workspace=Path(run['workspace']).parent
            before={str(p.relative_to(workspace)):p.read_bytes()
                    for p in workspace.rglob('*') if p.is_file()}
            with patch.object(launcher,'_visible_cards',return_value=4), \
                 patch.object(launcher.patch_tree,'_migrate',side_effect=lambda text:text), \
                 patch.object(launcher.subprocess,'run') as spawn:
                with self.assertRaises(launcher.LaunchRefused):
                    launcher.launch(run['entry'],run['result'])
            spawn.assert_not_called()
            after={str(p.relative_to(workspace)):p.read_bytes()
                   for p in workspace.rglob('*') if p.is_file()}
            self.assertEqual(before,after)


class ArchiveGateTests(unittest.TestCase):
    def test_nested_binding_created_with_stage2_interpreter_and_verified(self):
        module=server()
        run={'source':'/source','result':'/result','run_id':'formal_run'}
        receipt={'run_id':'formal_run','source_tree':'/source',
                 'export_manifest':{'path':'/result/export_manifest.json'}}
        with patch.object(module.subprocess,'run',return_value=type('Done',(),{'returncode':0,'stderr':'','stdout':''})()) as child, \
             patch('tools.issue252.archive_audit.sha256',return_value='a'*64), \
             patch('tools.issue252.archive_audit.validate_binding',return_value=(receipt,{})) as validate:
            result=module.stage3_archive_binding(Path('/result/item_sids.json'),run,Path('/result/export_manifest.json'))
        self.assertTrue(result['allow'])
        self.assertEqual(child.call_args.args[0][0],module.PY2)
        self.assertIn('create_binding',child.call_args.args[0][2])
        validate.assert_called_once_with(Path('/result/archive_binding.json'),Path('/result/item_sids.json'),
                                        Path('/result/rqvae_best.pth'),Path('/source'))

    def test_missing_wrong_source_or_checkpoint_binding_refuses(self):
        module=server();run={'source':'/source','result':'/result','run_id':'formal_run'}
        for error in ('missing binding','explicit source tree differs','checkpoint binding hash mismatch'):
            with patch.object(module.subprocess,'run',return_value=type('Done',(),{'returncode':0,'stderr':'','stdout':''})()), \
                 patch('tools.issue252.archive_audit.validate_binding',side_effect=ValueError(error)):
                result=module.stage3_archive_binding(Path('/result/item_sids.json'),run,Path('/result/export_manifest.json'))
            self.assertFalse(result['allow']);self.assertIn(error,result['reason'])


if __name__=='__main__':unittest.main()
