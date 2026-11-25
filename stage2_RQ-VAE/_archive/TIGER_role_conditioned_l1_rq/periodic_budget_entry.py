"""Actual60epoch four-rank refresh regression; no formal artifacts or test access."""
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

TREE = Path(__file__).resolve().parent
ROOT = next(p for p in TREE.parents if (p/'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(TREE))
sys.path.insert(1, str(TREE/'tests'))
import numpy as np
import torch
import train_rqvae as training
from benchmark_entry import configure, source_binding
from test_periodic_budget import assert_anchor_receipt
from tools.issue256.r03.layer_validation_entry import LayerObserver


def main():
    run_id, output = configure('periodic_budget')
    rank = int(os.environ['RANK'])
    binding = source_binding('periodic_budget')
    report = {'run_id':run_id, 'rank':rank, 'worker_pid':os.getpid(),
              'formal_training':False, 'source_hashes':binding, 'anchors':[], 'status':'running',
              'recipe':{'epochs':3000,'global_batch':4096,'world_size':4,'seed':42}}
    peek = training._calibration_batch
    calibrate = training.RQVAE.calibrate_curvature_weight
    current = {}

    def observed_peek(loader, device):
        state = torch.random.get_rng_state().clone()
        batch, ids = peek(loader, device)
        assert torch.equal(state, torch.random.get_rng_state()), 'calibration peek must preserve the real training RNG'
        positions = list(iter(loader.sampler))[:loader.batch_size]
        expected_ids = loader.dataset.tensors[1][positions]
        assert torch.equal(ids.cpu(), expected_ids.cpu()), 'anchor must use its current epoch first actual sampler population'
        current.update(epoch=loader.sampler.epoch+1, step=loader.sampler.epoch*len(loader))
        return batch, ids

    def observed_calibration(model, batch, item_ids, catalogue_embeddings, *args, **kwargs):
        epoch, step = current['epoch'], current['step']
        sample = kwargs['structural_sample']
        sampler = model.structural_sampler()
        prior_receipt = sampler.receipt()
        expected = sampler.draw(step,rank,4,track=False)
        assert sample.keys() == expected.keys(), 'anchor structural sample schema must match the actual first step'
        for key in expected:
            assert np.array_equal(np.asarray(sample[key]),np.asarray(expected[key])), 'anchor must use its current first structural step: '+key
        before = [p.detach().clone() for p in model.parameters()]
        calibration_started = time.perf_counter()
        value = calibrate(model,batch,item_ids,catalogue_embeddings,*args,**kwargs)
        calibration_seconds = time.perf_counter() - calibration_started
        assert all(torch.equal(p,v) for p,v in zip(model.parameters(),before)), 'refresh must not update any model parameter'
        assert sampler.receipt() == prior_receipt, 'refresh must not advance training structural coverage'
        probe = SimpleNamespace(report={})
        LayerObserver.verify_initial_budget(probe,model,{'batch':batch,'item_ids':item_ids,
            'catalogue':catalogue_embeddings,'structural_sample':sample})
        stale_mutation = None
        if report['anchors']:
            fresh = model.curvature_weight.detach().clone()
            prior_weight = report['anchors'][0]['lambda']
            model.curvature_weight.fill_(prior_weight)
            try:
                stale = SimpleNamespace(report={})
                try:
                    LayerObserver.verify_initial_budget(stale,model,{'batch':batch,
                        'item_ids':item_ids,'catalogue':catalogue_embeddings,'structural_sample':sample})
                except AssertionError as error:
                    assert str(error) in ('actual initial global module budget exceeded',
                        'actual initial global coefficient is not maximal'), 'stale mutation must fail its designated budget assertion'
                    stale_mutation = {'caught':True,'assertion':str(error),
                        'injected_weight':prior_weight,'restored_weight':float(fresh)}
                else:
                    raise AssertionError('stale initial coefficient escaped the actual current global budget assertion')
            finally:
                model.curvature_weight.copy_(fresh)
        report['anchors'].append({'epoch':epoch,'step':step,'lambda':value['lambda'],
            'independent_global_budget':probe.report['initial_global_module_budget'],
            'calibration_seconds':calibration_seconds,
            'current_sampler_population_exact':True,'current_structural_sample_exact':True,
            'stale_coefficient_mutation':stale_mutation,
            'parameters_unchanged':True,'coverage_unchanged':True})
        return value

    training._calibration_batch = observed_peek
    training.RQVAE.calibrate_curvature_weight = observed_calibration
    started = time.monotonic()
    try:
        training.main(diagnostic_epochs=60)
        records = [json.loads(line) for line in training.METRICS_PATH.read_text().splitlines()]
        calibrations = [r for r in records if r['event']=='curvature_calibration']
        report['actual_calibration_records'] = calibrations
        assert_anchor_receipt(calibrations)
        assert [row['epoch'] for row in report['anchors']] == [1,51], 'actual calibration calls must occur at both current native block anchors'
        assert [row['step'] for row in report['anchors']] == [0,300], 'actual current epoch structural samples must be used at both anchors'
        assert report['anchors'][0]['lambda'] != report['anchors'][1]['lambda'], 'refresh must replace the stale initial coefficient on the real changed model'
        assert report['anchors'][1]['stale_coefficient_mutation']['caught'], 'actual stale coefficient mutation must be caught before training resumes'
        assert not any(p.exists() for p in (training.CHECKPOINT,training.OUTPUT_SID,training.OUTPUT_JSON)), 'diagnostic must not write formal artifacts'
        assert binding == source_binding('periodic_budget'), 'source changed during diagnostic'
        report['status']='PASS'
    except Exception:
        report['status']='FAIL'
        report['error']=traceback.format_exc()
        raise
    finally:
        training._calibration_batch=peek
        training.RQVAE.calibrate_curvature_weight=calibrate
        report['wall_seconds']=time.monotonic()-started
        (output/('rank_%d.json'%rank)).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('periodic regression',rank,report['status'],run_id,flush=True)


if __name__=='__main__':
    main()
