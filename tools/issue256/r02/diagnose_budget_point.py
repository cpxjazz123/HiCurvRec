"""One real epoch distinguishing calibration, native first-step and replica states."""
from pathlib import Path
import json
import os
import sys
import time
import traceback

from tools.issue256.r02.identity import ROOT,SOURCE,EVIDENCE,PYTHON
from tools.research.stage2 import launcher


def child():
    import numpy as np
    import torch
    import torch.distributed as dist
    sys.path.insert(0,str(SOURCE))
    import train_rqvae as training
    import model.mechanism as mechanism
    import validation_entry as checks
    from benchmark_entry import configure,source_binding
    run_id,output=configure('budget_point')
    rank=int(os.environ['RANK'])
    report={'run_id':run_id,'rank':rank,'worker_pid':os.getpid(),
        'source_hashes':source_binding('budget_point'),'formal_training':False,
        'purpose':'Discrimination only, never admission or a coefficient search.','status':'RUNNING'}
    traces={'calibration':[],'train_first':[],'replica':[]}
    gradients=[]
    phase='other'
    original_quantize=training.RQVAE.quantize
    original_calibrate=training.RQVAE.calibrate_curvature_weight
    original_mean=mechanism.mean_gradients
    actual_model=None

    def scales(model):
        return {'tangent_host':model._tangent_scale_value,'tangent_buffer':float(model.tangent_scale),
            'target_host':model._target_scale_value,'target_buffer':float(model.target_scale),
            'levels_host':list(model._level_scales),'levels_buffer':model.level_scales.tolist(),
            'training':model.training,'matmul_precision':torch.get_float32_matmul_precision()}

    def quantize(model,encoded,weights,*,balanced):
        result=original_quantize(model,encoded,weights,balanced=balanced)
        if phase in traces:
            traces[phase].append({'encoded':encoded.detach().clone(),'weights':weights.detach().clone(),
                'tokens':result[3].detach().clone(),'costs':[v.detach().clone() for v in model.last_costs],
                'balanced':balanced})
        return result

    def reduce(gradients_in):
        original_mean(gradients_in)
        if phase=='calibration':
            gradients.append(tuple(g.detach().clone() for g in gradients_in))

    def calibrate(model,*args,**kwargs):
        nonlocal phase,actual_model
        phase='calibration'
        actual_model=model
        report['calibration_parameters']=checks.digest(model.named_parameters())
        report['calibration_scales']=scales(model)
        result=original_calibrate(model,*args,**kwargs)
        report['calibration']=result
        phase='train_first'
        return result

    class Observer:
        def __call__(self,event,**fields):
            nonlocal phase
            model=fields['model']
            if event=='first_step_before_backward':
                phase='replica'
                report['first_parameters']=checks.digest(model.named_parameters())
                report['first_scales']=scales(model)
                report['same_parameters_before_first_optimizer']=report['first_parameters']==report['calibration_parameters']
                report['first_losses']={k:float(fields[k]) for k in ('original_loss','structural_loss','combined_loss')}
                with torch.random.fork_rng(devices=[fields['catalogue'].device.index]):
                    raw=training.RQVAE(training._tokenizer_config(),in_dim=768,distance=training.DISTANCE).to(fields['catalogue'].device)
                raw.load_state_dict(model.state_dict())
                raw.train()
                raw.rank_local_checks=True
                report['replica_parameters']=checks.digest(raw.named_parameters())
                report['replica_scales']=scales(raw)
                result=raw(fields['batch'],fields['item_ids'],catalogue_embeddings=fields['catalogue'],structural_sample=fields['structural_sample'])
                loss=training.CRITERION(result[0],fields['batch'],result[1])[0]
                parameters=tuple(p for name,p in raw.named_parameters() if not name.startswith('decoder.'))
                og=torch.autograd.grad(loss,parameters,retain_graph=True)
                cg=torch.autograd.grad(result[4],parameters)
                for g in (*og,*cg):
                    dist.all_reduce(g,op=dist.ReduceOp.SUM)
                    g.div_(4)
                a=checks.statistics(raw,dict(zip(map(id,parameters),og)))
                b=checks.statistics(raw,dict(zip(map(id,parameters),cg)))
                weight=float(model.curvature_weight)
                report['replica_global_budgets']={name:{'original_norm':row['norm'],
                    'curvature_norm':b[name]['norm'],
                    'weighted_ratio':weight*b[name]['norm']/row['norm'] if row['norm'] else None}
                    for name,row in a.items() if name!='decoder'}
                phase='other'
            elif event=='first_step_after_backward':
                parameters=tuple(p for name,p in model.named_parameters() if not name.startswith('decoder.'))
                if len(gradients)!=3:
                    raise ValueError('expected actual original/reconstruction/curvature calibration VJPs')
                combined={id(p):o+float(model.curvature_weight)*c for p,o,c in zip(parameters,gradients[0],gradients[2])}
                differences={id(p):p.grad-combined[id(p)] for p in parameters}
                report['calibration_expected_combined']=checks.statistics(model,combined)
                report['native_first_combined']=checks.statistics(model,{id(p):p.grad for p in parameters})
                report['native_minus_calibration_combined']=checks.statistics(model,differences)
            elif event=='diagnostic_complete':
                report['completed_epoch']=fields['completed_epoch']

    training.RQVAE.quantize=quantize
    training.RQVAE.calibrate_curvature_weight=calibrate
    mechanism.mean_gradients=reduce
    training.DIAGNOSTIC_OBSERVER=Observer()
    started=time.monotonic()
    try:
        training.main(diagnostic_epochs=1)
        report['status']='COLLECTED_NOT_ADMISSION'
    except Exception:
        report['status']='FAIL'
        report['error']=traceback.format_exc()
        raise
    finally:
        arrays={}
        for label,records in traces.items():
            for branch,row in enumerate(records):
                for key in ('encoded','weights','tokens'):
                    arrays['%s_%d_%s'%(label,branch,key)]=row[key].cpu().numpy()
                for level,cost in enumerate(row['costs']):
                    arrays['%s_%d_cost%d'%(label,branch,level)]=cost.cpu().numpy()
        comparisons={}
        for label in ('train_first','replica'):
            rows=[]
            for left,right in zip(traces['calibration'],traces[label]):
                rows.append({'balanced_equal':left['balanced']==right['balanced'],
                    'encoded_max_abs':float((left['encoded']-right['encoded']).abs().max()),
                    'weights_max_abs':float((left['weights']-right['weights']).abs().max()),
                    'different_codes_per_level':(left['tokens']!=right['tokens']).sum(0).tolist(),
                    'cost_max_abs_per_level':[float((a-b).abs().max()) for a,b in zip(left['costs'],right['costs'])]})
            comparisons[label]=rows
        report['comparisons_to_calibration']=comparisons
        report['trace_calls']={name:len(rows) for name,rows in traces.items()}
        path=output/('rank_%d_arrays.npz'%rank)
        np.savez(path,**arrays)
        report['arrays']=str(path)
        report['wall_seconds']=time.monotonic()-started
        (output/('rank_%d.json'%rank)).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('budget-point',rank,report['status'],run_id,flush=True)
        training.RQVAE.quantize=original_quantize
        training.RQVAE.calibrate_curvature_weight=original_calibrate
        mechanism.mean_gradients=original_mean


def main():
    return launcher.launch(SOURCE/'budget_point_entry.py',
        EVIDENCE/'budget_point_launcher'/str(os.getpid()),python=PYTHON,log_name='budget_point.log')


if __name__=='__main__':
    raise SystemExit(main())
