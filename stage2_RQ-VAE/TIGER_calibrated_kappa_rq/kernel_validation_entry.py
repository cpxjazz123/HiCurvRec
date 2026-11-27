"""Four real GPU independent full-ego and compiled-three-geometry checks, no training."""
from pathlib import Path
import sys
import os
import io
import json
import time
import traceback
import unittest

ROOT=next(p for p in Path(__file__).resolve().parents
          if (p/'tools/research/stage2/contract.py').is_file())
TREE=Path(__file__).resolve().parent
sys.path[:0]=[str(TREE),str(ROOT)]


def geometry_checks(device):
    import importlib.util
    import torch
    from model import geometry
    from model.layers import VQLayer
    spec=importlib.util.spec_from_file_location('independent_ambient253',TREE/'tests/reference.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    rows=[]
    generator=torch.Generator(device=device).manual_seed(42)
    for dtype in (torch.float32,torch.float64):
        x=(torch.randn((16,32),device=device,dtype=dtype,generator=generator)*.12).requires_grad_()
        y=(torch.randn((7,32),device=device,dtype=dtype,generator=generator)*.14).requires_grad_()
        weights=torch.softmax(torch.randn((16,3),device=device,dtype=dtype,generator=generator),-1).detach().requires_grad_()
        rx=x.detach().double().requires_grad_();ry=y.detach().double().requires_grad_()
        rw=weights.detach().double().requires_grad_()
        u,v=rx/1.3,ry/1.3
        independent=torch.stack((reference.lorentz_squared(u,v,-1),
            (u[:,None]-v[None]).square().sum(-1),reference.sphere_squared(u,v,1)),-1)
        actual=geometry.expert_squared(x,y,scale=1.3)
        tolerance={'atol':1e-5,'rtol':1e-4} if dtype==torch.float32 else {'atol':1e-9,'rtol':1e-8}
        torch.testing.assert_close(actual.double(),independent,**tolerance)
        mixed=geometry.mixed_cost(x,y,weights,scale=1.3)
        expected=(independent*rw[:,None,:]).sum(-1)
        torch.testing.assert_close(mixed.double(),expected,**tolerance)
        fixed=geometry.fixed_squared(x,y,scale=1.3)
        torch.testing.assert_close(fixed.double(),independent.mean(-1),**tolerance)
        probe=torch.cos(torch.arange(mixed.numel(),device=device,dtype=dtype)).reshape_as(mixed)
        gradients=torch.autograd.grad((mixed*probe).sum()+fixed.sum(),(x,y,weights))
        reference_gradients=torch.autograd.grad((expected*probe.double()).sum()+independent.mean(-1).sum(),(rx,ry,rw))
        for gradient,wanted in zip(gradients,reference_gradients):
            torch.testing.assert_close(gradient.double(),wanted,**tolerance)
        rows.append({'dtype':str(dtype),'experts':3,'matrix_shape':list(actual.shape),
                     'max_value_error':float((actual.double()-independent).abs().max()),
                     'max_gradient_error':max(float((a.double()-b).abs().max()) for a,b in zip(gradients,reference_gradients)),
                     'passed':True})
        codes=torch.zeros((256,32),device=device,dtype=dtype)
        codes[:,0]=-2-torch.arange(256,device=device,dtype=dtype)/256
        codes[0,:2]=torch.tensor([0.,1.],device=device,dtype=dtype)
        codes[1,:2]=torch.tensor([2.45,0.],device=device,dtype=dtype)
        population=codes[torch.arange(1024,device=device)%256].clone()
        population[0].zero_();population[0,0]=1
        layer=VQLayer(256,32,sk_epsilon=-1,sk_iters=50).to(device=device,dtype=dtype).eval()
        winners=[]
        for expert in (1,0):
            gate=torch.zeros((1024,3),device=device,dtype=dtype);gate[:,expert]=1
            cost=geometry.mixed_cost(population,codes,gate)
            winners.append(int(layer._indices(cost,False)[0]))
        assert winners==[0,1],'three-expert cost intervention must change actual floor-constrained hard ID'
        rows[-1]['controlled_hard_ids']=winners
    return rows


def main():
    if int(os.environ.get('WORLD_SIZE','0'))!=4 or int(os.environ.get('MASTER_PORT','0'))!=50202:
        raise RuntimeError('kernel checks require sanctioned four-rank/50202 launcher')
    import torch
    import torch.distributed as dist
    from tools.issue252.training_admission import source_binding
    from tools.issue253.test_ego_kernel import KernelEquivalenceTests
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',local)
    dist.init_process_group('nccl')
    controller=os.getppid()
    tick=(Path('/proc')/str(controller)/'stat').read_text().rsplit(')',1)[1].split()[19]
    run_id='issue253_kernel_validation_'+str(controller)+'_'+tick
    output=ROOT/'results/research/issue253/kernel_validation'/run_id
    output.mkdir(parents=True,exist_ok=True)
    before=source_binding(ROOT,TREE,'validation')
    report={'status':'FAIL','run_id':run_id,'rank':rank,'worker_pid':os.getpid(),
            'controller_pid':controller,'world_size':4,'formal_training':False,
            'source_hashes':before,'device':str(device),'gpu':torch.cuda.get_device_name(device)}
    started=time.monotonic();error=None
    try:
        report['geometry']=geometry_checks(device)
        KernelEquivalenceTests.production_device=str(device)
        sys.path.insert(0, str(TREE/'tests'))
        from test_sync_free_batch import SyncFreeAssemblyTests
        capture=io.StringIO()
        loader=unittest.TestLoader()
        suite=unittest.TestSuite((loader.loadTestsFromTestCase(KernelEquivalenceTests),
                                 loader.loadTestsFromTestCase(SyncFreeAssemblyTests)))
        result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
        text=capture.getvalue();print(text,flush=True)
        log=output/('rank_%d.log'%rank);log.write_text(text)
        report.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                      skipped=len(result.skipped),test_log=str(log))
        if not result.wasSuccessful() or result.testsRun!=10 or result.skipped:
            raise AssertionError('actual CUDA GNN value/input/parameter-gradient suite failed')
        torch.cuda.synchronize(device)
        if before!=source_binding(ROOT,TREE,'validation'):
            raise RuntimeError('source changed during numerical validation')
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        report['status']='PASS'
    except BaseException:
        error=traceback.format_exc();report['error']=error
    finally:
        report['wall_seconds']=time.monotonic()-started
        (output/('rank_%d.json'%rank)).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if dist.is_initialized():dist.destroy_process_group()
    if error:raise RuntimeError(error)

if __name__=='__main__':main()
