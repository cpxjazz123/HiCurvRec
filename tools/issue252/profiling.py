"""One actual trainer optimizer step, with initialization and nested phase costs.

This is instrumentation, not a throughput benchmark or a substitute training run.
CUDA spans are stream elapsed times; nested phase totals must not be summed.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import functools
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function


class ProfileComplete(Exception):
    pass


class Tracker:
    def __init__(self, output, boot_time, training):
        self.output=Path(output);self.boot_time=boot_time;self.training=training
        self.stage='initialization';self.rows=[];self.footprints=defaultdict(Counter)
        self.steps=0;self.profiler=None;self.window_start=None
        self.loader=None;self.init_peak=0;self.initialization_seconds=None

    @contextmanager
    def section(self,label):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        stage=self.stage;start.record();begin=time.perf_counter()
        with record_function('phase::'+label):
            try:yield
            finally:
                wall=time.perf_counter()-begin;end.record()
                self.rows.append((stage,label,wall,start,end))

    def wrap(self,parent,name,label,observe=None):
        original=getattr(parent,name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            actual=label(*args,**kwargs) if callable(label) else label
            if observe is not None:observe(args,kwargs,None,self.stage,actual)
            with self.section(actual):result=original(*args,**kwargs)
            if observe is not None and result is not None:observe(args,kwargs,result,self.stage,actual)
            return result
        setattr(parent,name,wrapped)

    def begin_step(self):
        if self.profiler is not None:return
        torch.cuda.synchronize()
        self.initialization_seconds=time.perf_counter()-self.boot_time
        self.init_peak=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self.stage='training_step'
        self.profiler=profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                              record_shapes=False,profile_memory=False,with_stack=False)
        self.profiler.__enter__();self.window_start=time.perf_counter()

    def install(self):
        import pandas
        import model.geometry as geometry
        import model.curvature as curvature
        import model.selection as selection
        import model.layers as layers
        from model.mechanism import RQVAE
        from model.structure import SubgraphTokens
        from model.attention import ExpertAttention
        from tools.graph_infra.ego_gnn import EgoStore
        self.wrap(self.training,'load_embeddings','embedding_read')
        self.wrap(pandas,'read_parquet','train_parquet_read')
        self.wrap(RQVAE,'init_codebook','codebook_initialization')
        self.wrap(RQVAE,'freeze_tangent_scale','scale_calibration')
        self.wrap(RQVAE,'calibrate_curvature_weight','gradient_calibration')
        self.wrap(EgoStore,'collate','ego_build',self.observe_ego)
        self.wrap(SubgraphTokens,'forward','GNN',self.observe_shapes)
        self.wrap(ExpertAttention,'forward','Attention',self.observe_shapes)
        self.wrap(layers.MLP,'forward',lambda owner,x:'Encoder' if x.shape[-1]==768 else 'Decoder',self.observe_shapes)
        self.wrap(RQVAE,'catalogue_q','whole_catalogue_forward',self.observe_shapes)
        self.wrap(RQVAE,'assignment_cost','five_geometry_assignment_cost',self.observe_shapes)
        self.wrap(geometry,'expert_squared','five_geometry_distances',self.observe_shapes)
        self.wrap(layers.VQLayer,'sinkhorn','hard_Sinkhorn',self.observe_shapes)
        self.wrap(selection,'probabilities','soft_allocation',self.observe_shapes)
        self.wrap(layers,'constrained_assignment','assignment_floor',self.observe_shapes)
        for name in ('verify_assignment','verify_codebook','verify_margins','verify_separation'):
            self.wrap(layers,name,'assignment_guards',self.observe_shapes)
        self.wrap(curvature,'reference_moments','full_reference_distances',self.observe_shapes)
        self.wrap(curvature,'sectional_values','triple_statistic',self.observe_shapes)
        self.wrap(curvature.FrozenCurvature,'loss','full_curvature_loss',self.observe_curvature)
        blocks=curvature.FrozenCurvature.blocks
        tracker=self
        @functools.wraps(blocks)
        def timed_blocks(*args,**kwargs):
            iterator=blocks(*args,**kwargs)
            while True:
                try:
                    with tracker.section('target_read_and_unpack'):block=next(iterator)
                except StopIteration:return
                tracker.footprints[(tracker.stage,'target_read_and_unpack')]['rows']+=len(block[0])
                yield block
        curvature.FrozenCurvature.blocks=timed_blocks
        self.wrap(torch.autograd,'backward','backward',self.observe_shapes)
        data_loader=self.training.DataLoader
        class MeasuredLoader:
            def __init__(self,inner):self.inner=inner
            def __len__(self):return len(self.inner)
            def __iter__(self):
                iterator=iter(self.inner)
                while True:
                    tracker.begin_step()
                    try:
                        with tracker.section('data_read'):batch=next(iterator)
                    except StopIteration:return
                    yield batch
        def make_loader(*args,**kwargs):
            inner=data_loader(*args,**kwargs)
            self.loader={'items':len(inner.dataset),'steps_per_epoch':len(inner),
                         'batch_per_rank':inner.batch_size}
            return MeasuredLoader(inner)
        self.training.DataLoader=make_loader
        for cls in (torch.optim.AdamW,torch.optim.Adagrad):
            original=cls.step
            def one_step(optimizer,*args,_original=original,**kwargs):
                with tracker.section('optimizer'):result=_original(optimizer,*args,**kwargs)
                tracker.steps+=1
                raise ProfileComplete()
            cls.step=one_step

    def observe_ego(self,args,kwargs,result,stage,label):
        if result is not None:
            row=self.footprints[(stage,label)]
            row['items']+=len(result['items']);row['expanded_nodes']+=len(result['nodes'])
            row['expanded_edges']+=len(result['edges'])
            row['numpy_bytes']+=sum(value.nbytes for value in result.values() if hasattr(value,'nbytes'))

    def observe_shapes(self,args,kwargs,result,stage,label):
        if result is None:
            shapes=tuple(tuple(value.shape) for value in args if torch.is_tensor(value))
            self.footprints[(stage,label)]['shapes:'+str(shapes)]+=1

    def observe_curvature(self,args,kwargs,result,stage,label):
        if result is not None:
            self.footprints[(stage,label)]['valid_rows']+=args[0].last_rows
            self.footprints[(stage,label)]['centres']+=args[0].last_centres

    def finish(self,error,source_hashes):
        torch.cuda.synchronize();window=None if self.window_start is None else time.perf_counter()-self.window_start
        peak=torch.cuda.max_memory_allocated()
        if self.profiler is not None:self.profiler.__exit__(None,None,None)
        totals=defaultdict(lambda:{'calls':0,'cpu_dispatch_wall_seconds':0.,'cuda_stream_span_seconds':0.})
        for stage,label,wall,start,end in self.rows:
            row=totals[(stage,label)];row['calls']+=1
            row['cpu_dispatch_wall_seconds']+=wall;row['cuda_stream_span_seconds']+=start.elapsed_time(end)/1000
        phases=[{'stage':stage,'phase':label,**row,'footprint':dict(self.footprints[(stage,label)])}
                for (stage,label),row in sorted(totals.items())]
        operators=[]
        if self.profiler is not None:
            for event in self.profiler.key_averages():
                device=float(getattr(event,'self_device_time_total',getattr(event,'self_cuda_time_total',0.)))
                if device>0 or any(word in event.key.lower() for word in ('allreduce','nccl','c10d')):
                    operators.append({'name':event.key,'calls':event.count,
                        'self_cpu_seconds':event.self_cpu_time_total/1e6,'self_device_seconds':device})
        operators.sort(key=lambda row:row['self_device_seconds'],reverse=True)
        rank=int(__import__('os').environ['RANK']);world=int(__import__('os').environ['WORLD_SIZE'])
        return {'status':'PROFILED' if error is None and self.steps==1 else 'FAIL','rank':rank,
            'world_size':world,'global_batch':self.training.BATCH_SIZE_PER_RANK*world,
            'seed':self.training.SEED,'completed_optimizer_steps':self.steps,'completed_epochs':0,
            'formal_training':False,'run_id':'issue252_baseline_profile_'+str(__import__('os').getppid()),
            'worker_pid':__import__('os').getpid(),'master_port':int(__import__('os').environ['MASTER_PORT']),
            'torch':torch.__version__,'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'initialization_seconds':self.initialization_seconds,'initialization_peak_bytes':self.init_peak,
            'actual_one_step_wall_seconds':window,'training_step_peak_bytes':peak,'loader':self.loader,
            'phases':phases,'cuda_operators':operators,'source_hashes':source_hashes,'error':error,
            'interpretation':'One full-sized actual trainer step with instrumentation; not a completed epoch or speed acceptance. Phase ranges are nested/inclusive, CUDA spans include stream waits; do not sum them. Scheduler/evaluation/export after optimizer step were not executed.'}


def run(training,output,boot_time):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=training.REPO_ROOT;tree=Path(training.__file__).resolve().parent
    paths=list(tree.rglob('*.py'))+list((tree/'config').glob('*.json'))
    paths += [root/name for name in ('assets/manifest.json','tools/issue252/profiling.py',
               'tools/issue252/launch_profile.py','tools/graph_infra/ego_gnn.py',
               'tools/graph_infra/assets.py','tools/research/stage2/launcher.py')]
    hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if '__pycache__' not in path.parts}
    training.EPOCHS=1;training.RESULT_DIR=output
    training.LOG_DIR=output/'logs';training.METRICS_PATH=training.LOG_DIR/'training_metrics.jsonl'
    training.OUTPUT_SID=output/'sids_for_hgrec.npy';training.OUTPUT_JSON=output/'item_sids.json'
    training.CHECKPOINT=output/'rqvae_best.pth'
    tracker=Tracker(output,boot_time,training);tracker.install();error=None
    try:training.main()
    except ProfileComplete:pass
    except BaseException:error=traceback.format_exc()
    try:
        result=tracker.finish(error,hashes)
        for name,expected in hashes.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                result['status']='FAIL';result['error']='source/input changed while profiling: '+name
        (output/('rank_%d.json'%result['rank'])).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('[issue252 baseline profile]',result['rank'],result['status'],result['actual_one_step_wall_seconds'],flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()
    if result['status']!='PROFILED':raise RuntimeError(result['error'] or 'no completed optimizer step')
