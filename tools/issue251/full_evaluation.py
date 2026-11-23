"""Full-table CPU remeasurement with independent ambient geometry.

Unlike the training radial/angular formulas, use Lorentz/sphere inner products
and symmetric upper-block accumulation. No production curvature/selection import.
"""
from pathlib import Path
import math
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .evaluation import _curvature_report


def _fixed(x,y,scale,paired):
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('scale must be finite and positive')
    x=np.asarray(x,dtype=np.float64)/scale
    y=np.asarray(y,dtype=np.float64)/scale
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1]:
        raise ValueError('expected matching finite tangent matrices')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite tangent coordinates')
    if paired and x.shape!=y.shape:
        raise ValueError('paired coordinates must align')
    rx=np.linalg.norm(x,axis=-1);ry=np.linalg.norm(y,axis=-1)
    if max(rx.max(initial=0),ry.max(initial=0))*math.sqrt(3)>300:
        raise ValueError('ambient reference exceeds its finite numerical domain')
    ux=x/np.where(rx>0,rx,1)[:,None];uy=y/np.where(ry>0,ry,1)[:,None]
    cosine=np.einsum('ij,ij->i',ux,uy) if paired else ux@uy.T
    cosine=np.clip(cosine,-1.,1.)
    a,b=(rx,ry) if paired else (rx[:,None],ry[None,:])
    result=np.maximum(a*a+b*b-2*a*b*cosine,0.)
    for curvature in (-3.,-1.):
        k=math.sqrt(-curvature)
        inner=np.cosh(k*a)*np.cosh(k*b)-np.sinh(k*a)*np.sinh(k*b)*cosine
        result+=np.arccosh(np.maximum(inner,1.))**2/(-curvature)
    for curvature in (1.,3.):
        k=math.sqrt(curvature)
        inner=np.cos(k*a)*np.cos(k*b)+np.sin(k*a)*np.sin(k*b)*cosine
        result+=np.arccos(np.clip(inner,-1.,1.))**2/curvature
    return result/5.


def fixed_squared_numpy(x,y,scale=1.):
    return _fixed(x,y,scale,False)


def fixed_paired_numpy(x,y,scale=1.):
    return _fixed(x,y,scale,True)


def _moments(q,labels,scale,metric,chunk):
    means=np.zeros(len(q),dtype=np.float64)
    for component in np.unique(labels):
        indices=np.flatnonzero(labels==component)
        totals=np.zeros(len(indices),dtype=np.float64)
        for start in range(0,len(indices),chunk):
            left=indices[start:start+chunk]
            for offset in range(start,len(indices),chunk):
                right=indices[offset:offset+chunk]
                distance=metric(q[left],q[right],scale=scale)
                totals[start:start+len(left)]+=distance.sum(axis=1)
                if offset!=start:
                    totals[offset:offset+len(right)]+=distance.sum(axis=0)
        means[indices]=totals/len(indices)
    return means


def full_evaluate(q,assets,*,scale=1.,metric=fixed_squared_numpy,
                  paired_metric=fixed_paired_numpy,chunk=512,triple_chunk=16384):
    """Every frozen valid triple and every legal reference, without sampling.

    Distance distortion uses every triple's (b,c) pair with its frozen graph
    distance. Pair repetition is explicit triple weighting, never a fitted scale.
    """
    started=time.perf_counter();assets=Path(assets)
    q=np.asarray(q,dtype=np.float64)
    if chunk<=0 or triple_chunk<=0 or q.ndim!=2 or not np.isfinite(q).all():
        raise ValueError('invalid full-evaluation input or chunk')
    with np.load(assets/'node_moments.npz') as blob:labels=blob['component']
    with np.load(assets/'index.npz') as blob:names=[str(x) for x in blob['shard_names']]
    if labels.shape!=(len(q),):raise ValueError('components must cover the full catalogue')
    with threadpool_limits(limits=1):
        moments=_moments(q,labels,scale,metric,chunk)
    per_item={};count=excluded=0;error_sum=macro_sum=correct_sum=0.
    distortion_sum=relative_sum=0.;relative_count=0
    strata={name:{'count':0,'absolute_sum':0.,'macro_sum':0.,'centres':0,'correct':0.}
            for name in ('negative','zero','positive')}
    for name in names:
        with np.load(assets/'shards'/name) as blob:
            source={key:blob[key] for key in ('centre','b','c','valid','curvature','bc_dist_sq')}
        valid=source['valid'];excluded+=int((~valid).sum())
        centre=source['centre'][valid];b=source['b'][valid];c=source['c'][valid]
        target=source['curvature'][valid];graph_squared=source['bc_dist_sq'][valid]
        if not len(centre):continue
        if min(centre.min(),b.min(),c.min())<0 or max(centre.max(),b.max(),c.max())>=len(q):
            raise ValueError('triple outside full catalogue')
        if not np.all((labels[centre]==labels[b])&(labels[centre]==labels[c])):
            raise ValueError('triple crosses a frozen component')
        if not np.isfinite(target).all() or not np.isfinite(graph_squared).all() or (graph_squared<0).any():
            raise ValueError('invalid lawful target or graph distance')
        predicted=np.empty(len(centre),dtype=np.float64)
        for start in range(0,len(centre),triple_chunk):
            end=min(start+triple_chunk,len(centre));bb=b[start:end];cc=c[start:end]
            pair=paired_metric(q[bb],q[cc],scale=scale)
            predicted[start:end]=(moments[centre[start:end]]-.5*(moments[bb]+moments[cc])+.25*pair)
            graph=np.sqrt(graph_squared[start:end]);error=np.abs(np.sqrt(pair)-graph)
            distortion_sum+=float(error.sum())
            positive=graph>0;relative_sum+=float((error[positive]/graph[positive]).sum())
            relative_count+=int(positive.sum())
        if not np.isfinite(predicted).all():raise ValueError('nonfinite full-table prediction')
        items,starts,counts=np.unique(centre,return_index=True,return_counts=True)
        for item,start,size in zip(items,starts,counts):
            if not np.all(centre[start:start+size]==item) or str(item) in per_item:
                raise ValueError('frozen centre rows must be contiguous and owned by one shard')
            local=_curvature_report(target[start:start+size],predicted[start:start+size],
                                    centre[start:start+size],0)
            per_item[str(item)]=local['per_item'][str(item)]
            count+=int(size);error_sum+=local['micro_mae']*size;macro_sum+=local['macro_mae']
            correct_sum+=local['sign']['accuracy']*size
            for sign,total in strata.items():
                error=local['strata'][sign]
                if error['status']=='NA':continue
                n=error['count'];total['count']+=n;total['centres']+=1
                total['absolute_sum']+=error['micro_mae']*n;total['macro_sum']+=error['macro_mae']
                total['correct']+=local['sign']['recall'][sign]['recall']*n
        print('[full-evaluation] %s valid=%d cumulative=%d'%(name,len(centre),count),flush=True)
    if not count:raise ValueError('no valid frozen triples')
    stratified={};recalls={};present=[]
    for name,total in strata.items():
        if not total['count']:
            stratified[name]={'status':'NA','reason':'target sign absent'}
            recalls[name]={'status':'NA','reason':'target sign absent'}
        else:
            present.append(name)
            stratified[name]={'status':'OK','count':total['count'],
                'micro_mae':total['absolute_sum']/total['count'],
                'macro_mae':total['macro_sum']/total['centres']}
            recalls[name]={'status':'OK','count':total['count'],'recall':total['correct']/total['count']}
    return {'reference_scope':'entire frozen component','shards':names,
        'metric':'equal_weight_five_expert_squared_distance; independent ambient implementation',
        'tangent_scale':float(scale),'max_scaled_tangent_norm':float(np.linalg.norm(q,axis=-1).max()/scale),
        'curvature':{'status':'OK','count':count,'excluded':excluded,'centres':len(per_item),
            'macro_mae':macro_sum/len(per_item),'micro_mae':error_sum/count,'strata':stratified,
            'per_item':per_item,'sign':{'accuracy':correct_sum/count,
                'balanced_accuracy':float(np.mean([recalls[x]['recall'] for x in present])),
                'present_signs':present,'recall':recalls,'zero_tolerance':0.}},
        'distance_distortion':{'count':count,'mean_abs_error':distortion_sum/count,
            'mean_relative_error':relative_sum/relative_count if relative_count else None,
            'distance_scale':1.,'scale_fitted_on_evaluation':False,
            'scope':'all valid frozen triple (b,c) pairs, weighted by triple occurrence'},
        'wall_seconds':time.perf_counter()-started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
