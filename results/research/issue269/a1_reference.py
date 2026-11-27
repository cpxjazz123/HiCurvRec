"""A.1 curvature prediction of the frozen Z_ref under the calibrated kappa."""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, spearmanr
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tools.graph_infra.assets import load_sparse
import importlib.util
spec=importlib.util.spec_from_file_location("own", ROOT/"results/research/issue269/own_kappa.py")
own=importlib.util.module_from_spec(spec); spec.loader.exec_module(own)
S=own.SCALE
z=np.load(ROOT/"results/research/issue268/artifacts/z_ref.npy").astype(np.float64)
g=load_sparse(own.GRAPH).tocsr().astype(np.int8); g=g.maximum(g.T).tocsr(); g.setdiag(0); g.eliminate_zeros()
coo=g.tocoo(); up=coo.row<coo.col
edges=set((coo.row[up].astype(np.int64)*len(z)+coo.col[up].astype(np.int64)).tolist())
_,labels=connected_components(g,directed=False)
rng=np.random.default_rng(own.SEED); blocks=[]
for c in range(int(labels.max())+1):
    m=np.flatnonzero(labels==c)
    if len(m)>=2: blocks.append(np.sort(rng.choice(m,min(own.REFERENCE_SIZE,len(m)),replace=False)))
refs=np.concatenate(blocks); gd=shortest_path(g,unweighted=True,directed=False,indices=refs).T
use=np.isfinite(gd)&(gd>0); gv=np.where(use,gd,1.0).astype(np.float32)
T=np.load(own.TARGETS,mmap_mode="r"); O=np.load(own.OFFSETS,mmap_mode="r")
rows=np.sort(np.random.default_rng(own.SEED+1).choice(int(O[-1]),own.SAMPLE_ROWS,replace=False)); rows=rows[np.isfinite(np.asarray(T["target"][rows]))]
cen=np.searchsorted(O,rows,side="right")-1; b=np.asarray(T["b"][rows],dtype=np.int64); c=np.asarray(T["c"][rows],dtype=np.int64)
target=np.asarray(T["target"][rows],dtype=np.float64)
kap=np.load(ROOT/"results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/kappa.npy",mmap_mode="r")
ks=np.asarray(kap[rows],dtype=np.float64)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
w=torch.from_numpy((z/(2*S)).astype(np.float32)).to(dev); wr=w[torch.from_numpy(refs).to(dev)]
gvt=torch.from_numpy(gv).to(dev); ut=torch.from_numpy(use).to(dev)
pg=np.fromiter((1.0 if (min(int(u),int(v))*len(z)+max(int(u),int(v))) in edges else 2.0 for u,v in zip(b,c)),dtype=np.float32,count=len(b))
pred=np.empty(len(rows)); denom=(use[cen].sum(1)+use[b].sum(1)+use[c].sum(1)+1.0)
for start in range(0,len(rows),256):
    stop=min(start+256,len(rows)); kt=torch.from_numpy(ks[start:stop].astype(np.float32)).to(dev)
    tot=[]
    for ids in (cen[start:stop],b[start:stop],c[start:stop]):
        node=torch.from_numpy(ids.astype(np.int64)).to(dev)
        d2=own.squared(w[node][:,None,:].expand(-1,len(refs),-1),wr[None],kt[:,None])
        r=d2.clamp_min(0).sqrt()/gvt[node]
        tot.append(torch.where(ut[node],(r-1).square(),torch.zeros_like(r)).sum(1))
    pair=own.squared(w[torch.from_numpy(b[start:stop].astype(np.int64)).to(dev)],w[torch.from_numpy(c[start:stop].astype(np.int64)).to(dev)],kt)
    pred[start:stop]=(tot[0]-0.5*(tot[1]+tot[2])+0.25*pair).cpu().numpy()/denom[start:stop]
out={"schema":"genrec.issue269.a1_reference/1","space":"Z_ref","rows":int(len(rows)),"references":int(len(refs)),
     "pearson":float(pearsonr(target,pred).statistic),"spearman":float(spearmanr(target,pred).statistic),
     "mae":float(np.mean(np.abs(pred-target))),"rmse":float(np.sqrt(np.mean((pred-target)**2))),
     "sign_accuracy":float((np.sign(target)==np.sign(pred)).mean()),
     "prediction_mean":float(pred.mean()),"target_mean":float(target.mean()),
     "z_ref_sha256":own.sha(ROOT/"results/research/issue268/artifacts/z_ref.npy"),
     "cache_sha256":own.sha(own.CACHE/"kappa.npy")}
(ROOT/"results/research/issue269/analysis/A1_ZREF.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=1))
