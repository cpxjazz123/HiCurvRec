"""Small dependency-free SVG plots for the issue #261 report."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader(open(OUT/'CORRELATIONS.csv')))
def f(x):
 try:return float(x)
 except:return None
def choose(target='test_recall@10'):
 want=['validation.positive.0','validation.l1_hard_er','sid.distinct_L2','sid.collision','hard_q.curvature.micro_mae','hard_q.reconstruction.mean_l2_error','quant.codebooks.0.spacing_mean','quant.effective_rank_z','hard_q.strict_heldout.banks.strict_struct_test_shared_in.topology.jaccard_in.distance_spearman','training.curvature_windows.last','training.recon_windows.last','relations.no_edge_L1.rate']
 by={r['predictor']:r for r in rows if r['target']==target}
 return [(w,by[w]) for w in want if w in by and f(by[w]['pearson']) is not None]
sel=choose(); targs=['test_recall@5','test_recall@10','test_ndcg@5','test_ndcg@10']
by={(r['predictor'],r['target']):r for r in rows}
cellw,cellh=70,34; left=330; top=55
W=left+cellw*4+20; H=top+cellh*len(sel)+20
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
 '<rect x="0" y="0" width="100%" height="100%" fill="none"/>']
for i,t in enumerate(targs): parts.append(f'<text x="{left+i*cellw+cellw/2}" y="30" text-anchor="middle" font-size="13" fill="currentColor">{t.replace("test_","")}</text>')
for j,(name,_) in enumerate(sel):
 y=top+j*cellh
 parts.append(f'<text x="{left-10}" y="{y+21}" text-anchor="end" font-size="11" fill="currentColor">{name[:58]}</text>')
 for i,t in enumerate(targs):
  r=by.get((name,t)); v=f(r['pearson']) if r else None
  if v is None: fill='var(--surface)'; txt='NA'
  else:
   # red negative, blue positive through a neutral surface
   strength=min(1.0,abs(v)); col=('var(--error)' if v<0 else 'var(--accent)')
   parts.append(f'<rect x="{left+i*cellw+2}" y="{y+2}" width="{cellw-4}" height="{cellh-4}" rx="4" fill="{col}" fill-opacity="{0.12+0.68*strength}"/>')
   txt=f'{v:+.2f}'
  parts.append(f'<text x="{left+i*cellw+cellw/2}" y="{y+21}" text-anchor="middle" font-size="11" fill="currentColor">{txt}</text>')
parts.append('</svg>'); (OUT/'HEATMAP.svg').write_text('\n'.join(parts)+'\n')
# scatter validation.positive.0 vs test R10
name='validation.positive.0'; r=by.get((name,'test_recall@10')); series=json.load(open(OUT/'CORRELATIONS.json'))['series'][f'{name}::test_recall@10']['series']
pts=[(x['x'],x['y'],x['index']) for x in series if x.get('x') is not None and x.get('y') is not None]
xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
def sx(v): return 70+(v-xmin)/(xmax-xmin+1e-12)*560
def sy(v): return 340-(v-ymin)/(ymax-ymin+1e-12)*270
p=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 400" width="700" height="400">',
 '<text x="70" y="22" font-size="14" fill="currentColor">L1 positive hit rate vs Stage3 test R@10</text>',
 f'<line x1="70" y1="340" x2="630" y2="340" stroke="var(--border)"/><line x1="70" y1="70" x2="70" y2="340" stroke="var(--border)"/>']
for v,l in ((xmin,f'{xmin:.4f}'),(xmax,f'{xmax:.4f}')): p.append(f'<text x="{sx(v)}" y="365" text-anchor="middle" font-size="10" fill="var(--muted)">{l}</text>')
for v,l in ((ymin,f'{ymin:.4f}'),(ymax,f'{ymax:.4f}')): p.append(f'<text x="60" y="{sy(v)+4}" text-anchor="end" font-size="10" fill="var(--muted)">{l}</text>')
for x,y,i in pts: p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="var(--accent)"/><text x="{sx(x)+7:.1f}" y="{sy(y)-7:.1f}" font-size="10" fill="currentColor">{i}</text>')
p.append(f'<text x="350" y="392" text-anchor="middle" font-size="11" fill="var(--muted)">Pearson {f(r["pearson"]):+.3f} · Spearman {f(r["spearman"]):+.3f} · n=8</text></svg>')
(OUT/'SCATTER_L1.svg').write_text('\n'.join(p)+'\n')
print(len(sel),'heatmap rows')
