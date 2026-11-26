"""Small-sample Pearson/Spearman with explicit n and NA handling."""
from __future__ import annotations
import math
from itertools import combinations

def _finite(xs,ys):
 pairs=[(float(x),float(y)) for x,y in zip(xs,ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
 if len(pairs)<2: return None
 return pairs

def rankdata(values):
 """Average ranks for ties (1-based)."""
 vals=list(values); order=sorted(range(len(vals)),key=lambda i:(vals[i],i)); ranks=[0.0]*len(vals); i=0
 while i<len(order):
  j=i
  while j+1<len(order) and vals[order[j+1]]==vals[order[i]]: j+=1
  avg=(i+1+j+1)/2.0
  for k in range(i,j+1): ranks[order[k]]=avg
  i=j+1
 return ranks

def pearson(xs,ys):
 pairs=_finite(xs,ys)
 if not pairs or len(pairs)<2: return None
 n=len(pairs); mx=sum(x for x,_ in pairs)/n; my=sum(y for _,y in pairs)/n
 sxx=sum((x-mx)**2 for x,_ in pairs); syy=sum((y-my)**2 for _,y in pairs); sxy=sum((x-mx)*(y-my) for x,y in pairs)
 if sxx<=0 or syy<=0: return None
 return sxy/math.sqrt(sxx*syy)

def spearman(xs,ys):
 pairs=_finite(xs,ys)
 if not pairs or len(pairs)<2: return None
 return pearson(rankdata([x for x,_ in pairs]),rankdata([y for _,y in pairs]))

def fisher_ci(r,n,alpha=0.05):
 if r is None or n is None or n<4 or abs(r)>=1: return (None,None)
 z=math.atanh(r); se=1/math.sqrt(n-3); zc=1.959963984540054
 return (math.tanh(z-zc*se),math.tanh(z+zc*se))

def p_value_pearson(r,n):
 if r is None or n is None or n<3 or abs(r)>=1: return None
 # two-sided t test; normal approximation for the exploratory small n
 t=abs(r)*math.sqrt((n-2)/max(1e-15,1-r*r))
 # regularized incomplete beta via scipy when present; fallback normal
 try:
  from scipy.stats import t as student
  return float(2*student.sf(t,df=n-2))
 except Exception:
  return math.erfc(t/math.sqrt(2))

def leave_one_out(rx,ry):
 pairs=_finite(rx,ry)
 if not pairs or len(pairs)<3: return None
 vals=[]
 for drop in range(len(pairs)):
  sub=[p for i,p in enumerate(pairs) if i!=drop]
  r=pearson([x for x,_ in sub],[y for _,y in sub])
  if r is not None: vals.append(r)
 return (min(vals),max(vals)) if vals else None
