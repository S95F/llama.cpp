# EXP-0004 review check: per-window decomposition of the timeline JSONs (warm windows,
# cold short runs, and the pinned-minus-pageable gain per window). Run from anywhere:
#   python win.py
# Written during the adversarial review and kept unchanged below this header.
import json,sys
D='C:/code/llama.cpp-fork/research/results/experiments/EXP-0004/'
def load(a):
    return json.load(open(D+f'timeline_{a}.json'))['tokens']
def agg(T,lo,hi):
    S=[t for t in T if lo<=t['token']<=hi]
    n=len(S)
    f=lambda k:sum(t[k] for t in S)/n/1e6
    g=lambda k:sum(t['idle_while_host_in_ns'][k] for t in S)/n/1e6
    pin=sum(t['h2d_by_src_kind'].get('pinned',{}).get('bytes',0) for t in S)
    tot=sum(t['h2d_bytes'] for t in S)
    return dict(n=n,ms=f('t_ns'),comp=f('compute_ns'),dma=f('h2d_ns'),d2h=f('d2h_ns'),idle=f('idle_ns'),slow=g('expert_copy_slow'),fast=g('expert_copy_fast'),sync=g('cudaStreamSynchronize'),slowN=sum(t['n_expert_copy_slow'] for t in S)/n,mib=sum(t['h2d_expert_bytes'] for t in S)/n/2**20,pinshare=pin/tot)
arms=['2048-pageable_long','2048-pinned_long','586-pageable_long']
R={}
for a in arms:
    T=load(a); R[a]={}
    for lo,hi in [(3,126),(3,130),(131,386),(387,1022),(771,898),(899,1022),(131,1022)]:
        r=agg(T,lo,hi); R[a][(lo,hi)]=r
        print(a,lo,hi,' '.join(f'{k}={v:.2f}' if isinstance(v,float) else f'{k}={v}' for k,v in r.items()))
for a in ['2048-pageable_t1','2048-pageable_t2','2048-pinned_t1','2048-pinned_t2','586-pageable_t1','586-pageable_t2','1024-pageable_t1','1024-pinned_t1']:
    T=load(a); r=agg(T,3,126)
    print(a,' '.join(f'{k}={v:.2f}' if isinstance(v,float) else f'{k}={v}' for k,v in r.items()))
pg,pn='2048-pageable_long','2048-pinned_long'
for w in R[pg]:
    a,b=R[pg][w],R[pn][w]
    print('pin gain',w,' '.join(f'{k}={b[k]-a[k]:.2f}' for k in ['ms','dma','slow','fast','sync']))
