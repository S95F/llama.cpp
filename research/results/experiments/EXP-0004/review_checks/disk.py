# EXP-0004 review check: slow-copy bytes against Windows disk-read counters, aligned to the
# trace's decode window (first to last logits read-back), with the per-interval correlation.
#   python disk.py
# Reads C:/nsys/exp0004/<arm>_long.sqlite and paging_<arm>_long.csv. typeperf writes local
# time; the UTC-6 offset below is the measuring host's zone (MDT) on 2026-10-31.
# Written during the adversarial review and kept unchanged below this header.
import sqlite3, csv, datetime, collections, sys
D='C:/code/llama.cpp-fork/research/results/experiments/EXP-0004/'
GiB=2**30
for arm in ('2048-pageable','2048-pinned','586-pageable'):
    con=sqlite3.connect(f'file:C:/nsys/exp0004/{arm}_long.sqlite?mode=ro',uri=True)
    t0=con.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]/1e9
    htod=[r[0] for r in con.execute('select id,name from ENUM_CUDA_MEMCPY_OPER') if 'HTOD' in r[1].upper()]
    dtoh=[r[0] for r in con.execute('select id,name from ENUM_CUDA_MEMCPY_OPER') if 'DTOH' in r[1].upper()]
    mc=list(con.execute('select start,end,bytes,copyKind,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY order by start'))
    api={c:(s,e) for s,e,c in con.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    sz=collections.Counter(m[2] for m in mc if m[3] in dtoh and m[2]>=65536).most_common(1)[0][0]
    ends=[m[1] for m in mc if m[3] in dtoh and m[2]==sz]
    slow=[];allexp=[]
    for s,e,b,k,c in mc:
        if k in htod and b>=512*1024 and c in api:
            a=api[c]; r=b/max(1,a[1]-a[0])
            allexp.append((a[0]/1e9+t0,a[1]/1e9+t0,b,r))
            if r<1.5: slow.append((a[0]/1e9+t0,a[1]/1e9+t0,b))
    rr=list(csv.reader(open(D+f'paging_{arm}_long.csv')))
    pts=[]
    for r in rr[1:]:
        try:
            ts=datetime.datetime.strptime(r[0],'%m/%d/%Y %H:%M:%S.%f').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-6))).timestamp()
            pts.append((ts,float(r[5]),float(r[1])))
        except Exception: pts.append((datetime.datetime.strptime(r[0],'%m/%d/%Y %H:%M:%S.%f').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-6))).timestamp(),None,None))
    ivs=[(pts[i-1][0],pts[i][0],pts[i][1],pts[i][2]) for i in range(1,len(pts)) if pts[i][1] is not None]
    gaps=[b-a for a,b,_,_ in ivs]
    def disk(lo,hi,col=2):
        tot=0
        for a,b,r,p in ivs:
            ov=min(b,hi)-max(a,lo)
            if ov>0: tot+=(r if col==2 else p*4096)*ov
        return tot
    def cop(L,lo,hi):
        tot=0
        for a,b,by,*_ in L:
            ov=min(b,hi)-max(a,lo)
            if ov>0: tot+=by*ov/max(1e-12,b-a)
        return tot
    lo,hi=ends[0]/1e9+t0, ends[-1]/1e9+t0
    wall=dict(l.split() for l in open(D+f'wall_{arm}_long.txt'))
    ws,we=float(wall['start_unix']),float(wall['end_unix'])
    print(arm,f'sess-wall={t0-ws:.2f}s decode {lo-ws:.2f}..{hi-ws:.2f}s; typeperf {ivs[0][0]-ws:.2f}..{ivs[-1][1]-ws:.2f}, max gap {max(gaps):.2f}s')
    print(f'  decode: disk {disk(lo,hi)/GiB:.2f} GiB, pagesinput {disk(lo,hi,3)/GiB:.2f} GiB, slow-copy bytes {cop(slow,lo,hi)/GiB:.2f} GiB, 1.5-4GB/s copies {cop([x for x in allexp if 1.5<=x[3]<4],lo,hi)/GiB:.2f} GiB')
    print(f'  before decode: disk {disk(ws,lo)/GiB:.2f}; whole wall run: disk {disk(ws,we)/GiB:.2f} GiB, sum-of-rates (1s) {sum(r for a,b,r,p in ivs if ws<=b<=we+1)/GiB:.2f}; all typeperf {disk(-1e18,1e18)/GiB:.2f}')
    # per-second correlation, binned by typeperf intervals inside decode
    xs=[];ys=[]
    for a,b,r,p in ivs:
        if a>=lo and b<=hi:
            xs.append(r*(b-a)); ys.append(cop(slow,a,b))
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    print(f'  full intervals in decode: {n}, disk {sum(xs)/GiB:.2f} vs slow {sum(ys)/GiB:.2f} GiB, r={cov/(vx*vy)**.5:.3f}')
