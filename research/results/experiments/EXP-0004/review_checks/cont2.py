# EXP-0004 review check: GPU-side rate of pageable expert copies when the host is, or is not,
# staging the next pageable copy at the same time (contended = >50% overlap).
#   python cont2.py C:/nsys/exp0004/<tag>.sqlite
# Written during the adversarial review and kept unchanged below this header.
import sys, sqlite3, bisect, collections
db=sys.argv[1]
con=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
rt=list(con.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"))
api={c:(s,e) for s,e,c in rt}
sz=collections.Counter(r[0] for r in con.execute("select bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and bytes>=65536")).most_common(1)[0][0]
ends=[r[0] for r in con.execute("select end from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and bytes=? order by start",(sz,))]
lo,hi=ends[2],ends[-2]
mc=list(con.execute("select start,end,bytes,srcKind,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes>=524288 and start>=? and start<? order by start",(lo,hi)))
cls={m[4]:("slow" if m[2]/max(1,api[m[4]][1]-api[m[4]][0])<1.5 else "fast") for m in mc}
src={m[4]:m[3] for m in mc}
rs=[r[0] for r in rt]
acc=collections.defaultdict(lambda:[0,0,0])
nextgap=collections.defaultdict(list)
for i,m in enumerate(mc):
    if m[3]!=0: continue
    j=max(bisect.bisect_left(rs,m[0])-1,0); ov=0; k=j
    while k<len(rt) and rt[k][0]<m[1]:
        s,e,c=rt[k]
        if e>m[0] and c in cls and src.get(c)==0 and cls[c]=="fast": ov+=min(e,m[1])-max(s,m[0])
        k+=1
    grp="contended" if ov/(m[1]-m[0])>0.5 else "free"
    for key in [(grp,'all'),(grp,m[2])]:
        a=acc[key]; a[0]+=1; a[1]+=m[2]; a[2]+=m[1]-m[0]
tot={g:sum(acc[(g,'all')][2] for _ in [0]) for g in ('contended','free')}
for k,(n,b,t) in sorted(acc.items(),key=str):
    if k[1]=='all' or n>200: print(db.split('/')[-1],k,"n",n,"GB/s %.2f"%(b/t),"ms_total %.0f"%(t/1e6))
