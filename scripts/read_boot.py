"""読みの文字の再現率の差の 95% 信頼区間（作品の復元抽出 2,000 回、seed 20261103）。数え方は eval_read.py と同じ（CONT=0）。
使い方: python3 read_boot.py <系A> <系B> ... → 各系の再現率と、隣り合う組の差 A−B の区間"""
import sys,json,csv,collections
import numpy as np
K=8; FILL=0.2
norm=lambda t:"".join(t.split()).replace("／","")
R=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): R[(r["item"],r["ci"])].append(r)
def page(s):
    out={}
    for d in map(json.loads,open(f"../data/read/{s}.jsonl")):
        key=tuple(d["key"]); rs=R[key]
        W=max([int((float(r["x"])+float(r["w"]))/K)+2 for r in rs]+[int(b[2]/K)+2 for b in d["boxes"]]+[1])
        H=max([int((float(r["y"])+float(r["h"]))/K)+2 for r in rs]+[int(b[3]/K)+2 for b in d["boxes"]]+[1])
        G=np.zeros((H,W),bool)
        for r in rs:
            x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        used=[t for b,t in zip(d["boxes"],d["text"]) if (g:=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]).size and g.mean()>=FILL]
        gtc=collections.Counter("".join(norm(r["text"]) for r in rs)); pc=collections.Counter("".join(used))
        out[key]=(sum((gtc&pc).values()),sum(gtc.values()))
    return out
S=sys.argv[1:]; P={s:page(s) for s in S}
keys=sorted(set.intersection(*[set(p) for p in P.values()]))
works=sorted({k[0] for k in keys}); byw={w:[k for k in keys if k[0]==w] for w in works}
A={s:np.array([[sum(P[s][k][0] for k in byw[w]),sum(P[s][k][1] for k in byw[w])] for w in works]) for s in S}
rng=np.random.default_rng(20261103); idx=rng.integers(0,len(works),(2000,len(works)))
for s in S: print(s,f"{A[s][:,0].sum()/A[s][:,1].sum()*100:.1f}%")
for a,b in zip(S,S[1:]):
    d=lambda ix:(A[a][ix,0].sum()-A[b][ix,0].sum())/A[a][ix,1].sum()*100
    bs=np.array([d(i) for i in idx]); print(f"{a} − {b}: {d(np.arange(len(works))):+.1f} [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]")
