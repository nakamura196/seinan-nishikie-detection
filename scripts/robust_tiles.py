"""査読向けの頑健さの確認。NDL・前のモデル（全体 0.25）・片でも学習したモデル（全体+2×2 0.15）について、
(1) 資料単位のブートストラップ（2,000 回、seed 20261101）で拾えた割合の 95% 区間と、モデルと NDL の差の区間を出す
(2) 「拾えた」の線（覆い 0.3 / 0.5 / 0.7）を変えたときの値を出す
使い方: python3 robust_tiles.py"""
import sys,json,csv,collections
sys.argv=["x"]; import numpy as np
exec(open("sweep_tiles.py").read().split("acc=collections")[0])
base={r["id"]:float(r["cov"]) for r in csv.DictReader(open("../data/ndlkoten_cov_fill0.2.tsv"),delimiter="\t")}
CF={"前 full@0.25":("",["full"],0.25),"片学習 full+t2@0.15":("t2train_",["full","t2"],0.15)}
cov={n:{} for n in CF}; item={}
for n,(tag,parts,th) in CF.items():
    for k in range(5):
        for d in map(json.loads,open(f"../data/tiles/{tag}fold{k}.jsonl")):
            if d["role"]!="test": continue
            L=rows[(d["item"],d["ci"])]; W,H=d["W"]//K+1,d["H"]//K+1; G=np.zeros((H,W),bool)
            for r in L:
                x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
            P=np.zeros_like(G)
            for b in nms([b for p in parts for b in d[p]]):
                if b[4]<th: continue
                m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
                if m.size==0 or m.mean()<FILL: continue
                P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
            for r in L:
                x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
                cov[n][r["id"]]=float(m.mean()) if m.size else 0.0; item[r["id"]]=r["item"]
cov={"NDL":base,**cov}; ids=sorted(item)
print("覆いの線を変えたとき（拾えた割合）")
for t in (0.3,0.5,0.7,0.9): print(f"  {t}",*[f"{n} {np.mean([cov[n][i]>=t for i in ids]):.3f}" for n in cov])
byi=collections.defaultdict(list)
for i in ids: byi[item[i]].append(i)
its=sorted(byi); hit={n:{it:np.array([cov[n][i]>=0.5 for i in byi[it]]) for it in its} for n in cov}
rng=np.random.default_rng(20261101); B=2000; res=collections.defaultdict(list)
for _ in range(B):
    s=rng.choice(its,len(its))
    tot=sum(len(byi[it]) for it in s); v={n:sum(hit[n][it].sum() for it in s)/tot for n in cov}
    for n in cov: res[n].append(v[n])
    for n in list(CF): res[n+" − NDL"].append(v[n]-v["NDL"])
    res["片学習 − 前"].append(v["片学習 full+t2@0.15"]-v["前 full@0.25"])
print(f"資料 {len(its)} 件、範囲 {len(ids)} 件でブートストラップ {B} 回（95% 区間）")
for n,v in res.items(): print(f"  {n}: {np.mean(v):.3f} [{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]")
