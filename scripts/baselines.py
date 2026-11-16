"""比較基準を足した表: NDL Lite / NDL 通常版 / Surya / Gemma 4 / 主設定 C を、paper_tables.py と同じ尺度で並べる。
尺度: 1/8 の格子、覆い 0.5 以上で拾えた、正解の文字が 20% 未満の枠は覆いに数えない、正解との重なり 10% 未満の枠を「外」。
いずれも全 739 面。C は ../data/main_cv_all.jsonl（paper_tables.py が書く、閾値 0.15 適用済み）。
資料単位のブートストラップ（2,000 回、seed 20261101）で差の 95% 区間も出す。
NDL 通常版は 100 枠の上限に届いた 6 面を上限 1000 で流し直したもの（run_ndlkoten_full_cap.py）。
使い方: python3 baselines.py"""
import json,csv,collections
import numpy as np
K=8; FILL=0.2
R={r["id"]:r for r in csv.DictReader(open(__import__("os").environ.get("ROWS","../data/rows.tsv")),delimiter="\t")}; ids=sorted(R)
byimg=collections.defaultdict(list)
for i in ids: byimg[(R[i]["item"],str(R[i]["ci"]))].append(R[i])
def evaluate(path,th):
    cov={}; nb=0; out=0
    for d in map(json.loads,open(path)):
        L=byimg.get((d["item"],str(d["ci"])))
        if not L: continue
        W,H=d["W"]//K+1,d["H"]//K+1; G=np.zeros((H,W),bool); P=np.zeros_like(G)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for b in d["boxes"]:
            if len(b)>4 and b[4]<th: continue
            m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]; nb+=1
            if m.size==0 or m.mean()<0.1: out+=1
            if m.size==0 or m.mean()<FILL: continue
            P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
            cov[r["id"]]=float(m.mean()) if m.size else 0.0
    assert len(cov)==len(ids),(path,len(cov))
    return cov,nb,out
S={"NDL Lite":("../data/ndlkoten_det.jsonl",0.3),"C":("../data/main_cv_all.jsonl",0)}
import os,sys
for th in [float(a) for a in sys.argv[1:]] or [0.0,0.3,0.5]:
    if sum(1 for _ in open("../data/ndlkoten_full_det_cap1000.jsonl"))>=739: S[f"NDL full {th}"]=("../data/ndlkoten_full_det_cap1000.jsonl",th)
    if os.path.exists("../data/surya_det.jsonl"): S[f"Surya {th}"]=("../data/surya_det.jsonl",th)
GM=os.environ.get("GEMMA","../data/gemma4_det.jsonl")
if os.path.exists(GM) and sum(1 for _ in open(GM))>=739: S["Gemma 4"]=(GM,0)
res={n:evaluate(*a) for n,a in S.items()}; cov={n:v[0] for n,v in res.items()}
pc=lambda n,v,t=0.5:100*np.mean([cov[n][i]>=t for i in v])
def L(t): n=len(t); return "1" if n==1 else "2-3" if n<=3 else "4-10" if n<=10 else "11-30" if n<=30 else "31+"
base=cov["NDL Lite"]; im=lambda i:(R[i]["item"],str(R[i]["ci"]))
bi=collections.defaultdict(list)
for i in ids: bi[im(i)].append(i)
hard6=[i for v in bi.values() if len(v)>=2 and np.mean([base[x]>=.5 for x in v])<.5 for i in v]
top30=[i for k in sorted(bi,key=lambda k:-sum(base[x]<.5 for x in bi[k]))[:30] for i in bi[k]]
G=[("全体",ids),*[(c,[i for i in ids if R[i]["pred"]==c]) for c in ("詞書","札","札?")],
   *[(f"{b}字",[i for i in ids if L(R[i]["text"])==b]) for b in ("1","2-3","4-10","11-30","31+")],
   ("難6面",hard6),("上位30面",top30),("格子",bi[("2012-06-03","1")])]
print("系 | "+" | ".join(f"{g}(n={len(v)})" for g,v in G)+" | ≥0.9 | 覆い0 | 取りこぼし | 枠 | 外")
for n in S:
    print(n,"|"," | ".join(f"{pc(n,v):.1f}" for g,v in G),f"| {pc(n,ids,0.9):.1f} | {100*np.mean([cov[n][i]==0 for i in ids]):.1f} | {sum(cov[n][i]<.5 for i in ids)} | {res[n][1]} | {res[n][2]}")
print("== 種類別 ≥0.9 / 覆い0")
for n in S: print(n,*[f"{g} {pc(n,v,0.9):.1f}/{100*np.mean([cov[n][i]==0 for i in v]):.1f}" for g,v in G[:4]])
item={i:R[i]["item"] for i in ids}; byi=collections.defaultdict(list)
for i in ids: byi[item[i]].append(i)
its=sorted(byi); hit={n:{it:np.array([cov[n][i]>=0.5 for i in byi[it]]) for it in its} for n in S}
rng=np.random.default_rng(20261101); bs=collections.defaultdict(list)
for _ in range(2000):
    s=rng.choice(its,len(its)); tot=sum(len(byi[it]) for it in s); v={n:sum(hit[n][it].sum() for it in s)/tot for n in S}
    for n in S:
        bs[n].append(v[n])
        if n!="NDL Lite": bs[f"{n} − NDL Lite"].append(v[n]-v["NDL Lite"])
        if n not in("C","NDL Lite"): bs[f"C − {n}"].append(v["C"]-v[n])
print("== 資料単位のブートストラップ 2,000 回（平均 [95% 区間]）")
for n,v in bs.items(): print(f"  {n}: {100*np.mean(v):.1f} [{100*np.percentile(v,2.5):.1f}, {100*np.percentile(v,97.5):.1f}]")
