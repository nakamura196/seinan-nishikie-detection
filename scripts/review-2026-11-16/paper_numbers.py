"""Numbers quoted in the DH2027 paper that are not printed by paper_tables.py / vlm_subset.py / eval_read.py (2026-11-16).
All detection intervals use one rule: resample prints with replacement, 2,000 replicates, a fresh generator seeded 20261116 for each interval.
Measure as in vlm_subset.py (1/8 grid, coverage >= 0.5, boxes with < 20% annotated text ignored, < 10% overlap = outside).
Usage: python3 paper_numbers.py   (reads ../data/rows.tsv; run from anywhere)"""
import os,sys,json,csv,collections,re
import numpy as np
from PIL import Image
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
D="../data"; K=8; SEED=20261116
R={r["id"]:r for r in csv.DictReader(open(f"{D}/rows.tsv"),delimiter="\t")}; ids=list(R)
byimg=collections.defaultdict(list)
for r in R.values(): byimg[(r["item"],str(r["ci"]))].append(r)
def grid(L,W,H):
    G=np.zeros((H//K+1,W//K+1),bool)
    for r in L:
        x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
    return G
def score(L,G,boxes,FILL=0.2):
    P=np.zeros_like(G); nb=no=0; cov={}
    for b in boxes:
        m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]; nb+=1
        if m.size==0 or m.mean()<0.1: no+=1
        if m.size==0 or m.mean()<FILL: continue
        P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
    for r in L:
        x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
        cov[r["id"]]=float(m.mean()) if m.size else 0.0
    return cov,nb,no
def evaluate(path,th=0,FILL=0.2,replace=None):
    cov={}; nb=no=0; dets={(d["item"],str(d["ci"])):d for d in map(json.loads,open(path))}
    if replace: dets.update(replace)
    for k,d in dets.items():
        L=byimg[k]; c,a,b=score(L,grid(L,d["W"],d["H"]),[b for b in d["boxes"] if not (len(b)>4 and b[4]<th)],FILL); cov.update(c); nb+=a; no+=b
    return cov,nb,no
# C's upper bound: same tiles and fill-in as fillsweep.py, threshold 0.05, discard at 70% overlap
src=open("sweep_tiles.py").read(); exec(src[src.index("def nms"):src.index("acc=collections")])
def fillin(full,t2,ov=0.5):
    ar=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1])); out=list(full)
    for b in nms(t2):
        if all(max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))<ov*ar(b) for o in full): out.append(b)
    return out
Cb={}; nbCb=noCb=0
for k in range(5):
    for d in map(json.loads,open(f"{D}/tiles/t2train_fold{k}.jsonl")):
        if d["role"]!="test": continue
        L=byimg[(d["item"],str(d["ci"]))]
        bs=fillin(nms([b for b in d["full"] if b[4]>=0.05]),[b for b in d["t2"] if b[4]>=0.05],0.7)
        c,a,b=score(L,grid(L,d["W"],d["H"]),bs); Cb.update(c); nbCb+=a; noCb+=b
G,nbG,noG=evaluate(f"{D}/gemini4_det.jsonl"); C,nbC,noC=evaluate(f"{D}/main_cv_all.jsonl"); Lt,nbL,noL=evaluate(f"{D}/ndlkoten_det.jsonl",0.3)
ok=lambda cv,t=.5:{i:cv[i]>=t for i in ids}
g,c,l,cb=ok(G),ok(C),ok(Lt),ok(Cb)
pc=lambda a,v:round(100*np.mean([a[i] for i in v]),1)
item=lambda i:R[i]["item"]
def boot(a,b,sel):
    by=collections.defaultdict(list)
    for i in sel: by[item(i)].append(i)
    its=sorted(by); A={it:(sum(a[i] for i in by[it]),sum(b[i] for i in by[it]),len(by[it])) for it in its}
    f=lambda S:100*(sum(A[s][0] for s in S)-sum(A[s][1] for s in S))/sum(A[s][2] for s in S)
    rng=np.random.default_rng(SEED); bs=[f(rng.choice(its,len(its))) for _ in range(2000)]
    return f"{f(its):+.1f} [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}] (regions {len(sel)}, prints {len(its)})"
print("== counts: regions",len(ids),"images",len(byimg),"prints",len({item(i) for i in ids}))
per_img=[len(v) for v in byimg.values()]; nch=[len("".join(R[i]["text"].split())) for i in ids]
print("   regions per image median",np.median(per_img),"; characters",sum(nch),"median per region",np.median(nch))
print("   by annotator",sorted(collections.Counter(R[i]["user"] for i in ids).values(),reverse=True))  # public copy: names removed
print("   labels",collections.Counter(R[i]["pred"] for i in ids))
print("== Table 2 (Lite | Gemini 4 | C | C upper)")
one=[i for i in ids if len(R[i]["text"])==1]; two=[i for i in ids if 2<=len(R[i]["text"])<=3]
for nm,sel in [("all",ids),("inscriptions",[i for i in ids if R[i]["pred"]=="詞書"]),("labels",[i for i in ids if R[i]["pred"]=="札"]),("1 char",one),("2-3 chars",two)]:
    print(f"   {nm} n={len(sel)}",*[pc(x,sel) for x in (l,g,c,cb)])
print("   cov>=0.9",*[pc(ok(x,0.9),ids) for x in (Lt,G,C,Cb)])
print("   cov=0",*[round(100*np.mean([x[i]==0 for i in ids]),1) for x in (Lt,G,C,Cb)])
print("   boxes",nbL,nbG,nbC,nbCb,"outside",noL,noG,noC,noCb)
print("== intervals (seed",SEED,")")
print("   G - Lite",boot(g,l,ids)); print("   G - C",boot(g,c,ids)); print("   G - Cupper",boot(g,cb,ids)); print("   C - Lite",boot(c,l,ids))
dense={"1990-032-206","2013-24-1"}; rest=[i for i in ids if item(i) not in dense]
print("   G - C without 2 dense prints",boot(g,c,rest))
print("   1 char G - C",boot(g,c,one)); print("   1 char G - Cupper",boot(g,cb,one))
d={it:0 for it in {item(i) for i in ids}}
for i in ids: d[item(i)]+=g[i]-c[i]
print("   prints G>C",sum(v>0 for v in d.values()),"C>G",sum(v<0 for v in d.values()),"tie",sum(v==0 for v in d.values()))
top30=sorted(byimg,key=lambda k:-sum(Lt[r["id"]]<.5 for r in byimg[k]))[:30]; t30={r["id"] for k in top30 for r in byimg[k]}
r709=[i for i in ids if i not in t30]
print("   outside the 30 worst images:",len(byimg)-30,"images, recall Lite/G/C",pc(l,r709),pc(g,r709),pc(c,r709))
for fn in ("gemini4_det_rep2.jsonl","gemini4_det_rep3.jsonl","gemini4_read_det.jsonl"):
    x=ok(evaluate(f"{D}/{fn}")[0]); print(f"   {fn}: recall {pc(x,ids)}; - C {boot(x,c,ids)}")
print("== 5.3 overlap: both",sum(g[i] and c[i] for i in ids),"only G",sum(g[i] and not c[i] for i in ids),"only C",sum(c[i] and not g[i] for i in ids),"neither",sum(not g[i] and not c[i] for i in ids))
# size bands: short side of region / long side of image (image size W,H from the region rows)
rel={i:min(float(R[i]["w"]),float(R[i]["h"]))/max(int(R[i]["W"]),int(R[i]["H"]))*100 for i in ids}
onlyC=[i for i in ids if c[i] and not g[i]]; cntimg=collections.Counter((item(i),R[i]["ci"]) for i in onlyC)
top4={k for k,_ in cntimg.most_common(4)}
print("   only-C top 4 images",cntimg.most_common(4),"sum",sum(v for k,v in cntimg.most_common(4)),"prints",len({k[0] for k in top4}))
for lo,hi in ((0,0.5),(0.5,0.8),(0.8,1.2),(1.2,100)):
    sel=[i for i in ids if lo<=rel[i]<hi]
    print(f"   band [{lo},{hi}) n={len(sel)} images {len({(item(i),R[i]['ci']) for i in sel})} G {pc(g,sel)} C {pc(c,sel)}  C-G {boot(c,g,sel)}")
    if lo==0.5:
        s2=[i for i in sel if (item(i),R[i]["ci"]) not in top4]; print(f"      without the 4 images n={len(s2)} G {pc(g,s2)} C {pc(c,s2)}")
# dark regions: median grey < 80 inside the region
dark=[]
for k,L in byimg.items():
    im=Image.open(f"{D}/img/{k[0]}_{k[1]}.jpg").convert("L"); s=int(L[0]["W"])/im.size[0]; a=np.asarray(im)
    for r in L:
        x,y,w,h=(float(r[c])/s for c in "xywh"); m=a[int(y):int(y+h)+1,int(x):int(x+w)+1]
        if m.size and np.median(m)<80: dark.append(r["id"])
print("== dark regions",len(dark),"C",pc(c,dark),"G",pc(g,dark))
print("== figure 3 images: 2015-017-19 p1",[(g[r["id"]],c[r["id"]]) for r in byimg[("2015-017-19","1")]].count((True,False)),"of",len(byimg[("2015-017-19","1")]),
      "; 1990-032-206 p1 n",len(byimg[("1990-032-206","1")]),"G",sum(g[r["id"]] for r in byimg[("1990-032-206","1")]),"C",sum(c[r["id"]] for r in byimg[("1990-032-206","1")]))
L0,_,_=evaluate(f"{D}/ndlkoten_det.jsonl",0.3,FILL=0); print("== Lite without the 20% rule",pc(ok(L0),ids))
unc={(d["item"],str(d["ci"])):d for d in map(json.loads,open(f"{D}/lite_uncap.jsonl"))}
Lu,_,_=evaluate(f"{D}/ndlkoten_det.jsonl",0.3,replace=unc); print("== Lite without the 200-box cap",pc(ok(Lu),ids),"(images replaced",len(unc),")")
miss=[i for i in ids if not l[i]]; per=collections.Counter((item(i),R[i]["ci"]) for i in miss)
print("== Lite misses",len(miss),"on",len(per),"images; top 30 hold",round(100*sum(v for _,v in per.most_common(30))/len(miss)),"%")
kw=re.compile("画工|画|彫|版元|出版|御届|届|住|町|丁目|番地")
marg=[i for i in ids if re.search(r"画工|彫|出版|御届|丁目|番地",R[i]["text"])]
print("== marginal-publication keyword hits",len(marg))  # public copy: annotator names removed
for i in marg: print("   ",R[i]["item"])  # public copy: transcriptions removed
# Claude Opus 5.5 on the 90 images (30 with the most baseline misses, 60 at random)
O5,_,_=evaluate(f"{D}/vlm_claude-opus-5-5_det.jsonl"); sub=json.load(open(f"{D}/vlm_subset.json"))
for grp,keys in sub.items():
    keys={tuple(k) for k in keys}; sel=[r["id"] for k in keys for r in byimg[k]]; o={i:O5.get(i,0)>=.5 for i in ids}
    print(f"== Opus {grp}: images {len(keys)} regions {len(sel)} Opus {pc(o,sel)} G {pc(g,sel)} C {pc(c,sel)} Lite {pc(l,sel)}; G - Opus {boot(g,o,sel)}; Opus - Lite {boot(o,l,sel)}")
