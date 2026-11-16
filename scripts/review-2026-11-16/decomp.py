import json,csv,collections,pickle,os,numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
D="../data";K=8
R={r["id"]:r for r in csv.DictReader(open(f"{D}/rows.tsv"),delimiter="\t")}
byimg=collections.defaultdict(list)
for r in R.values(): byimg[(r["item"],str(r["ci"]))].append(r)
def evaluate(path,th=0,FILL=0.2):
    cov={}
    for d in map(json.loads,open(path)):
        k=(d["item"],str(d["ci"])); L=byimg[k]; W,H=d["W"]//K+1,d["H"]//K+1; Gr=np.zeros((H,W),bool); P=np.zeros_like(Gr)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); Gr[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for b in d["boxes"]:
            if len(b)>4 and b[4]<th: continue
            m=Gr[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
            if m.size==0 or m.mean()<FILL: continue
            P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
            cov[r["id"]]=float(m.mean()) if m.size else 0.0
    return cov
G=evaluate(f"{D}/gemini4_det.jsonl"); C=evaluate(f"{D}/main_cv_all.jsonl"); Lt=evaluate(f"{D}/ndlkoten_det.jsonl",0.3)
ids=list(R); n=len(ids)
ok=lambda cv,t=.5:{i:cv[i]>=t for i in ids}
g,c,l=ok(G),ok(C),ok(Lt)
print("recall G C L",*[round(100*np.mean(list(x.values())),2) for x in (g,c,l)])
fres=pickle.load(open("review-2026-11-16/fill_res.pkl","rb"))
cb=fres[(0.05,0.7)]; print("Cbest",100*np.mean([cb[i] for i in ids]))
# per print diff
items=sorted({R[i]["item"] for i in ids}); byit=collections.defaultdict(list)
for i in ids: byit[R[i]["item"]].append(i)
def contrib(a,b,name):
    d={it:sum(a[i]-b[i] for i in byit[it]) for it in items}; tot=sum(d.values())
    pos=sorted(d.items(),key=lambda x:-abs(x[1]))
    print(f"== {name}: total diff regions {tot} ({100*tot/n:.2f} pt); top by |diff|:",[(k,v,len(byit[k])) for k,v in pos[:8]])
    s=sorted(d.values(),reverse=True); print("  sum positive",sum(v for v in s if v>0),"sum negative",sum(v for v in s if v<0))
    return d
dGC=contrib(g,c,"Gemini-C")
dGL=contrib(g,l,"Gemini-Lite"); dCL=contrib(c,l,"C-Lite")
# share of gain over Lite from top30 images of Lite misses
base={i:Lt[i] for i in ids}
top30=sorted(byimg,key=lambda k:-sum(Lt[r["id"]]<.5 for r in byimg[k]))[:30]
t30=set(r["id"] for k in top30 for r in byimg[k])
for nm,a in (("G",g),("C",c)):
    gain=sum(a[i]-l[i] for i in ids); gt=sum(a[i]-l[i] for i in t30)
    rest=[i for i in ids if i not in t30]
    print(nm,"gain over Lite",gain,"from top30",gt,f"{100*gt/gain:.1f}%","; outside top30 recall",round(100*np.mean([a[i] for i in rest]),2),"Lite",round(100*np.mean([l[i] for i in rest]),2), "n rest",len(rest))
# Gemini-C without the 2 dense prints
dense={"1990-032-206","2013-24-1"}
rest=[i for i in ids if R[i]["item"] not in dense]
print("without 2 dense prints: G",round(100*np.mean([g[i] for i in rest]),2),"C",round(100*np.mean([c[i] for i in rest]),2),len(rest))
# 1-char with Cbest
one=[i for i in ids if len(R[i]["text"])==1]
print("1char G",100*np.mean([g[i] for i in one]),"C",100*np.mean([c[i] for i in one]),"Cbest",100*np.mean([cb[i] for i in one]))
# bootstrap G - Cbest
rng=np.random.default_rng(1)
def diff(a,b,its): v=[i for it in its for i in byit[it]]; return 100*(np.mean([a[i] for i in v])-np.mean([b[i] for i in v]))
bs=[diff(g,cb,rng.choice(items,len(items))) for _ in range(2000)]
print("G - Cbest",diff(g,cb,items),np.percentile(bs,[2.5,97.5]))
its1=sorted({R[i]["item"] for i in one}); by1=collections.defaultdict(list)
for i in one: by1[R[i]["item"]].append(i)
def d1(its): v=[i for it in its for i in by1[it]]; return 100*(np.mean([g[i] for i in v])-np.mean([cb[i] for i in v]))
bs=[d1(rng.choice(its1,len(its1))) for _ in range(2000)]; print("1char G - Cbest",d1(its1),np.percentile(bs,[2.5,97.5]))
# measure sensitivity: coverage threshold, fill
for t in (0.3,0.5,0.7,0.9):
    print("thr",t,"G",round(100*np.mean([G[i]>=t for i in ids]),1),"C",round(100*np.mean([C[i]>=t for i in ids]),1),"L",round(100*np.mean([Lt[i]>=t for i in ids]),1))
for F in (0.0,0.1,0.3):
    G2=evaluate(f"{D}/gemini4_det.jsonl",0,F); C2=evaluate(f"{D}/main_cv_all.jsonl",0,F)
    print("fill",F,"G",round(100*np.mean([G2[i]>=.5 for i in ids]),2),"C",round(100*np.mean([C2[i]>=.5 for i in ids]),2))
# Claude on 90: images per subagent effect? skip
# per-image: how many images contain the regions where G>C
