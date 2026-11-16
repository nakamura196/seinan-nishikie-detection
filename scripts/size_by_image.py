"""作業 6（2026-11-03）: 小さく詰まった字で C が Gemini 4 より上、が少数の面の偏りでないかを面単位で確かめる。
尺度は vlm_subset.py と同じ（1/8 の格子、覆い 0.5 以上で拾えた、正解の文字 20% 未満の枠は数えない）。
範囲の短い辺 ÷ 画像の長い辺で分け、各帯で (1) 範囲単位の拾えた割合、(2) 面ごとの差の向き（C が上の面・Gemini が上の面・同じ）、
(3) 面ごとの割合を平均した値、(4) 作品単位のブートストラップ 2,000 回の差の区間、(5) 偏りの 4 面を除いた値を出す。
使い方: python3 size_by_image.py"""
import json,csv,collections
import numpy as np
K=8; FILL=0.2; D="../data"
R={r["id"]:r for r in csv.DictReader(open(f"{D}/rows.tsv"),delimiter="\t")}
byimg=collections.defaultdict(list)
for r in R.values(): byimg[(r["item"],str(r["ci"]))].append(r)
def evaluate(path):
    cov={}; size={}
    for d in map(json.loads,open(path)):
        k=(d["item"],str(d["ci"])); L=byimg[k]
        W,H=d["W"]//K+1,d["H"]//K+1; Gr=np.zeros((H,W),bool); P=np.zeros_like(Gr)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); Gr[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for b in d["boxes"]:
            m=Gr[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
            if m.size==0 or m.mean()<FILL: continue
            P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
            cov[r["id"]]=float(m.mean()) if m.size else 0.0
            size[r["id"]]=100*min(float(r["w"]),float(r["h"]))/max(d["W"],d["H"])
    return cov,size
cg,size=evaluate(f"{D}/gemini4_det.jsonl"); cc,_=evaluate(f"{D}/main_cv_all.jsonl")
ids=[i for i in cg if i in cc]
img=lambda i:(R[i]["item"],str(R[i]["ci"]))
BINS=[(0,0.5),(0.5,0.8),(0.8,1.2),(1.2,2),(2,100)]
FOUR={("1990-032-206","1"),("1990-032-206","2"),("2013-24-1","1"),("2013-24-1","2")}
rng=np.random.default_rng(20261103)
print(f"範囲 {len(ids)}、面 {len({img(i) for i in ids})}")
for lo,hi in BINS:
    v=[i for i in ids if lo<=size[i]<hi]
    g=np.array([cg[i]>=.5 for i in v]); c=np.array([cc[i]>=.5 for i in v])
    per=collections.defaultdict(list)
    for i,a,b in zip(v,g,c): per[img(i)].append((a,b))
    up=sum(np.mean([b for a,b in x])>np.mean([a for a,b in x]) for x in per.values())
    dn=sum(np.mean([b for a,b in x])<np.mean([a for a,b in x]) for x in per.values())
    mg=100*np.mean([np.mean([a for a,b in x]) for x in per.values()]); mc=100*np.mean([np.mean([b for a,b in x]) for x in per.values()])
    v4=[k for k,i in enumerate(v) if img(i) not in FOUR]
    items=sorted({R[i]["item"] for i in v}); byit=collections.defaultdict(list)
    for k,i in enumerate(v): byit[R[i]["item"]].append(k)
    def diff(its):
        ix=[k for it in its for k in byit[it]]; return 100*(c[ix].mean()-g[ix].mean())
    bs=[diff(rng.choice(items,len(items))) for _ in range(2000)]
    print(f"[{lo},{hi}) 範囲 {len(v)} 面 {len(per)} 作品 {len(items)} | 範囲単位 G {100*g.mean():.1f} C {100*c.mean():.1f} 差C−G {diff(items):.1f} [{np.percentile(bs,2.5):.1f},{np.percentile(bs,97.5):.1f}]"
          f" | 面平均 G {mg:.1f} C {mc:.1f} | C上 {up} G上 {dn} 同 {len(per)-up-dn}"
          f" | 4面除く 範囲 {len(v4)} G {100*g[v4].mean():.1f} C {100*c[v4].mean():.1f}"
          f" | Cのみ {int((c&~g).sum())}（4面 {sum(1 for k,i in enumerate(v) if c[k] and not g[k] and img(i) in FOUR)}） Gのみ {int((g&~c).sum())}")
# 1 字の範囲でも同じ見方をする（Gemini が上、が少数の面の偏りでないか）
v=[i for i in ids if len(R[i]["text"])==1]
g=np.array([cg[i]>=.5 for i in v]); c=np.array([cc[i]>=.5 for i in v])
per=collections.defaultdict(list)
for i,a,b in zip(v,g,c): per[img(i)].append((a,b))
up=sum(np.mean([b for a,b in x])>np.mean([a for a,b in x]) for x in per.values()); dn=sum(np.mean([b for a,b in x])<np.mean([a for a,b in x]) for x in per.values())
items=sorted({R[i]["item"] for i in v}); byit=collections.defaultdict(list)
for k,i in enumerate(v): byit[R[i]["item"]].append(k)
def diff(its):
    ix=[k for it in its for k in byit[it]]; return 100*(g[ix].mean()-c[ix].mean())
bs=[diff(rng.choice(items,len(items))) for _ in range(2000)]
top=collections.Counter(img(i) for k,i in enumerate(v) if g[k] and not c[k]).most_common(4)
print(f"1字 範囲 {len(v)} 面 {len(per)} 作品 {len(items)} | G {100*g.mean():.1f} C {100*c.mean():.1f} 差G−C {diff(items):.1f} [{np.percentile(bs,2.5):.1f},{np.percentile(bs,97.5):.1f}] | G上 {dn} C上 {up} | Gのみ {int((g&~c).sum())} 上位4面 {sum(n for _,n in top)} Cのみ {int((c&~g).sum())}")
