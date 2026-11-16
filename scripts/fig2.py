"""図 2: 比較基準（NDL古典籍OCR-Lite）が取りこぼす 3 種類の画像の例。図 3 とは別の面を使う。
正解は赤の枠、拾えなかった正解（覆い<0.5）は赤く塗る。検出は青、覆いに数えなかった大きな枠（正解の占める割合<0.2）は青の点線。
各面は作品の部分に切り抜く（色票・物差しを除く）。座標は data/img の画像（長辺 2560）での画素。
使い方: python fig2.py <出力.jpg>"""
import sys,json,csv,collections
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
K=8; FILL=0.2
PICK=[("1990-032-206","2",(380,0,2240,1300),"(a) crowded scene with many small labels"),
      ("2015-017-24","1",(520,170,1490,2420),"(b) grid of warrior portraits"),
      ("2015-017-19","1",(150,600,2440,1680),"(c) dark background, no box at all")]
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): rows[(r["item"],r["ci"])].append(r)
D={(d["item"],str(d["ci"])):d for d in map(json.loads,open("../data/ndlkoten_det.jsonl"))}
C={r["id"]:float(r["cov"]) for r in csv.DictReader(open("../data/ndlkoten_cov_fill0.2.tsv"),delimiter="\t")}
fig=plt.figure(figsize=(10,8.4))
gs=fig.add_gridspec(2,2,width_ratios=[1,0.5],height_ratios=[1300/1860,1080/2290])
axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[:,1]),fig.add_subplot(gs[1,0])]
for a,(item,ci,(x0,y0,x1,y1),kind) in zip(axes,PICK):
    im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB")
    d=D[(item,ci)]; s=d["scale"]; L=rows[(item,ci)]
    a.imshow(im); a.set_xlim(x0,x1); a.set_ylim(y1,y0); a.set_xticks([]); a.set_yticks([])
    G=np.zeros((d["H"]//K+1,d["W"]//K+1),bool)
    for r in L:
        x,y,w,h=(float(r[k])/K for k in "xywh"); G[max(int(y),0):int(y+h)+1,max(int(x),0):int(x+w)+1]=True
    for b in d["boxes"]:
        m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
        big=m.size==0 or m.mean()<FILL
        a.add_patch(Rectangle((b[0]/s,b[1]/s),(b[2]-b[0])/s,(b[3]-b[1])/s,fill=False,ec=(0,.3,1),lw=0.5 if big else 0.7,ls=":" if big else "-"))
    miss=0
    for r in L:
        x,y,w,h=(float(r[k])/s for k in "xywh"); ok=C[r["id"]]>=0.5; miss+=not ok
        a.add_patch(Rectangle((x,y),w,h,fill=not ok,fc=(.9,0,0,.45),ec=(.9,0,0),lw=0.6))
    n=len(L); a.set_title(f"{kind}\n{item} p{ci}: {n-miss}/{n} recovered ({(n-miss)/n:.0%})",fontsize=9)
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=200,bbox_inches="tight",pad_inches=0.05); print("ok")
