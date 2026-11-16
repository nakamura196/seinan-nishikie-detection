"""図 3: 難しい面 3 つに、正解・NDL・学習モデルの枠を重ねる。
左が NDL、右が主設定の学習モデル（片でも学習＋2×2 で隙間を埋める、0.15。paper_tables.py が書き出す。その面を学習に使っていない分割のモデル）。
正解は赤の枠、拾えなかった正解（覆い<0.5）は赤く塗る。検出は青、覆いに数えなかった大きな枠（正解の占める割合<0.2）は青の点線。
使い方: python fig3.py <出力.jpg>"""
import sys,json,csv,collections
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
K=8; FILL=0.2
PICK=[("2014-005-15","1","crowded battle scene"),("2015-017-16","1","strongly coloured multi-panel sheet"),("2012-06-03","1","grid of warrior portraits")]
SYS=[("../data/ndlkoten_det.jsonl","../data/ndlkoten_cov_fill0.2.tsv","Baseline (NDL Kotenseki OCR-Lite)"),
     ("../data/main_cv_all.jsonl","../data/main_cv_cov_fill0.2.tsv","Trained detector, tiled (5-fold CV)")]
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): rows[(r["item"],r["ci"] if "ci" in r else r["canvas"].split("/p")[-1])].append(r)
def dets(p):
    return {(d["item"],str(d["ci"])):d for d in map(json.loads,open(p))}
def covs(p):
    return {r["id"]:float(r["cov"]) for r in csv.DictReader(open(p),delimiter="\t")}
S=[(dets(j),covs(c),t) for j,c,t in SYS]
fig,ax=plt.subplots(len(PICK),2,figsize=(10,5.2*len(PICK)))
for i,(item,ci,kind) in enumerate(PICK):
    im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB")
    for j,(D,C,title) in enumerate(S):
        d=D[(item,ci)]; s=d["scale"]; a=ax[i][j]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
        G=np.zeros((d["H"]//K+1,d["W"]//K+1),bool); L=rows[(item,ci)]
        for r in L:
            x,y,w,h=(float(r[k])/K for k in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for b in d["boxes"]:
            m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
            big=m.size==0 or m.mean()<FILL
            a.add_patch(Rectangle((b[0]/s,b[1]/s),(b[2]-b[0])/s,(b[3]-b[1])/s,fill=False,ec=(0,.3,1),lw=0.6 if big else 0.9,ls=":" if big else "-"))
        miss=0
        for r in L:
            x,y,w,h=(float(r[k])/s for k in "xywh"); ok=C[r["id"]]>=0.5; miss+=not ok
            a.add_patch(Rectangle((x,y),w,h,fill=not ok,fc=(.9,0,0,.45),ec=(.9,0,0),lw=0.7))
        n=len(L); a.set_title(f"{title}\n{item} p{ci}: {n-miss}/{n} recovered ({(n-miss)/n:.0%})",fontsize=9)
    ax[i][0].set_ylabel(kind,fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=200,bbox_inches="tight",pad_inches=0.05); print("ok")
