"""図 1: 注釈の外の検出の例。正解（赤）に重ならない学習モデルの枠（橙）が、余白や出版の枠の中の実在する文字（届の日付・画工・出版人）の上にある。
学習モデルは主設定（paper_tables.py が書き出す main_cv_all.jsonl。その面を学習に使っていない分割のモデル）。
正解は赤の枠。注釈の外の枠（正解の占める割合 <0.1）は橙、それ以外の検出は青。
使い方: python fig1.py <出力.jpg>"""
import sys,json,csv,collections
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
K=8; OUT=0.1
PICK=[("1990-032-020","3",(60,250,760,740),"")]
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): rows[(r["item"],r["ci"])].append(r)
D={(d["item"],str(d["ci"])):d for d in map(json.loads,open("../data/main_cv_all.jsonl"))}
fig,ax=plt.subplots(figsize=(6,4.3)); axes=[ax]
for a,(item,ci,(x0,y0,x1,y1),kind) in zip(axes,PICK):
    im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB")
    d=D[(item,ci)]; s=d["scale"]; L=rows[(item,ci)]
    a.imshow(im); a.set_xlim(x0,x1); a.set_ylim(y1,y0); a.set_xticks([]); a.set_yticks([])
    G=np.zeros((d["H"]//K+1,d["W"]//K+1),bool)
    for r in L:
        x,y,w,h=(float(r[k])/K for k in "xywh"); G[max(int(y),0):int(y+h)+1,max(int(x),0):int(x+w)+1]=True
    n=0
    for b in d["boxes"]:
        m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
        out=m.size==0 or m.mean()<OUT; n+=out
        a.add_patch(Rectangle((b[0]/s,b[1]/s),(b[2]-b[0])/s,(b[3]-b[1])/s,fill=False,ec=(1,.55,0) if out else (0,.3,1),lw=1.6 if out else 0.8))
    for r in L:
        x,y,w,h=(float(r[k])/s for k in "xywh")
        a.add_patch(Rectangle((x,y),w,h,fill=False,ec=(.9,0,0),lw=0.8,ls="--"))
    a.set_title(f"{item} p{ci}",fontsize=9); print(item,ci,"outside boxes on image:",n)
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=200,bbox_inches="tight",pad_inches=0.05); print("ok")
