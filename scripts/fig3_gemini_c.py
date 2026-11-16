"""図 3（2026-11-03、3 者比較の版）: Gemini 4 と C の落とす所の違いを 2 面で示す。左が Gemini 4、右が C（その面を学習に使っていない分割のモデル）。
上: 地の暗い面（2015-017-19 p1、C は枠を 1 つも返さない）。下: 小さな文字が密に並ぶ面（1990-032-206 p1、密な部分を拡大）。
正解は赤の枠、拾えなかった正解（覆い<0.5）は赤く塗る。検出は青、覆いに数えなかった大きな枠（正解の占める割合<0.2）は青の点線。尺度は vlm_subset.py と同じ。
使い方: python3 fig3_gemini_c.py <出力.jpg>"""
import sys,json,csv,collections
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
K=8; FILL=0.2
# (資料, 面, 説明, 拡大の範囲 [x0,y0,x1,y1] を画像の幅・高さに対する割合で。None は全体)
PICK=[("2015-017-19","1","dark background",[0.04,0.33,0.97,0.88]),("1990-032-206","1","small, densely packed text",[0.25,0.5,0.88,1.0])]
SYS=[("../data/gemini4_det.jsonl","Gemini 4 (no training)"),("../data/main_cv_all.jsonl","C: YOLO11s trained on annotations (5-fold CV)")]
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): rows[(r["item"],r["ci"])].append(r)
SC={(d["item"],str(d["ci"])):d["scale"] for d in map(json.loads,open("../data/main_cv_all.jsonl"))}
D=[{(d["item"],str(d["ci"])):d for d in map(json.loads,open(p))} for p,_ in SYS]
fig,ax=plt.subplots(len(PICK),2,figsize=(10,3.0*len(PICK)))
for i,(item,ci,kind,zoom) in enumerate(PICK):
    im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB"); s=SC[(item,ci)]; L=rows[(item,ci)]
    for j,(_,title) in enumerate(SYS):
        d=D[j][(item,ci)]; a=ax[i][j]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
        G=np.zeros((d["H"]//K+1,d["W"]//K+1),bool); P=np.zeros_like(G)
        for r in L:
            x,y,w,h=(float(r[k])/K for k in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for b in d["boxes"]:
            m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]; big=m.size==0 or m.mean()<FILL
            if not big: P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
            a.add_patch(Rectangle((b[0]/s,b[1]/s),(b[2]-b[0])/s,(b[3]-b[1])/s,fill=False,ec=(0,.3,1),lw=0.6 if big else 1.0,ls=":" if big else "-"))
        miss=0
        for r in L:
            x,y,w,h=(float(r[k])/K for k in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]; ok=m.size and m.mean()>=0.5; miss+=not ok
            a.add_patch(Rectangle((float(r["x"])/s,float(r["y"])/s),float(r["w"])/s,float(r["h"])/s,fill=not ok,fc=(.9,0,0,.45),ec=(.9,0,0),lw=0.7))
        if zoom:
            W,H=im.size; a.set_xlim(zoom[0]*W,zoom[2]*W); a.set_ylim(zoom[3]*H,zoom[1]*H)
        n=len(L); a.set_title(f"{title}\n{item} p{ci}: {n-miss}/{n} recovered",fontsize=9)
    ax[i][0].set_ylabel(kind,fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=200,bbox_inches="tight",pad_inches=0.05); print("ok")
