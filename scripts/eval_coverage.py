"""正解の範囲が、検出の範囲でどれだけ覆われたかを測る（区切り方の違いに左右されない物差し）。
各面を 1/8 に縮めた格子で塗り、正解の範囲ごとに「検出に覆われた面積の割合」を出す。
覆われた割合 >= 0.5 を「拾えた」とする。あわせて、正解にかからない検出（付け漏れか誤検出）も数える。
--fill X を付けると、枠の面積のうち正解の範囲が占める割合が X 未満の検出（中身がほぼ絵の大きな枠）を覆いに数えない。
使い方: python eval_coverage.py <検出の jsonl> [出力 tsv] [--fill X]"""
import sys,json,csv,collections
import numpy as np
K=8
args=sys.argv[1:]
FILL=0.0
if "--fill" in args:
    i=args.index("--fill"); FILL=float(args[i+1]); del args[i:i+2]
sys.argv=[sys.argv[0]]+args
det={}
for l in open(sys.argv[1]):
    d=json.loads(l); det[d["canvas"]]=d
rows=list(csv.DictReader(open(__import__("os").environ.get("ROWS","../data/rows.tsv")),delimiter="\t"))
byc=collections.defaultdict(list)
for r in rows: byc[r["canvas"]].append(r)
out=[]; orphan=0; ndet=0; dropped=0
for c,L in byc.items():
    if c not in det: continue
    d=det[c]; W,H=d["W"]//K+1,d["H"]//K+1
    P=np.zeros((H,W),bool); G=np.zeros((H,W),bool)
    for r in L:
        x,y,w,h=(float(r[k])/K for k in "xywh")
        G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
    for b in d["boxes"]:
        s=(slice(int(b[1]/K),int(b[3]/K)+1),slice(int(b[0]/K),int(b[2]/K)+1))
        if FILL>0 and (G[s].size==0 or G[s].mean()<FILL): dropped+=1; continue
        P[s]=True
    for r in L:
        x,y,w,h=(float(r[k])/K for k in "xywh")
        m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
        out.append(dict(id=r["id"],canvas=c,item=r["item"],pred=r["pred"],text=r["text"],cov=round(float(m.mean()) if m.size else 0,3)))
    for b in d["boxes"]:
        ndet+=1
        m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
        if m.size==0 or m.mean()<0.1: orphan+=1
cov=np.array([o["cov"] for o in out])
print(f"正解 {len(out)} 件 / 面 {len([c for c in byc if c in det])}")
print(f"拾えた(覆い>=0.5) {np.mean(cov>=0.5):.3f}  ほぼ全部(>=0.9) {np.mean(cov>=0.9):.3f}  全く無し(=0) {np.mean(cov==0):.3f}  平均覆い {cov.mean():.3f}")
for k in ["詞書","札","札?"]:
    v=np.array([o["cov"] for o in out if o["pred"]==k]); print(f"  {k}: n={len(v)} 拾えた {np.mean(v>=0.5):.3f} 全く無し {np.mean(v==0):.3f}")
if FILL>0: print(f"覆いに数えなかった検出（正解の占める割合<{FILL}） {dropped} 件")
print(f"検出 {ndet} 件のうち、正解にほぼかからない（重なり<10%） {orphan} 件")
if len(sys.argv)>2:
    with open(sys.argv[2],"w") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()),delimiter="\t"); w.writeheader(); w.writerows(out)
