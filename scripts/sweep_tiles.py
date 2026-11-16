"""run_yolo_tiles.py の結果から、見せ方の組み合わせ × しきい値ごとに、拾えた割合と注釈の外の枠の数を出す。
物差しは eval_coverage.py と同じ（1/8 の格子、覆い>=0.5 で拾えた、--fill 0.2）。重なった枠は IoU 0.5 で 1 つにまとめる。
検証の面（val、学習側の資料）と評価の面（test、全 739 面）を分けて出す。しきい値と組み合わせは val で選ぶ。
使い方: python3 sweep_tiles.py [TAG] → 表を出し、../data/tiles/<TAG>sweep.tsv に書く"""
import sys,json,csv,collections
import numpy as np
K=8; FILL=0.2; TAG=sys.argv[1] if len(sys.argv)>1 else ""
VARS={"full":["full"],"t2":["t2"],"t3":["t3"],"full+t2":["full","t2"],"full+t3":["full","t3"],"full+t2+t3":["full","t2","t3"]}
TH=[0.05,0.1,0.15,0.2,0.25,0.3]
rows=collections.defaultdict(list)
for r in csv.DictReader(open(__import__("os").environ.get("ROWS","../data/rows.tsv")),delimiter="\t"): rows[(r["item"],r["ci"])].append(r)
def nms(b,iou=0.5):
    if not b: return b
    b=np.array(sorted(b,key=lambda x:-x[4])); keep=[]; alive=np.ones(len(b),bool)
    area=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    for i in range(len(b)):
        if not alive[i]: continue
        keep.append(b[i]); xx1=np.maximum(b[i,0],b[:,0]); yy1=np.maximum(b[i,1],b[:,1]); xx2=np.minimum(b[i,2],b[:,2]); yy2=np.minimum(b[i,3],b[:,3])
        inter=np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None); alive&=inter/(area[i]+area-inter+1e-9)<iou; alive[i]=False
    return keep
acc=collections.defaultdict(lambda:[0,0,0,0])  # (role,var,th) -> 正解数, 拾えた, 枠, 外の枠
for k in range(5):
    for d in map(json.loads,open(f"../data/tiles/{TAG}fold{k}.jsonl")):
        L=rows[(d["item"],d["ci"])]; W,H=d["W"]//K+1,d["H"]//K+1; G=np.zeros((H,W),bool)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for v,parts in VARS.items():
            allb=nms([b for p in parts for b in d[p]])
            for th in TH:
                P=np.zeros_like(G); nb=no=0
                for b in allb:
                    if b[4]<th: continue
                    m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]; nb+=1
                    if m.size==0 or m.mean()<0.1: no+=1
                    if m.size==0 or m.mean()<FILL: continue
                    P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
                a=acc[(d["role"],v,th)]; a[0]+=len(L); a[2]+=nb; a[3]+=no
                for r in L:
                    x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
                    a[1]+=bool(m.size) and m.mean()>=0.5
with open(f"../data/tiles/{TAG}sweep.tsv","w") as f:
    f.write("role\tvar\tth\tregions\trecall\tboxes\toutside\n")
    for (role,v,th),(n,ok,nb,no) in sorted(acc.items()):
        f.write(f"{role}\t{v}\t{th}\t{n}\t{ok/n:.4f}\t{nb}\t{no}\n")
        print(f"{role:4} {v:11} {th:.2f}  拾えた {ok/n:.3f}  枠 {nb:6d}  外 {no:5d}")
