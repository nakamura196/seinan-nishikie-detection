"""5 分割の切り分け推論の結果を、字数・難しい面・NDL の取りこぼし上位 30 面・格子の面（2012-06-03 p1）に分けて、拾えた割合を出す。
使い方: python3 breakdown_tiles.py [TAG]（run_yolo_tiles.py の出力 ../data/tiles/<TAG>fold<k>.jsonl を読む）"""
import sys,json,csv,collections
_T=sys.argv[1] if len(sys.argv)>1 else ""; sys.argv=["x"]; import numpy as np
exec(open("sweep_tiles.py").read().split("acc=collections")[0]); TAG=_T
base={r["id"]:float(r["cov"]) for r in csv.DictReader(open("../data/ndlkoten_cov_fill0.2.tsv"),delimiter="\t")}
CF=[("full",0.25),("full",0.1),("full+t2",0.25),("full+t2",0.15),("full+t2",0.1),("fill",0.15)]
res={c:{} for c in CF}
def fillin(full,t2,ov=0.5):
    """全体の枠を主にし、2×2 の枠は全体のどの枠とも自分の面積の ov 未満しか重ならないときだけ足す"""
    ar=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1])); out=list(full)
    for b in nms(t2):
        if all(max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))<ov*ar(b) for o in full): out.append(b)
    return out
for k in range(5):
    for d in map(json.loads,open(f"../data/tiles/{TAG}fold{k}.jsonl")):
        if d["role"]!="test": continue
        L=rows[(d["item"],d["ci"])]; W,H=d["W"]//K+1,d["H"]//K+1; G=np.zeros((H,W),bool)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for v,th in CF:
            P=np.zeros_like(G)
            if v=="fill": bs=fillin(nms([b for b in d["full"] if b[4]>=th]),[b for b in d["t2"] if b[4]>=th])
            else: bs=nms([b for p in VARS[v] for b in d[p]])
            for b in bs:
                if b[4]<th: continue
                m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
                if m.size==0 or m.mean()<FILL: continue
                P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
            for r in L:
                x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
                res[(v,th)][r["id"]]=(r,bool(m.size) and m.mean()>=0.5)
R={r["id"]:r for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t")}
img=collections.defaultdict(list)
for i,r in R.items(): img[(r["item"],r["ci"])].append(i)
hard6=[i for k,v in img.items() if len(v)>=2 and np.mean([base[x]>=.5 for x in v])<.5 for i in v]
miss=sorted(img,key=lambda k:-sum(base[x]<.5 for x in img[k]))[:30]; top30=[i for k in miss for i in img[k]]
grid=img[("2012-06-03","1")]
def L(t): n=len(t); return "1" if n==1 else "2-3" if n<=3 else "4-10" if n<=10 else "11+"
groups={"全体":list(R),"1字":[i for i in R if L(R[i]["text"])=="1"],"2-3字":[i for i in R if L(R[i]["text"])=="2-3"],"難6面":hard6,"上位30面":top30,"格子":grid}
print("設定".ljust(16),*[g for g in groups])
print("NDL".ljust(16),*[f"{np.mean([base[i]>=.5 for i in v]):.3f}" for v in groups.values()])
for c in CF: print(f"{c[0]}@{c[1]}".ljust(16),*[f"{np.mean([res[c][i][1] for i in v]):.3f}" for v in groups.values()])
