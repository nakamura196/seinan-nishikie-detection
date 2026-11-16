import sys,json,csv,collections,os,pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.argv=["x","t2train_"]
src=open("sweep_tiles.py").read().split("acc=collections")[0]; exec(src); TAG="t2train_"
import numpy as np
def fillin(full,t2,ov=0.5):
    ar=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1])); out=list(full)
    for b in nms(t2):
        if all(max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))<ov*ar(b) for o in full): out.append(b)
    return out
CF=[(th,ov) for th in [0.05,0.1,0.15,0.2,0.25,0.3] for ov in [0.1,0.3,0.5,0.7]]
res={c:{} for c in CF}; cnt={(role,c):[0,0,0,0] for role in ("test","val") for c in CF}
for k in range(5):
    for d in map(json.loads,open(f"../data/tiles/{TAG}fold{k}.jsonl")):
        L=rows[(d["item"],d["ci"])]; W,H=d["W"]//K+1,d["H"]//K+1; G=np.zeros((H,W),bool)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for th,ov in CF:
            bs=fillin(nms([b for b in d["full"] if b[4]>=th]),[b for b in d["t2"] if b[4]>=th],ov)
            P=np.zeros_like(G); nb=no=0
            for b in bs:
                m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]; nb+=1
                if m.size==0 or m.mean()<0.1: no+=1
                if m.size==0 or m.mean()<FILL: continue
                P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
            a=cnt[(d["role"],(th,ov))]; a[0]+=len(L); a[2]+=nb; a[3]+=no
            for r in L:
                x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
                ok=bool(m.size) and m.mean()>=0.5; a[1]+=ok
                if d["role"]=="test": res[(th,ov)][r["id"]]=ok
for (role,c),a in sorted(cnt.items()):
    print(role,c,a[0],f"{100*a[1]/a[0]:.2f}",a[2],a[3])
pickle.dump(res,open("review-2026-11-16/fill_res.pkl","wb"))
