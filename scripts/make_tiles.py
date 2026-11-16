"""切った画像でも学習させるため、data/yolo の各画像を 2×2（重なり 20%、run_yolo_tiles.py と同じ切り方）に切り、正解の枠も切り直す。
枠は片の内側に切り詰め、元の面積の 30% 以上が片に入るものだけ残す。
分割ごとに、学習 = 元の学習画像 + その片、検証 = 元の検証画像（全体のみ、前の学習と同じ）の一覧を作る。
使い方: python3 make_tiles.py [data の絶対パス]  → data/yolo_t2/{images,labels}/all/, data/cv/fold<k>_t2.yaml"""
import sys,os,glob
from PIL import Image
D=os.path.abspath(sys.argv[1] if len(sys.argv)>1 else "../data"); n=2
O=f"{D}/yolo_t2"; os.makedirs(f"{O}/images/all",exist_ok=True); os.makedirs(f"{O}/labels/all",exist_ok=True)
tiles={}; nb=0
for p in sorted(glob.glob(f"{D}/yolo/images/*/*.jpg")):
    name=os.path.basename(p)[:-4]; im=Image.open(p).convert("RGB"); W,H=im.size
    lp=p.replace("/images/","/labels/")[:-4]+".txt"
    B=[]
    for l in (open(lp) if os.path.exists(lp) else []):
        c,cx,cy,w,h=l.split(); cx,cy,w,h=float(cx)*W,float(cy)*H,float(w)*W,float(h)*H
        B.append((c,cx-w/2,cy-h/2,cx+w/2,cy+h/2))
    tw,th=W/(n-0.2*(n-1)),H/(n-0.2*(n-1)); out=[]
    for i in range(n):
        for j in range(n):
            x0,y0=int(i*tw*0.8),int(j*th*0.8); x1,y1=min(W,int(x0+tw)),min(H,int(y0+th)); tW,tH=x1-x0,y1-y0
            t=f"{name}_t{i}{j}"; im.crop((x0,y0,x1,y1)).save(f"{O}/images/all/{t}.jpg",quality=92)
            with open(f"{O}/labels/all/{t}.txt","w") as f:
                for c,a,b,cc,d in B:
                    ia,ib,ic,id_=max(a,x0),max(b,y0),min(cc,x1),min(d,y1)
                    if ic<=ia or id_<=ib or (ic-ia)*(id_-ib)<0.3*(cc-a)*(d-b): continue
                    f.write(f"{c} {((ia+ic)/2-x0)/tW:.6f} {((ib+id_)/2-y0)/tH:.6f} {(ic-ia)/tW:.6f} {(id_-ib)/tH:.6f}\n"); nb+=1
            out.append(f"{O}/images/all/{t}.jpg")
    tiles[name]=out
print(len(tiles),"面",sum(map(len,tiles.values())),"片",nb,"枠")
C=f"{D}/cv"
for k in range(5):
    tr=[l.strip() for l in open(f"{C}/fold{k}_train.txt") if l.strip()]
    ext=[t for p in tr for t in tiles[os.path.basename(p)[:-4]]]
    open(f"{C}/fold{k}_train_t2.txt","w").write("\n".join(tr+ext)+"\n")
    open(f"{C}/fold{k}_t2.yaml","w").write(f"path: {C}\ntrain: {C}/fold{k}_train_t2.txt\nval: {C}/fold{k}_val.txt\ntest: {C}/fold{k}_test.txt\nnames:\n  0: text\n")
    print(f"fold{k} 学習 {len(tr)}+{len(ext)}")
