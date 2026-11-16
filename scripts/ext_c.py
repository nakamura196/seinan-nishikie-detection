"""外部の評価用の画像に C（片でも学習した YOLO11s）をかける（2026-11-03）。主設定と同じ推論: 全体（1280）＋2×2 の片（重なり 20%）、しきい値 0.15、
全体の枠を主にし、片の枠は全体のどの枠とも自分の面積の 5 割未満しか重ならないときだけ足す（fill-in。paper_tables.py の fillin と同じ）。
重みは 5 分割のうちの 1 本（既定 fold0。西南戦争の評価ではどの面も自分を学習に使っていない分割のモデルで推論したが、外部の画像はどれも未見なので 1 本で足りる）。
出力: ../data/external/machine/c_det.jsonl（座標は面の原寸、得点つき）。学習はしない（推論のみ、Mac の mps）。
使い方: [FOLD=0] python3 ext_c.py"""
import json,csv,os
import numpy as np
from PIL import Image
from ultralytics import YOLO
D="../data/external"; os.makedirs(f"{D}/machine",exist_ok=True); FOLD=os.environ.get("FOLD","0"); TH=0.15
m=YOLO(f"../data/cv_t2_weights/fold{FOLD}.pt")
def pred(im,ox,oy,s):
    r=m.predict(im,imgsz=1280,conf=TH,device="mps",verbose=False,max_det=1000)[0]
    return [[(b[0]+ox)*s,(b[1]+oy)*s,(b[2]+ox)*s,(b[3]+oy)*s,c] for b,c in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist())]
def nms(b,iou=0.5):
    b=sorted(b,key=lambda x:-x[4]); keep=[]
    for x in b:
        ok=True
        for k in keep:
            i=max(0,min(x[2],k[2])-max(x[0],k[0]))*max(0,min(x[3],k[3])-max(x[1],k[1]))
            if i/((x[2]-x[0])*(x[3]-x[1])+(k[2]-k[0])*(k[3]-k[1])-i+1e-9)>=iou: ok=False; break
        if ok: keep.append(x)
    return keep
def fillin(full,t2,ov=0.5):
    ar=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1])); out=list(full)
    for b in nms(t2):
        if all(max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))<ov*ar(b) for o in full): out.append(b)
    return out
with open(f"{D}/machine/c_det.jsonl","w") as f:
    for r in csv.DictReader(open(f"{D}/images.tsv"),delimiter="\t"):
        im=Image.open(f"{D}/images/{r['item']}_{r['ci']}.jpg").convert("RGB"); W,H=im.size; s=int(r["W"])/W
        full=nms(pred(im,0,0,s)); n=2; tw,th=W/(n-0.2*(n-1)),H/(n-0.2*(n-1)); t2=[]
        for i in range(n):
            for j in range(n):
                x0,y0=int(i*tw*0.8),int(j*th*0.8); t2+=pred(im.crop((x0,y0,min(W,int(x0+tw)),min(H,int(y0+th)))),x0,y0,s)
        bx=fillin(full,t2)
        f.write(json.dumps(dict(canvas=r["canvas"],item=r["item"],ci=r["ci"],W=int(r["W"]),H=int(r["H"]),scale=s,model=f"cv_t2 fold{FOLD}",boxes=bx))+"\n"); f.flush()
        print(r["item"],r["ci"],len(bx),flush=True)
