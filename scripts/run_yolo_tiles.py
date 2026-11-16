"""しきい値と切り分け推論を試すため、分割 k のモデルで評価の面と検証の面を検出し、低い確信度の枠まで残す。
見せ方は 3 つ: full（画像全体を 1280 で）、t2（2×2 に切る）、t3（3×3 に切る）。切った片は重なり 20% で、それぞれ IMGSZ で推論する。
枠は元画像の座標で、見せ方ごとに分けて書く（合わせ方としきい値は sweep_tiles.py で決める）。
使い方: python3 run_yolo_tiles.py <k> [conf=0.05]  → ../data/tiles/<TAG>fold<k>.jsonl
環境変数: DEV（既定 mps、mdx-gpu では 0）、IMGSZ（既定 1280）、WT（重みの位置、{k} を分割番号に置き換える。既定 ../data/cv_weights/fold{k}.pt）、TAG（出力名の頭）"""
import sys,json,csv,os
from PIL import Image
from ultralytics import YOLO
DEV=os.environ.get("DEV","mps"); SZ=int(os.environ.get("IMGSZ","1280")); TAG=os.environ.get("TAG","")
WT=os.environ.get("WT","../data/cv_weights/fold{k}.pt")
k=sys.argv[1]; conf=float(sys.argv[2]) if len(sys.argv)>2 else 0.05
faces=[(r["item"],r["ci"],"test") for r in csv.DictReader(open("../data/cv.tsv"),delimiter="\t") if r["fold"]==k]
faces+=[(*os.path.basename(p).strip()[:-4].rsplit("_",1),"val") for p in open(f"../data/cv/fold{k}_val.txt") if p.strip()]
faces=faces[:int(os.environ["LIMIT"])] if "LIMIT" in os.environ else faces
ndl={(d["item"],str(d["ci"])):d for d in map(json.loads,open("../data/ndlkoten_det.jsonl"))}
m=YOLO(WT.replace("{k}",k)); os.makedirs("../data/tiles",exist_ok=True)
def pred(im,ox,oy,s):
    r=m.predict(im,imgsz=SZ,conf=conf,device=DEV,verbose=False,max_det=1000)[0]
    return [[(b[0]+ox)*s,(b[1]+oy)*s,(b[2]+ox)*s,(b[3]+oy)*s,c] for b,c in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist())]
with open(f"../data/tiles/{TAG}fold{k}.jsonl","w") as f:
    for item,ci,role in faces:
        d=ndl[(item,ci)]; s=d["scale"]; im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB"); W,H=im.size
        out=dict(item=item,ci=ci,canvas=d["canvas"],W=d["W"],H=d["H"],role=role,full=pred(im,0,0,s))
        for n in (2,3):
            tw,th=W/(n-0.2*(n-1)),H/(n-0.2*(n-1)); bx=[]
            for i in range(n):
                for j in range(n):
                    x0,y0=int(i*tw*0.8),int(j*th*0.8)
                    bx+=pred(im.crop((x0,y0,min(W,int(x0+tw)),min(H,int(y0+th)))),x0,y0,s)
            out[f"t{n}"]=bx
        f.write(json.dumps(out)+"\n")
print("fold",k,len(faces),"面")
