"""学習した YOLO で面を検出し、run_ndlkoten_det.py と同じ形式の jsonl に書く（枠は元画像の座標）。
あわせて、NDL の結果から同じ面だけを抜いた jsonl も書く（同じ面で比べるため）。
使い方: python3 run_yolo_det.py <重み .pt> <出力 jsonl> [split=test] [conf=0.25]
split に fold<k>（例 fold0）を渡すと、data/cv.tsv の分割 k の面を対象にする（5 分割交差検証）。"""
import sys,json,csv
from ultralytics import YOLO
wt,outp=sys.argv[1],sys.argv[2]; split=sys.argv[3] if len(sys.argv)>3 else "test"
conf=float(sys.argv[4]) if len(sys.argv)>4 else 0.25
if split.startswith("fold"): faces=[r for r in csv.DictReader(open("../data/cv.tsv"),delimiter="\t") if r["fold"]==split[4:]]
else: faces=[r for r in csv.DictReader(open("../data/split.tsv"),delimiter="\t") if r["split"]==split]
ndl={}
for l in open("../data/ndlkoten_det.jsonl"):
    d=json.loads(l); ndl[d["canvas"]]=d
m=YOLO(wt)
with open(outp,"w") as f, open(f"../data/ndlkoten_det_{split}.jsonl","w") as g:
    for r in faces:
        d=ndl[r["canvas"]]; s=d["scale"]
        res=m.predict(f"../data/img/{r['item']}_{r['ci']}.jpg",imgsz=1280,conf=conf,device="mps",verbose=False,max_det=1000)[0]
        bx=[[*(float(v)*s for v in b),float(c)] for b,c in zip(res.boxes.xyxy.tolist(),res.boxes.conf.tolist())]
        f.write(json.dumps(dict(canvas=d["canvas"],item=d["item"],ci=d["ci"],W=d["W"],H=d["H"],scale=s,boxes=bx),ensure_ascii=False)+"\n")
        g.write(json.dumps(d,ensure_ascii=False)+"\n")
print(len(faces),"面")
