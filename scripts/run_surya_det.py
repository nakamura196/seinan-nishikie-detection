"""比較基準: Surya 0.17.1 の文字の行の検出器（DetectionPredictor）だけを錦絵の各面にかけ、範囲を JSONL に出す。
画像は run_ndlkoten_det.py が取得した ../data/img（長辺 2560px）を使い、座標は原寸に戻す。出力の形は ndlkoten_det.jsonl と同じ。
使い方: ~/.venvs/surya/bin/python run_surya_det.py [件数上限]"""
import sys,os,json,csv
from PIL import Image
from surya.detection import DetectionPredictor
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","data")
rows=list(csv.DictReader(open(os.path.join(OUT,"rows.tsv")),delimiter="\t"))
canv={}
for r in rows: canv.setdefault(r["canvas"],(int(r["W"]),int(r["H"]),r["item"],r["ci"]))
outp=os.path.join(OUT,"surya_det.jsonl")
done={json.loads(l)["canvas"] for l in open(outp)} if os.path.exists(outp) else set()
lim=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
det=DetectionPredictor(); n=0
with open(outp,"a") as f:
    for c,(W,H,item,ci) in canv.items():
        if c in done: continue
        if n>=lim: break
        im=Image.open(f"{OUT}/img/{item}_{ci}.jpg").convert("RGB"); s=W/im.size[0]
        res=det([im])[0]
        boxes=[[b.bbox[0]*s,b.bbox[1]*s,b.bbox[2]*s,b.bbox[3]*s,float(b.confidence or 0)] for b in res.bboxes]
        f.write(json.dumps(dict(canvas=c,item=item,ci=ci,W=W,H=H,scale=s,boxes=boxes))+"\n"); f.flush(); n+=1
        print(n,item,ci,len(boxes),flush=True)
