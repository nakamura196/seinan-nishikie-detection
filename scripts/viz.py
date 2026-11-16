"""1 面に正解（赤）と検出（青）を描く。使い方: python viz.py <検出 jsonl> <資料番号> <面> <出力.jpg>"""
import sys,json,csv
from PIL import Image,ImageDraw
jl,item,ci,outp=sys.argv[1:5]
d=[json.loads(l) for l in open(jl) if json.loads(l)["item"]==item and str(json.loads(l)["ci"])==ci][0]
im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB"); s=d["scale"]; dr=ImageDraw.Draw(im)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"):
    if r["canvas"]==d["canvas"]:
        x,y,w,h=(float(r[k])/s for k in "xywh"); dr.rectangle([x,y,x+w,y+h],outline=(230,0,0),width=4)
for b in d["boxes"]:
    dr.rectangle([b[0]/s,b[1]/s,b[2]/s,b[3]/s],outline=(0,80,255),width=3)
im.thumbnail((1400,1400)); im.save(outp,quality=85)
