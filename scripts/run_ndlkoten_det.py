"""NDL古典籍OCR-Lite の検出器（RTMDet、行の範囲）だけを錦絵の各面にかけ、範囲を JSONL に出す。
画像は cantaloupe の IIIF Image API から長辺 2560px で取得し、座標は原寸に戻す。
使い方: ndlkotenocr-lite/.venv/bin/python run_ndlkoten_det.py [件数上限]"""
import sys,os,json,csv,io,urllib.request
import numpy as np
from PIL import Image
NDL=os.environ.get("NDLKOTEN_LITE_SRC","../../ndlkotenocr-lite/src")
sys.path.insert(0,NDL)
from rtmdet import RTMDet
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","data")
os.makedirs(OUT+"/img",exist_ok=True)
rows=list(csv.DictReader(open(os.path.join(OUT,"rows.tsv")),delimiter="\t"))
canv={}
for r in rows: canv.setdefault(r["canvas"],(r["img"],int(r["W"]),int(r["H"]),r["item"],r["ci"]))
det=RTMDet(model_path=NDL+"/model/rtmdet-s-1280x1280.onnx",class_mapping_path=NDL+"/config/ndl.yaml",
           score_threshold=0.3,conf_thresold=0.3,iou_threshold=0.3,device="cpu")
outp=os.path.join(OUT,"ndlkoten_det.jsonl")
done={json.loads(l)["canvas"] for l in open(outp)} if os.path.exists(outp) else set()
lim=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
n=0
with open(outp,"a") as f:
    for c,(img,W,H,item,ci) in canv.items():
        if c in done: continue
        if n>=lim: break
        p=f"{OUT}/img/{item}_{ci}.jpg"
        if not os.path.exists(p):
            url=img.replace(" ","%20")+"/full/"+("2560," if W>=H else ",2560")+"/0/default.jpg"
            if max(W,H)<=2560: url=img.replace(" ","%20")+"/full/max/0/default.jpg"
            open(p,"wb").write(urllib.request.urlopen(url,timeout=120).read())
        im=Image.open(p).convert("RGB"); s=W/im.size[0]
        ds=det.detect(np.array(im))
        boxes=[[float(b["box"][0])*s,float(b["box"][1])*s,float(b["box"][2])*s,float(b["box"][3])*s,float(b["confidence"])] for b in ds]
        f.write(json.dumps(dict(canvas=c,item=item,ci=ci,W=W,H=H,scale=s,boxes=boxes))+"\n"); f.flush(); n+=1
        print(n,item,ci,len(boxes),flush=True)
