"""【使わない。read_regions_masked.py に置き換えた（2026-11-03）：切り出した細い枠が拡大され、Lite が 1 行を割ってしまうため】
検出した枠を、NDL古典籍OCR-Lite の全工程（レイアウト検出 RTMDet → 行の組み立て → 文字認識 PARSeq）で読む（2026-11-03 追加）。
read_regions.py は枠をそのまま 1 行として認識器に渡すので、段落を 1 枠で囲む Gemini 4 に不利だった。
ここでは枠ごとに切り出した画像を Lite の ocr.py と同じ手順にかけ、枠の中で行を分け直してから読む。
出力の boxes は、Lite が枠の中に見つけた行（面の座標に戻したもの）、text はその読み。eval_read.py でそのまま採点できる。
系: gemini（Gemini 4 の枠）、fill（C の主設定）、gt（正解の枠。上限の目安）。
枠の中に行が 1 つも見つからなかったときは、<系>_full には何も足さず、<系>_fullfb には枠全体を 1 行として読んだものを足す。
使い方: ndlkotenocr-lite/.venv/bin/python read_regions_full.py <系> → ../data/read/<系>_full.jsonl と <系>_fullfb.jsonl"""
import sys,os,json,csv,collections
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from yaml import safe_load
NDL=os.environ.get("NDLKOTEN_LITE_SRC","../../ndlkotenocr-lite/src"); sys.path.insert(0,NDL)
from rtmdet import RTMDet
from parseq import PARSEQ
from ndl_parser import convert_to_xml_string3
from reading_order.xy_cut.eval import eval_xml
sysname=sys.argv[1]
chars=list(safe_load(open(NDL+"/config/NDLmoji.yaml",encoding="utf-8"))["model"]["charset_train"])
rec=PARSEQ(model_path=NDL+"/model/parseq-ndl-32x384-tiny-10.onnx",charlist=chars,device="cpu")
# ocr.py の既定値と同じ
det=RTMDet(model_path=NDL+"/model/rtmdet-s-1280x1280.onnx",class_mapping_path=NDL+"/config/ndl.yaml",
           score_threshold=0.3,conf_thresold=0.3,iou_threshold=0.3,device="cpu")
classes=list(det.classes.values())
def nms(b,iou=0.5):
    if not b: return b
    b=np.array(sorted(b,key=lambda x:-x[4])); keep=[]; alive=np.ones(len(b),bool)
    area=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    for i in range(len(b)):
        if not alive[i]: continue
        keep.append(list(b[i])); xx1=np.maximum(b[i,0],b[:,0]); yy1=np.maximum(b[i,1],b[:,1]); xx2=np.minimum(b[i,2],b[:,2]); yy2=np.minimum(b[i,3],b[:,3])
        inter=np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None); alive&=inter/(area[i]+area-inter+1e-9)<iou; alive[i]=False
    return keep
def fillin(full,t2,ov=0.5):
    ar=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1])); out=list(full)
    for b in nms(t2):
        if all(max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))<ov*ar(b) for o in full): out.append(b)
    return out
def lite_lines(img):
    """ocr.py の process と同じ手順で、画像の中の行の範囲（画像の座標）を返す"""
    h,w=img.shape[:2]
    ds=det.detect(img)
    res=[{0:[]},{i:[] for i in range(16)}]
    for d in ds:
        x1,y1,x2,y2=d["box"]
        if d["class_index"]==0: res[0][0].append([x1,y1,x2,y2])
        res[1][d["class_index"]].append([x1,y1,x2,y2,d["confidence"]])
    xs=convert_to_xml_string3(w,h,"crop.jpg",classes,res,score_thr=0.3,min_bbox_size=5,use_block_ad=False)
    root=ET.fromstring("<OCRDATASET>"+xs+"</OCRDATASET>"); eval_xml(root,logger=None)
    out=[]
    for l in root.findall(".//LINE"):
        x,y,lw,lh=(int(l.get(k)) for k in ("X","Y","WIDTH","HEIGHT"))
        if lw>0 and lh>0: out.append((x,y,x+lw,y+lh))
    return out
pages={}
for k in range(5):
    for d in map(json.loads,open(f"../data/tiles/t2train_fold{k}.jsonl")):
        if d["role"]!="test": continue
        pages[(d["item"],d["ci"])]=dict(canvas=d["canvas"],W=d["W"],H=d["H"],
            fill=fillin(nms([b for b in d["full"] if b[4]>=0.15]),[b for b in d["t2"] if b[4]>=0.15]))
for d in map(json.loads,open("../data/gemini4_det.jsonl")):
    if (d["item"],str(d["ci"])) in pages: pages[(d["item"],str(d["ci"]))]["gemini"]=d["boxes"]
gt=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"):
    x,y,w,h=(float(r[c]) for c in "xywh"); gt[(r["item"],r["ci"])].append([x,y,x+w,y+h,1.0])
for key in pages: pages[key]["gt"]=gt[key]
outp=f"../data/read/{sysname}_full.jsonl"; outfb=f"../data/read/{sysname}_fullfb.jsonl"
done={tuple(json.loads(l)["key"]) for l in open(outp)} if os.path.exists(outp) else set()
nolines=0; nbox=0
with open(outp,"a") as f, open(outfb,"a") as ffb:
    for n,(key,p) in enumerate(sorted(pages.items())):
        if key in done: continue
        im=np.array(Image.open(f"../data/img/{key[0]}_{key[1]}.jpg").convert("RGB")); s=p["W"]/im.shape[1]
        B,T,Bfb,Tfb=[],[],[],[]
        for b in p.get(sysname,[]):
            x1,y1,x2,y2=(max(0,int(round(c/s))) for c in b[:4]); c=im[y1:max(y2,y1+2),x1:max(x2,x1+2)]
            nbox+=1
            if c.size==0 or min(c.shape[:2])<2: continue
            ls=lite_lines(c)
            if not ls:
                nolines+=1; Bfb.append([float(v) for v in b[:4]]+[1.0]); Tfb.append(rec.read(c)); continue
            for lx1,ly1,lx2,ly2 in ls:
                t=rec.read(c[ly1:ly2,lx1:lx2])
                bb=[(x1+lx1)*s,(y1+ly1)*s,(x1+lx2)*s,(y1+ly2)*s,1.0]
                B.append(bb); T.append(t); Bfb.append(bb); Tfb.append(t)
        f.write(json.dumps(dict(key=list(key),canvas=p["canvas"],boxes=B,text=T),ensure_ascii=False)+"\n"); f.flush()
        ffb.write(json.dumps(dict(key=list(key),canvas=p["canvas"],boxes=Bfb,text=Tfb),ensure_ascii=False)+"\n"); ffb.flush()
        if n%50==0: print(sysname,n,len(pages),"boxes",nbox,"no-line",nolines,flush=True)
print(sysname,"done","boxes",nbox,"no-line",nolines)
