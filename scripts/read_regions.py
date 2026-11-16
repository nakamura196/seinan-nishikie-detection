"""検出した枠を NDL古典籍OCR-Lite の文字認識（PARSeq tiny）で読み、枠ごとの読みを保存する（拾えた枠が実際に読めるかを測るため）。
系: gt（正解の枠そのもの＝上限）、ndl（NDL の検出）、old（前のモデル、全体 0.25）、new（片でも学習したモデル、全体+2×2 を IoU 0.5 でまとめ 0.15）、fill（同じモデル 0.15、全体の枠を主に片の枠で隙間だけ埋める）、gemini（Gemini 4 の枠。読みは Gemini 自身のものではなく、この認識器で読み直す。2026-11-02 追加）。
各面は 5 分割の評価側（role test）の予測を使う。枠は余白なしで切り出す（どの系も同じ）。
使い方: ndlkotenocr-lite/.venv/bin/python read_regions.py <系> → ../data/read/<系>.jsonl"""
import sys,os,json,csv,collections
import numpy as np
from PIL import Image
from yaml import safe_load
from concurrent.futures import ThreadPoolExecutor
NDL=os.environ.get("NDLKOTEN_LITE_SRC","../../ndlkotenocr-lite/src"); sys.path.insert(0,NDL)
from parseq import PARSEQ
sysname=sys.argv[1]
chars=list(safe_load(open(NDL+"/config/NDLmoji.yaml",encoding="utf-8"))["model"]["charset_train"])
rec=PARSEQ(model_path=NDL+"/model/parseq-ndl-32x384-tiny-10.onnx",charlist=chars,device="cpu")
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
    """全体の枠を主にし、片の枠は、全体のどの枠とも自分の面積の ov 未満しか重ならないときだけ足す"""
    ar=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1])); out=list(full)
    for b in nms(t2):
        if all(max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))<ov*ar(b) for o in full): out.append(b)
    return out
pages={}
for k in range(5):
    for tag in ("","t2train_"):
        for d in map(json.loads,open(f"../data/tiles/{tag}fold{k}.jsonl")):
            if d["role"]!="test": continue
            p=pages.setdefault((d["item"],d["ci"]),dict(canvas=d["canvas"],W=d["W"],H=d["H"]))
            if tag=="": p["old"]=[b for b in d["full"] if b[4]>=0.25]
            else:
                p["new"]=[b for b in nms(d["full"]+d["t2"]) if b[4]>=0.15]
                p["fill"]=fillin(nms([b for b in d["full"] if b[4]>=0.15]),[b for b in d["t2"] if b[4]>=0.15])
for d in map(json.loads,open("../data/ndlkoten_det.jsonl")):
    if (d["item"],d["ci"]) in pages: pages[(d["item"],d["ci"])]["ndl"]=d["boxes"]
for d in map(json.loads,open("../data/gemini4_det.jsonl")):
    if (d["item"],str(d["ci"])) in pages: pages[(d["item"],str(d["ci"]))]["gemini"]=d["boxes"]
gt=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"):
    x,y,w,h=(float(r[c]) for c in "xywh"); gt[(r["item"],r["ci"])].append([x,y,x+w,y+h,1.0,r["id"]])
for key in pages: pages[key]["gt"]=gt[key]
os.makedirs("../data/read",exist_ok=True)
outp=f"../data/read/{sysname}.jsonl"
done={tuple(json.loads(l)["key"]) for l in open(outp)} if os.path.exists(outp) else set()
with open(outp,"a") as f, ThreadPoolExecutor(4) as ex:
    for n,(key,p) in enumerate(sorted(pages.items())):
        if key in done: continue
        im=np.array(Image.open(f"../data/img/{key[0]}_{key[1]}.jpg").convert("RGB")); s=p["W"]/im.shape[1]
        bx=p.get(sysname,[]); crops=[]
        for b in bx:
            x1,y1,x2,y2=(max(0,int(round(c/s))) for c in b[:4]); c=im[y1:max(y2,y1+2),x1:max(x2,x1+2)]
            crops.append(c if c.size else np.zeros((4,4,3),np.uint8))
        txt=list(ex.map(rec.read,crops))
        f.write(json.dumps(dict(key=list(key),canvas=p["canvas"],boxes=[list(map(float,b[:5]))+list(b[5:]) for b in bx],text=txt),ensure_ascii=False)+"\n"); f.flush()
        if n%100==0: print(sysname,n,len(pages),flush=True)
print(sysname,"done")
