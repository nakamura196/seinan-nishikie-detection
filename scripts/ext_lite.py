"""外部の評価用の画像に NDL古典籍OCR-Lite の検出器（RTMDet）をかける（2026-11-03）。run_ndlkoten_det.py と同じ設定（しきい値 0.3、CPU）。
入力: ../data/external/images.tsv と images/。出力: ../data/external/machine/lite_det.jsonl（ndlkoten_det.jsonl と同じ形式。座標は面の原寸）。
使い方: ndlkotenocr-lite/.venv/bin/python ext_lite.py"""
import sys,os,json,csv
import numpy as np
from PIL import Image
NDL=os.environ.get("NDLKOTEN_LITE_SRC","../../ndlkotenocr-lite/src"); sys.path.insert(0,NDL)
from rtmdet import RTMDet
D="../data/external"; os.makedirs(f"{D}/machine",exist_ok=True)
det=RTMDet(model_path=NDL+"/model/rtmdet-s-1280x1280.onnx",class_mapping_path=NDL+"/config/ndl.yaml",
           score_threshold=0.3,conf_thresold=0.3,iou_threshold=0.3,device="cpu")
outp=f"{D}/machine/lite_det.jsonl"
done={json.loads(l)["canvas"] for l in open(outp)} if os.path.exists(outp) else set()
with open(outp,"a") as f:
    for r in csv.DictReader(open(f"{D}/images.tsv"),delimiter="\t"):
        if r["canvas"] in done: continue
        im=Image.open(f"{D}/images/{r['item']}_{r['ci']}.jpg").convert("RGB"); W,H=int(r["W"]),int(r["H"]); s=W/im.size[0]
        boxes=[[float(b["box"][i])*s for i in range(4)]+[float(b["confidence"])] for b in det.detect(np.array(im))]
        f.write(json.dumps(dict(canvas=r["canvas"],item=r["item"],ci=r["ci"],W=W,H=H,scale=s,boxes=boxes))+"\n"); f.flush()
        print(r["item"],r["ci"],len(boxes),flush=True)
