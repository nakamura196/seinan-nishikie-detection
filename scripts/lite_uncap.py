"""Lite の ONNX の NMS 上限 200 を 2000 に変えた複製で、200 枠に達した面だけ推論し直す。
上限は ONNX の NonMaxSuppression の max_output_boxes_per_class（/Constant_37_output_0 = 200）。
複製は <out.jsonl> と同じ所に rtmdet_uncap.onnx として先に作っておく（onnx の入った python で）:
  python3 -c "import onnx,numpy as np;from onnx import numpy_helper as h;m=onnx.load('<lite>/src/model/rtmdet-s-1280x1280.onnx');[n.attribute[0].t.CopyFrom(h.from_array(np.array(2000,dtype=np.int64))) for n in m.graph.node if n.op_type=='Constant' and n.output[0]=='/Constant_37_output_0'];onnx.save(m,'rtmdet_uncap.onnx')"
使い方: ndlkotenocr-lite/.venv/bin/python lite_uncap.py <data dir> <out.jsonl>"""
import sys,json,os
import numpy as np
from PIL import Image
NDL=os.environ.get("NDLKOTEN_LITE_SRC","../../ndlkotenocr-lite/src"); sys.path.insert(0,NDL)
from rtmdet import RTMDet
D,out=sys.argv[1],sys.argv[2]
mp=os.path.join(os.path.dirname(out),"rtmdet_uncap.onnx")
det=RTMDet(model_path=mp,class_mapping_path=NDL+"/config/ndl.yaml",score_threshold=0.3,conf_thresold=0.3,iou_threshold=0.3,device="cpu")
with open(out,"w") as f:
    for d in map(json.loads,open(D+"/ndlkoten_det.jsonl")):
        if len(d["boxes"])<200: continue
        im=Image.open(f"{D}/img/{d['item']}_{d['ci']}.jpg").convert("RGB"); s=d["scale"]
        ds=det.detect(np.array(im))
        d["boxes"]=[[float(b["box"][0])*s,float(b["box"][1])*s,float(b["box"][2])*s,float(b["box"][3])*s,float(b["confidence"])] for b in ds]
        f.write(json.dumps(d)+"\n"); print(d["item"],d["ci"],len(d["boxes"]),flush=True)
