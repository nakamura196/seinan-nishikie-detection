"""感度の確認: NDL 通常版で 1 面 100 枠の上限（公式設定 test_cfg.rcnn.max_per_img）に届いた面だけ、上限を 1000 に上げて流し直す。
ndlkoten_full_det.jsonl の該当面を差し替えたものを ../data/ndlkoten_full_det_cap1000.jsonl に書く。
使い方: ~/git/ndl/ndlkotenocr_cli/.venv/bin/python run_ndlkoten_full_cap.py"""
import os,json
from mmdet.apis import init_detector,inference_detector
from mmengine.config import Config
from PIL import Image
NDL=os.path.expanduser("~/git/ndl/ndlkotenocr_cli/src/ndl_kotenseki_layout/models")
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data")
cfg=Config.fromfile(NDL+"/ndl_kotenseki_layout_config_v3.py"); cfg.model.test_cfg.rcnn.max_per_img=1000
model=init_detector(cfg,NDL+"/ndl_kotenseki_layout_ver3.pth",None,"cpu")
with open(f"{D}/ndlkoten_full_det_cap1000.jsonl","w") as f:
    for d in map(json.loads,open(f"{D}/ndlkoten_full_det.jsonl")):
        if len(d["boxes"])>=100:
            p=f"{D}/img/{d['item']}_{d['ci']}.jpg"; s=d["scale"]; r=inference_detector(model,p).pred_instances
            d["boxes"]=[[float(b[0])*s,float(b[1])*s,float(b[2])*s,float(b[3])*s,float(sc)] for b,sc in zip(r.bboxes.tolist(),r.scores.tolist())]
            print(d["item"],d["ci"],len(d["boxes"]),flush=True)
        f.write(json.dumps(d)+"\n")
