"""1 種類「文字の範囲」の YOLO を学習する。学習は mdx-gpu（A100）で回す（Mac では回さない）。
MPS では、正解の多い面で枠の割り当て（TaskAlignedAssigner）が落ちる（torch 2.8 / ultralytics 8.4.7、
"size of tensor a must match the size of tensor b"）。MPS のときだけ割り当てを CPU で計算して避ける（CUDA では何もしない）。
使い方: python3 train_yolo.py <名前> [モデル=yolo11s.pt] [imgsz=1280] [epochs=80] [device=0] [batch=16] [data=../yolo/seinan.yaml（data/runs から見た位置）]"""
import sys,os
import ultralytics.utils.tal as tal
from ultralytics import YOLO
_orig=tal.TaskAlignedAssigner.forward
def _fwd(self,*a):
    dev=a[0].device
    if dev.type!="mps": return _orig(self,*a)
    return tuple(t.to(dev) for t in _orig(self,*[t.cpu() for t in a]))
tal.TaskAlignedAssigner.forward=_fwd
name=sys.argv[1]; model=sys.argv[2] if len(sys.argv)>2 else "yolo11s.pt"
imgsz=int(sys.argv[3]) if len(sys.argv)>3 else 1280; ep=int(sys.argv[4]) if len(sys.argv)>4 else 80
device=sys.argv[5] if len(sys.argv)>5 else "0"; batch=int(sys.argv[6]) if len(sys.argv)>6 else 16
data=sys.argv[7] if len(sys.argv)>7 else "../yolo/seinan.yaml"
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/runs"))
YOLO(model).train(data=data,imgsz=imgsz,epochs=ep,patience=20,batch=batch,device=device,project=".",name=name,seed=20261030,workers=4,exist_ok=True)
