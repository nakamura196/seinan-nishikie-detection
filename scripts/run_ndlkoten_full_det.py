"""比較基準: NDL古典籍OCR（通常版 ver.3）のレイアウト認識（Cascade Mask R-CNN + ConvNeXt-T、行の範囲）だけを錦絵の各面にかけ、範囲を JSONL に出す。
公式は CUDA の Docker で配布されているので、Mac では mmcv 2.1.0 を CPU 向けに組んだ環境（~/git/ndl/ndlkotenocr_cli/.venv）で動かす。
設定と重みは公式のまま（ndl_kotenseki_layout_config_v3.py / ndl_kotenseki_layout_ver3.pth）。公式の CLI は score_thr 0.001 で枠を採るので、
ここでは得点つきで全部書き出し、閾値は評価のときに選ぶ（モデル内部の下限 0.05・1 面 100 枠まで）。
画像は ../data/img（長辺 2560px）を使い、座標は原寸に戻す。出力の形は ndlkoten_det.jsonl と同じ。
使い方: ~/git/ndl/ndlkotenocr_cli/.venv/bin/python run_ndlkoten_full_det.py [件数上限]"""
import sys,os,json,csv
from mmdet.apis import init_detector,inference_detector
from PIL import Image
NDL=os.path.expanduser("~/git/ndl/ndlkotenocr_cli/src/ndl_kotenseki_layout/models")
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","data")
rows=list(csv.DictReader(open(os.path.join(OUT,"rows.tsv")),delimiter="\t"))
canv={}
for r in rows: canv.setdefault(r["canvas"],(int(r["W"]),int(r["H"]),r["item"],r["ci"]))
outp=os.path.join(OUT,"ndlkoten_full_det.jsonl")
done={json.loads(l)["canvas"] for l in open(outp)} if os.path.exists(outp) else set()
lim=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
model=init_detector(NDL+"/ndl_kotenseki_layout_config_v3.py",NDL+"/ndl_kotenseki_layout_ver3.pth",None,"cpu"); n=0
with open(outp,"a") as f:
    for c,(W,H,item,ci) in canv.items():
        if c in done: continue
        if n>=lim: break
        p=f"{OUT}/img/{item}_{ci}.jpg"; s=W/Image.open(p).size[0]
        r=inference_detector(model,p).pred_instances
        boxes=[[float(b[0])*s,float(b[1])*s,float(b[2])*s,float(b[3])*s,float(sc)] for b,sc in zip(r.bboxes.tolist(),r.scores.tolist())]
        f.write(json.dumps(dict(canvas=c,item=item,ci=ci,W=W,H=H,scale=s,boxes=boxes))+"\n"); f.flush(); n+=1
        print(n,item,ci,len(boxes),flush=True)
