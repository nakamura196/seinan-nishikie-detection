"""2026-11-02: Claude（Opus 5.5、Claude Code のサブエージェントが画像を見て答えた）の枠を、Gemini と同じ形の jsonl にする。
入力 ../data/claude_raw/<item>_<ci>.json（指示文は run_gemini4_agy.py と同じ、0-1000 の [ymin,xmin,ymax,xmax]）。出力 ../data/vlm_claude-opus-5-5_det.jsonl"""
import os,json,csv,re
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data")
meta={}
for r in csv.DictReader(open(f"{D}/rows.tsv"),delimiter="\t"): meta[(r["item"],str(r["ci"]))]=(r["canvas"],int(r["W"]),int(r["H"]))
s=json.load(open(f"{D}/vlm_subset.json")); ks=[tuple(k) for k in s["top30"]+s["random60"]]
n=0
with open(f"{D}/vlm_claude-opus-5-5_det.jsonl","w") as f:
    for item,ci in ks:
        p=f"{D}/claude_raw/{item}_{ci}.json"
        if not os.path.exists(p): print("無い",item,ci); continue
        txt=open(p).read(); c,W,H=meta[(item,ci)]
        ms=re.findall(r'"box_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',txt)
        boxes=[]
        for g in ms:
            y0,x0,y1,x1=(float(v)/1000 for v in g)
            boxes.append([min(x0,x1)*W,min(y0,y1)*H,max(x0,x1)*W,max(y0,y1)*H,1.0])
        f.write(json.dumps(dict(canvas=c,item=item,ci=ci,W=W,H=H,model="claude-opus-5-5",boxes=boxes,raw=txt),ensure_ascii=False)+"\n"); n+=1
print(n,"面")
