"""参考値の表: Gemini 4 を答えさせた 90 面（../data/vlm_subset.json）だけで、NDL Lite / NDL 通常版 / Surya / 主設定 C と並べる。
尺度は baselines.py と同じ（1/8 の格子、覆い 0.5 以上で拾えた、正解の文字 20% 未満の枠は覆いに数えない、重なり 10% 未満を「外」）。
上位 30 面と無作為 60 面を分けて出す（無作為 60 面が全体の傾向、上位 30 面が難しい面の参考）。
2026-11-02: Gemini 4 を全 739 面で回したので「全739」も出す。拾えた割合の差（Gemini − C、Gemini − NDL Lite）の区間を資料単位のブートストラップ（2,000 回）で添える。
使い方: python3 vlm_subset.py [NDL 通常版の閾値（既定 0.3）] [Surya の閾値（既定 0）]"""
import sys,json,csv,collections,os
import numpy as np
K=8; FILL=0.2; D="../data"
thN=float(sys.argv[1]) if len(sys.argv)>1 else 0.3; thS=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
R={r["id"]:r for r in csv.DictReader(open(__import__("os").environ.get("ROWS",f"{D}/rows.tsv")),delimiter="\t")}
byimg=collections.defaultdict(list)
for r in R.values(): byimg[(r["item"],str(r["ci"]))].append(r)
sub=json.load(open(f"{D}/vlm_subset.json")); G={g:[tuple(k) for k in v] for g,v in sub.items()}; G["全739"]=sorted(byimg)
def evaluate(path,th,keys):
    cov={}; nb=0; out=0
    for d in map(json.loads,open(path)):
        k=(d["item"],str(d["ci"]))
        if k not in keys: continue
        L=byimg[k]; W,H=d["W"]//K+1,d["H"]//K+1; Gr=np.zeros((H,W),bool); P=np.zeros_like(Gr)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); Gr[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for b in d["boxes"]:
            if len(b)>4 and b[4]<th: continue
            m=Gr[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]; nb+=1
            if m.size==0 or m.mean()<0.1: out+=1
            if m.size==0 or m.mean()<FILL: continue
            P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
            cov[r["id"]]=float(m.mean()) if m.size else 0.0
    return cov,nb,out
S={"NDL Lite":(f"{D}/ndlkoten_det.jsonl",0.3),f"NDL full {thN}":(f"{D}/ndlkoten_full_det.jsonl",thN),
   f"Surya {thS}":(f"{D}/surya_det.jsonl",thS),"Gemini 4":(f"{D}/gemini4_det.jsonl",0),"C":(f"{D}/main_cv_all.jsonl",0)}
# 2026-11-02（11-03 に 3 回目を追加）: 同じ指示文での 2・3 回目（ばらつき）と、別の世代・別の会社の VLM（参考の 90 面）。ファイルがあれば並べる
for name,fn in [("Gemini 4 rep2","gemini4_det_rep2.jsonl"),("Gemini 4 rep3","gemini4_det_rep3.jsonl"),("Gemini 3.8 Flash","vlm_gemini-3.8-flash-high_det.jsonl"),("Claude Opus 4.6","vlm_claude-opus-4-6-thinking_det.jsonl"),("Claude Opus 5.5","vlm_claude-opus-5-5_det.jsonl")]:
    if os.path.exists(f"{D}/{fn}"): S[name]=(f"{D}/{fn}",0)
gem={(d["item"],str(d["ci"])) for d in map(json.loads,open(f"{D}/gemini4_det.jsonl"))}
def L(t): n=len(t); return "1" if n==1 else "2-3" if n<=3 else "4-10" if n<=10 else "11-30" if n<=30 else "31+"
allcov={}
for g,keys in G.items():
    keys=set(keys)&gem; ids=[r["id"] for k in keys for r in byimg[k]]
    print(f"== {g}: {len(keys)} 面、範囲 {len(ids)}")
    res={n:evaluate(p,th,keys) for n,(p,th) in S.items() if os.path.exists(p)}
    print("  系 | 拾えた | 詞書 | 札 | 1字 | 2-3字 | ≥0.9 | 覆い0 | 枠 | 外")
    for n,(cov,nb,out) in res.items():
        if len(cov)<len(ids): print(f"  {n}: {len(cov)}/{len(ids)} 範囲のみ（未完）"); continue
        pc=lambda v,t=0.5:100*np.mean([cov[i]>=t for i in v]) if v else float("nan")
        sel=lambda f:[i for i in ids if f(R[i])]
        allcov[g,n]=cov
        print(f"  {n} | {pc(ids):.1f} | {pc(sel(lambda r:r['pred']=='詞書')):.1f} | {pc(sel(lambda r:r['pred']=='札')):.1f} | {pc(sel(lambda r:L(r['text'])=='1')):.1f} | {pc(sel(lambda r:L(r['text'])=='2-3')):.1f} | {pc(ids,0.9):.1f} | {100*np.mean([cov[i]==0 for i in ids]):.1f} | {nb} | {out}")

rng=np.random.default_rng(20261102)
for g,keys in G.items():
    keys=sorted(set(keys)&gem); items=sorted({k[0] for k in keys}); byit=collections.defaultdict(list)
    for k in keys: byit[k[0]]+= [r["id"] for r in byimg[k]]
    for a,b in (("Gemini 4","C"),("Gemini 4","NDL Lite"),("Gemini 4","Claude Opus 5.5"),("Claude Opus 5.5","NDL Lite"),("Gemini 4 rep2","C"),("Gemini 4 rep3","C")):
        if (g,a) not in allcov or (g,b) not in allcov: continue
        ca,cb=allcov[g,a],allcov[g,b]
        def diff(its):
            ids=[i for it in its for i in byit[it]]; return 100*(np.mean([ca[i]>=.5 for i in ids])-np.mean([cb[i]>=.5 for i in ids]))
        bs=[diff(rng.choice(items,len(items))) for _ in range(2000)]
        print(f"  {g}: {a} − {b} = {diff(items):.1f} [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]（資料 {len(items)}）")
