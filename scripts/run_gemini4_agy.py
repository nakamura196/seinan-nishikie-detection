"""参考値: Gemini 4（agy の gemini-4-argon-high、読み取りだけの plan モード）に錦絵の各面の文字の範囲を JSON で答えさせる。
対象は 90 面: NDL Lite の取りこぼしが多い上位 30 面（baselines.py の「上位30面」と同じ選び方）と、残りから無作為の 60 面（seed 20261101）。
面の一覧は ../data/vlm_subset.json に書く。画像は ../data/img（長辺 2560px）をそのまま見せる。
座標は box_2d = [ymin, xmin, ymax, xmax]、0〜1000 に正規化（1990-032-187 p1 で注釈と重ねて確かめた）。得点は無いので 1.0。
出力の形は ndlkoten_det.jsonl と同じ（../data/gemini4_det.jsonl、途中から再開できる、raw に返答をそのまま残す）。
2026-11-02: --all を付けると全 739 面を回す（VLM を主役にする組み直しのため。指示文・モデルは同じ。済んだ 90 面は飛ばす）。
2026-11-02: 元の指示文の "label": "text" を Gemini が書き換えずに返し、読みが 739 面中 6 面にしか入っていなかった。
VER=read を付けると、文字を書き起こすよう明示した指示文で回し、../data/gemini4_read_det.jsonl に書く（枠の結果は元のファイルを上書きしない）。
2026-11-02: REP=2 などを付けると、同じ指示文で回し直して ..._rep2.jsonl に書く（ばらつきを測るため）。
2026-11-02: MODEL=<agy のモデル名> で別の VLM を回し、../data/vlm_<モデル名>_det.jsonl に書く（世代による違いを見るため。--all を付けなければ参考の 90 面だけ）。
2026-11-03: SET=external で外部の評価用（NDL の日清戦争錦絵、../data/external/images.tsv の全面）を同じ指示文で回し、../data/external/machine/gemini4_det.jsonl に書く（--all は不要。VER・REP・MODEL も使える）。
使い方: [SET=external] [VER=read] [REP=N] [MODEL=...] python3 run_gemini4_agy.py [並列数（既定 2）] [--all] [--limit N]（--limit は未処理の先頭 N 面だけ回す。上限の減り方を測るため）"""
import sys,os,json,csv,re,random,subprocess,collections,threading
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data")
MODEL=os.environ.get("MODEL","gemini-4-argon-high")
PROMPT=("Open the image file {p} . It is a Japanese woodblock print. Detect every piece of written text in it: each column or block "
        "of the narrative inscription, every short name label (cartouche) beside figures, the title, and signatures. "
        'Return ONLY a JSON list, one entry per text region: [{{"box_2d": [ymin, xmin, ymax, xmax], "label": "text"}}], '
        "coordinates normalized to 0-1000 relative to the full image. No other text.")
VER=os.environ.get("VER","")
if VER=="read":
    PROMPT=("Open the image file {p} . It is a Japanese woodblock print (nishiki-e). Detect every piece of written text in it: each column or block "
            "of the narrative inscription, every short name label (cartouche) beside figures, the title, and signatures. "
            "For each region, transcribe the characters exactly as written in the print (kanji and kana as they appear; write illegible characters as □). "
            'Return ONLY a JSON list, one entry per text region: [{{"box_2d": [ymin, xmin, ymax, xmax], "transcription": "<the characters in this region>"}}], '
            "coordinates normalized to 0-1000 relative to the full image. No other text.")
R=list(csv.DictReader(open(f"{D}/rows.tsv"),delimiter="\t"))
base={r["id"]:float(r["cov"]) for r in csv.DictReader(open(f"{D}/ndlkoten_cov_fill0.2.tsv"),delimiter="\t")}
bi=collections.defaultdict(list); meta={}
for r in R:
    k=(r["item"],str(r["ci"])); bi[k].append(r["id"]); meta[k]=(r["canvas"],int(r["W"]),int(r["H"]))
top30=sorted(bi,key=lambda k:-sum(base[x]<.5 for x in bi[k]))[:30]
rest=sorted(k for k in bi if k not in top30); rnd=random.Random(20261101).sample(rest,60)
sub=[*top30,*rnd]
if "--all" in sys.argv: sub+=[k for k in sorted(bi) if k not in set(sub)]
EXT=os.environ.get("SET")=="external"; IMG=f"{D}/img"
if EXT:
    E=f"{D}/external"; os.makedirs(f"{E}/machine",exist_ok=True); IMG=f"{E}/images"; bi={}; meta={}
    for r in csv.DictReader(open(f"{E}/images.tsv"),delimiter="\t"):
        k=(r["item"],r["ci"]); bi[k]=[]; meta[k]=(r["canvas"],int(r["W"]),int(r["H"]))
    sub=sorted(bi,key=lambda k:(k[0],int(k[1])))
else: json.dump({"top30":[list(k) for k in top30],"random60":[list(k) for k in rnd]},open(f"{D}/vlm_subset.json","w"),ensure_ascii=False,indent=0)
outp=f"{D}/gemini4_read_det.jsonl" if VER=="read" else f"{D}/gemini4_det.jsonl"
if EXT: outp=outp.replace(f"{D}/",f"{D}/external/machine/")
REP=os.environ.get("REP","")
if REP: outp=outp.replace(".jsonl",f"_rep{REP}.jsonl")
if MODEL!="gemini-4-argon-high": outp=outp.replace("gemini4_",f"vlm_{MODEL}_")
done={(d["item"],str(d["ci"])) for d in map(json.loads,open(outp))} if os.path.exists(outp) else set()
lock=threading.Lock(); fails=[0]
def run(k):
    item,ci=k; c,W,H=meta[k]; p=os.path.abspath(f"{IMG}/{item}_{ci}.jpg")
    for _ in range(3):
        r=subprocess.run(["agy","--model",MODEL,"--mode","plan","--add-dir",os.path.dirname(p),"--print-timeout","900s","-p",PROMPT.format(p=p)],
                         capture_output=True,text=True)
        txt=r.stdout
        ms=re.findall(r'"box_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',txt)
        if ms or "[]" in txt: break
    if not ms and "[]" not in txt:
        # 答えが取れなかった（上限など）。書き込まずに飛ばし、次に流したときにやり直す。5 回続いたら止める
        with lock:
            fails[0]+=1; print(item,ci,"失敗",r.stderr[-200:].strip(),flush=True)
            if fails[0]>=5: print("失敗が 5 回続いたので止める",flush=True); os._exit(1)
        return
    fails[0]=0
    boxes=[]
    for g in ms:
        y0,x0,y1,x1=(float(v)/1000 for v in g)
        boxes.append([min(x0,x1)*W,min(y0,y1)*H,max(x0,x1)*W,max(y0,y1)*H,1.0])
    with lock:
        with open(outp,"a") as f: f.write(json.dumps(dict(canvas=c,item=item,ci=ci,W=W,H=H,model=MODEL,boxes=boxes,raw=txt,err=r.stderr[-500:]),ensure_ascii=False)+"\n")
        print(item,ci,len(boxes),flush=True)
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 2) as ex:
    todo=[k for k in sub if k not in done]
    if "--limit" in sys.argv: todo=todo[:int(sys.argv[sys.argv.index("--limit")+1])]
    list(ex.map(run,todo))
