"""比較基準: Gemma 4（26B-A4B-it、Q8_0（ggml-org）、llama-server）に錦絵の各面の文字の範囲を JSON で答えさせ、範囲を JSONL に出す。
座標は Gemma の流儀（box_2d = [ymin, xmin, ymax, xmax]、0〜NORM に正規化）で受け取り、原寸に戻す。得点は無いので 1.0 とする。
画像は ../data/img（長辺 2560px）を長辺 MAXS に縮めて送る。温度 0。出力の形は ndlkoten_det.jsonl と同じ。
使い方: python3 run_gemma4_det.py [件数上限] [出力ファイル名]（llama-server を port 8093 で起動しておく）"""
import sys,os,json,csv,base64,io,re,urllib.request
from PIL import Image
PORT=8093; MAXS=int(os.environ.get("MAXS","1536")); NORM=float(os.environ.get("NORM","1000"))
PROMPT=("This is a Japanese woodblock print. Detect every piece of written text in the image: each column or block of "
        "the narrative inscription, every short name label (cartouche) beside figures, the title, signatures and seals with text. "
        "Return only a JSON list, one entry per text region, in the form "
        '[{"box_2d": [ymin, xmin, ymax, xmax], "label": "text"}], with coordinates normalized to 0-1000.')
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","data")
rows=list(csv.DictReader(open(os.path.join(OUT,"rows.tsv")),delimiter="\t"))
canv={}
for r in rows: canv.setdefault(r["canvas"],(int(r["W"]),int(r["H"]),r["item"],r["ci"]))
lim=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
outp=os.path.join(OUT,sys.argv[2] if len(sys.argv)>2 else "gemma4_det.jsonl")
done={json.loads(l)["canvas"] for l in open(outp)} if os.path.exists(outp) else set()
n=0
with open(outp,"a") as f:
    for c,(W,H,item,ci) in canv.items():
        if c in done: continue
        if n>=lim: break
        im=Image.open(f"{OUT}/img/{item}_{ci}.jpg").convert("RGB"); im.thumbnail((MAXS,MAXS))
        buf=io.BytesIO(); im.save(buf,"JPEG",quality=92); b64=base64.b64encode(buf.getvalue()).decode()
        body={"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}},{"type":"text","text":PROMPT}]}],
              "temperature":0,"max_tokens":8192}
        req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
        txt=json.load(urllib.request.urlopen(req,timeout=1800))["choices"][0]["message"]["content"]
        boxes=[]
        for m in re.finditer(r'"box_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',txt):
            y0,x0,y1,x1=(float(v)/NORM for v in m.groups())
            boxes.append([min(x0,x1)*W,min(y0,y1)*H,max(x0,x1)*W,max(y0,y1)*H,1.0])
        f.write(json.dumps(dict(canvas=c,item=item,ci=ci,W=W,H=H,scale=W/im.size[0],boxes=boxes,raw=txt))+"\n"); f.flush(); n+=1
        print(n,item,ci,len(boxes),flush=True)
