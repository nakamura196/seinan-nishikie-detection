"""主設定 C（main_cv_all.jsonl）の注釈外の枠（正解の占める割合 <0.1）から無作為に選び、周りを含めて切り出して一覧にする（手検査用）。検出の枠は青、正解の注釈は赤の細線。
使い方: python3 outside_sheet_main.py <件数> <出力の頭> [1 枚あたりの件数]
2026-11-03: 環境変数 DET で検出の jsonl を替えられる（Gemini 4 なら DET=../data/gemini4_det.jsonl）。倍率は main_cv_all.jsonl の同じ面の値を使う
→ <出力の頭>-<k>.jpg（番号付きの一覧）と <出力の頭>.tsv（番号・面・枠・確信度、判定欄つき）。全体の件数も表示する"""
import sys,os,json,csv,random,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
K=8; OUT=0.1
N,outp=int(sys.argv[1]),sys.argv[2]; per=int(sys.argv[3]) if len(sys.argv)>3 else 30
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): rows[(r["item"],r["ci"])].append(r)
cand=[]
SC={(x["item"],str(x["ci"])):x["scale"] for x in map(json.loads,open("../data/main_cv_all.jsonl"))}
for d in map(json.loads,open(os.environ.get("DET","../data/main_cv_all.jsonl"))):
    key=(d["item"],str(d["ci"])); G=np.zeros((d["H"]//K+1,d["W"]//K+1),bool)
    for r in rows[key]:
        x,y,w,h=(float(r[c])/K for c in "xywh"); G[max(int(y),0):int(y+h)+1,max(int(x),0):int(x+w)+1]=True
    for b in d["boxes"]:
        m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
        if m.size==0 or m.mean()<OUT: cand.append((key[0],key[1],SC[key],[float(x) for x in b]))
    
random.Random(20261102).shuffle(cand); pick=cand[:N]
font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",28); T=360; cols=5
with open(outp+".tsv","w") as f:
    f.write("no\titem\tci\tx1\ty1\tx2\ty2\tconf\tjudge\n")
    for s0 in range(0,len(pick),per):
        part=pick[s0:s0+per]; sheet=Image.new("RGB",(cols*T,((len(part)+cols-1)//cols)*(T+36)),"white"); dr=ImageDraw.Draw(sheet)
        for j,(item,ci,s,b) in enumerate(part):
            n=s0+j+1; im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB")
            x1,y1,x2,y2=(c/s for c in b[:4]); m=max(max(x2-x1,y2-y1)*0.6+40,120); D=ImageDraw.Draw(im)
            for r in rows[(item,ci)]:
                rx,ry,rw,rh=(float(r[c])/s for c in "xywh"); D.rectangle([rx,ry,rx+rw,ry+rh],outline=(230,0,0),width=2)
            D.rectangle([x1,y1,x2,y2],outline=(0,80,255),width=3)
            crop=im.crop((int(x1-m),int(y1-m),int(x2+m),int(y2+m)))
            crop=crop.resize((T,int(T*crop.size[1]/crop.size[0]))) if crop.size[0]>=crop.size[1] else crop.resize((int(T*crop.size[0]/crop.size[1]),T)); X,Y=(j%cols)*T,(j//cols)*(T+36)
            sheet.paste(crop,(X,Y+36)); dr.text((X+6,Y+2),f"{n}  {b[4]:.2f}",fill="black",font=font)
            f.write(f"{n}\t{item}\t{ci}\t{b[0]:.0f}\t{b[1]:.0f}\t{b[2]:.0f}\t{b[3]:.0f}\t{b[4]:.3f}\t\n")
        sheet.save(f"{outp}-{s0//per+1}.jpg",quality=85)
print(len(cand),"件の外の枠から",len(pick),"件")
