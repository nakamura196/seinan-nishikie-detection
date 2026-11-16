"""注釈の外に出た枠（正解に 10% 未満しか重ならない枠）を無作為に選び、周りを含めて切り出して 1 枚に並べる（目で中身を確かめるため）。
使い方: python3 outside_sheet.py <見せ方 例 full / full+t2> <しきい値> <件数> <出力の頭>
→ <出力の頭>.jpg（番号付きの一覧）と <出力の頭>.tsv（番号・面・枠・確信度、判定を書き込む欄つき）"""
import sys,json,csv,random,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys_argv=sys.argv[1:]; sys.argv=["x"]
exec(open("sweep_tiles.py").read().split("acc=collections")[0])
v,th,N,outp=sys_argv[0],float(sys_argv[1]),int(sys_argv[2]),sys_argv[3]
cand=[]
for k in range(5):
    for d in map(json.loads,open(f"../data/tiles/fold{k}.jsonl")):
        if d["role"]!="test": continue
        L=rows[(d["item"],d["ci"])]; W,H=d["W"]//K+1,d["H"]//K+1; G=np.zeros((H,W),bool)
        for r in L:
            x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        for b in nms([b for p in VARS[v] for b in d[p]]):
            if b[4]<th: continue
            m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]
            if m.size==0 or m.mean()<0.1: cand.append((d["item"],d["ci"],d["W"],[float(x) for x in b]))
random.Random(20261031).shuffle(cand); pick=cand[:N]
font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",28); T=360; cols=5
sheet=Image.new("RGB",(cols*T,((N+cols-1)//cols)*(T+36)),"white"); dr=ImageDraw.Draw(sheet)
with open(outp+".tsv","w") as f:
    f.write("no\titem\tci\tx1\ty1\tx2\ty2\tconf\tjudge\n")
    for n,(item,ci,W0,b) in enumerate(pick,1):
        im=Image.open(f"../data/img/{item}_{ci}.jpg").convert("RGB"); s=W0/im.size[0]
        x1,y1,x2,y2=(c/s for c in b[:4]); m=max(x2-x1,y2-y1)*0.6+40
        crop=im.crop((int(x1-m),int(y1-m),int(x2+m),int(y2+m))); ImageDraw.Draw(crop).rectangle([m,m,m+x2-x1,m+y2-y1],outline=(0,80,255),width=3)
        crop.thumbnail((T,T)); X,Y=((n-1)%cols)*T,((n-1)//cols)*(T+36)
        sheet.paste(crop,(X,Y+36)); dr.text((X+6,Y+2),f"{n}  {b[4]:.2f}",fill="black",font=font)
        f.write(f"{n}\t{item}\t{ci}\t{b[0]:.0f}\t{b[1]:.0f}\t{b[2]:.0f}\t{b[3]:.0f}\t{b[4]:.3f}\t\n")
sheet.save(outp+".jpg",quality=88); print(len(cand),"件の外の枠から",len(pick),"件")
