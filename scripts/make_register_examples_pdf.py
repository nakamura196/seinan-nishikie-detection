"""位置合わせの確かめ: 代表的な資料を 1 頁に 1 点ずつ、全コマを元の並び（右からコマ 1, 2, 3 …）で大きく並べた PDF。
コマの間に、その組の分類（pair_class.tsv）を書く。→ ../data/register/register_examples.pdf。使い方: python3 make_register_examples_pdf.py"""
import csv,os
from PIL import Image,ImageDraw,ImageFont
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","register")
FT="/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"; f=lambda s:ImageFont.truetype(FT,s)
EX=[("1990-032-140","重なりあり: 紙がつながったまま撮られ、隣の紙の端が写り込む。局所特徴量でつなげる"),
    ("1990-032-138","重なりあり（別の例）"),
    ("1990-032-025","隣の紙なし: 紙を離して置いて撮っている。機械の条件は満たしたが、合っていたのは背景のマットと物差し"),
    ("1990-032-005","混在: コマ 1・2 はつながったまま、コマ 3 だけ離して撮っている"),
    ("1990-032-094","ずれ: 重なりはあるが、位置合わせが背景に引っ張られて外れた（Gemini の判定）")]
COL={"重なりあり":(0,120,60),"ずれ":(200,0,0),"隣の紙なし":(40,70,160),"未判定":(120,120,120)}
pc={}
for r in csv.DictReader(open(f"{D}/pair_class.tsv"),delimiter="\t"): pc[(r["item"],int(r["k"]))]=r
W,H=1754,1240; M=40; G=70; IH=960
pages=[]
for it,cap in EX:
    ks=sorted({k for (i,k) in pc if i==it}); ks=ks+[ks[-1]+1]
    ims=[Image.open(f"{D}/../img/{it}_{k}.jpg") for k in ks]
    s=min(IH/max(i.height for i in ims),(W-2*M-G*(len(ims)-1))/sum(i.width for i in ims))
    ims=[i.resize((int(i.width*s),int(i.height*s))) for i in ims]
    p=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(p)
    d.text((M,20),it,font=f(36),fill="black"); d.text((M,70),cap,font=f(24),fill="black")
    x=M; y0=130; order=list(zip(ks,ims))[::-1]  # 左に大きい番号
    for n,(k,im) in enumerate(order):
        p.paste(im,(x,y0)); d.text((x,y0+im.height+8),f"コマ {k}",font=f(26),fill="black"); x+=im.width
        if n<len(order)-1:
            r=pc[(it,order[n+1][0])]; c=COL[r["class"]]
            d.rectangle((x+G//2-3,y0,x+G//2+3,y0+IH),fill=c)
            for j,ch in enumerate(r["class"]): d.text((x+G//2-14,y0+IH//2-80+j*32),ch,font=f(28),fill=c,stroke_width=4,stroke_fill="white")
            x+=G
    pages.append(p)
pages[0].save(f"{D}/register_examples.pdf",save_all=True,append_images=pages[1:],resolution=150)
print(len(pages),"頁")
