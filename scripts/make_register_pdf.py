"""位置合わせの確かめの一覧 PDF。pair_class.tsv（aggregate_register_check.py の出力）と check/*.jpg から作る。
1 頁目に集計と見方、2 頁目から分類ごとに、組ごとの画像を並べる。各組は、元の写真の端を並べたもの（左がコマ k+1 の右 35%、右がコマ k の左 35%、間に灰色の線）と、
小さく確かめ用画像（左がコマ k の左端の帯、右がコマ k+1 を合わせて同じ範囲を切り出したもの）。
→ ../data/register/register_check.pdf。使い方: python3 make_register_pdf.py"""
import csv,os,collections
from PIL import Image,ImageDraw,ImageFont
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","register")
FT="/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"; f=lambda s:ImageFont.truetype(FT,s)
W,H=1754,1240; M=50; COLS,ROWS=4,2; TW=(W-2*M)//COLS; TH=(H-150)//ROWS; IH=TH-120; F=0.35
rows=list(csv.DictReader(open(f"{D}/pair_class.tsv"),delimiter="\t"))
SEC=[("ずれ（重なりはあるが、位置合わせが外れた）",lambda r:r["class"]=="ずれ"),
     ("機械の条件は満たすが、Gemini は「隣の紙なし」",lambda r:r["strict"]=="1" and r["class"]=="隣の紙なし"),
     ("機械の条件は満たさないが、Gemini は「重なりあり」",lambda r:r["strict"]=="0" and r["class"]=="重なりあり"),
     ("重なりあり（両方が一致）",lambda r:r["strict"]=="1" and r["class"]=="重なりあり"),
     ("隣の紙なし（両方が一致）",lambda r:r["strict"]=="0" and r["class"]=="隣の紙なし"),
     ("未判定（Gemini の返答なし）",lambda r:r["class"]=="未判定")]
COL={"重なりあり":(0,120,60),"ずれ":(200,0,0),"隣の紙なし":(40,70,160),"未判定":(120,120,120)}
pages=[]
# 1 頁目: 集計
p=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(p); y=M
d.text((M,y),"西南戦争錦絵: 隣り合うコマの位置合わせの確かめ（全 %d 組）"%len(rows),font=f(40),fill="black"); y+=80
C=["重なりあり","ずれ","隣の紙なし","未判定"]; x=collections.Counter((r["strict"],r["class"]) for r in rows)
hd=["機械の条件"]+C+["計"]; cw=[300,220,160,220,160,120]
for i,(lab,s) in enumerate([("",None),("満たす","1"),("満たさない","0"),("計",None)]):
    cells=hd if i==0 else [lab]+[str(x[(s,c)] if s else x[("1",c)]+x[("0",c)]) for c in C]
    if i: cells.append(str(sum(int(v) for v in cells[1:])))
    xx=M
    for c,w in zip(cells,cw): d.text((xx,y),c,font=f(30),fill="black"); xx+=w
    y+=50
y+=30
for s in ["機械の条件: 正しい対応 30 以上・倍率 ±3%・回転 1.5° 以内・重なりが幅の 0〜35%（register_all.py の数値）。",
          "Gemini の判定: 確かめ用画像を Gemini 4 に見せ、背景や物差しではなく錦絵の中身で、同じ絵が同じ位置にあるかを答えさせた。",
          "どの写真も紙 1 枚を中心に撮っている。違いは隣の紙がフレームに入っているか。",
          "  重なりあり = 紙がつながったまま（並べたまま）撮られ、隣の紙の端が写り込む（注釈を隣のコマから写してよい組）",
          "  ずれ = 重なりはあるが位置合わせが外れた　　隣の紙なし = 紙が離して置かれ、隣の紙が写っていない（合っていたのは背景だけ）",
          "画像の見方（各組 2 つ）: 大きい方は元の写真の端どうし。左がコマ k+1 の右 35%、右がコマ k の左 35%（間に灰色の線）。",
          "  両方の端に同じ絵が写っていれば重なりあり。それぞれ別の紙だけが写っていれば 隣の紙なし。",
          "  小さい方は確かめ用画像。白い線の左がコマ k の左端の帯、右がコマ k+1 を合わせて同じ範囲を切り出したもの（黒は写っていない所）。",
          "",
          "頁の並び（数）:"]+[f"  {i+1}. {t}  {sum(1 for r in rows if g(r))} 組" for i,(t,g) in enumerate(SEC)]:
    d.text((M,y),s,font=f(26),fill="black"); y+=44
pages.append(p)
for t,g in SEC:
    sel=[r for r in rows if g(r)]
    for i in range(0,len(sel),COLS*ROWS):
        p=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(p)
        d.text((M,30),f"{t}　{i+1}〜{min(i+COLS*ROWS,len(sel))} / {len(sel)} 組",font=f(32),fill="black")
        for j,r in enumerate(sel[i:i+COLS*ROWS]):
            x0=M+(j%COLS)*TW; y0=100+(j//COLS)*TH
            A=Image.open(f'{D}/../img/{r["item"]}_{r["k"]}.jpg'); B=Image.open(f'{D}/../img/{r["item"]}_{r["k2"]}.jpg')
            a=A.crop((0,0,int(A.width*F),A.height)); b=B.crop((int(B.width*(1-F)),0,B.width,B.height))
            a=a.resize((int(a.width*IH/a.height),IH)); b=b.resize((int(b.width*IH/b.height),IH))
            ck=Image.open(f'{D}/check/{r["item"]}_{r["k"]}-{r["k2"]}.jpg'); k=min(IH/ck.height,TW*0.25/ck.width); ck=ck.resize((int(ck.width*k),int(ck.height*k)))
            sc=min(1,(TW-24-ck.width)/(a.width+b.width+6)); a=a.resize((int(a.width*sc),int(IH*sc))); b=b.resize((int(b.width*sc),int(IH*sc)))
            p.paste(b,(x0,y0)); d.rectangle((x0+b.width,y0,x0+b.width+5,y0+a.height),fill=(160,160,160)); p.paste(a,(x0+b.width+6,y0))
            p.paste(ck,(x0+b.width+a.width+18,y0))
            d.text((x0,y0+IH+4),f'{r["item"]}  {r["k"]}–{r["k2"]}',font=f(22),fill=COL[r["class"]])
            d.text((x0,y0+IH+32),f'{r["class"]}・機械{"○" if r["strict"]=="1" else "×"}・{r["shift"]}',font=f(18),fill="black")
            ln,y1="",y0+IH+56
            for w in r["note"].split():
                if d.textlength(ln+" "+w,font=f(15))>TW-20:
                    if y1>y0+IH+56+2*20: break
                    d.text((x0,y1),ln,font=f(15),fill=(80,80,80)); ln,y1=w,y1+20
                else: ln=(ln+" "+w).strip()
            else: d.text((x0,y1),ln,font=f(15),fill=(80,80,80))
        pages.append(p)
pages[0].save(f"{D}/register_check.pdf",save_all=True,append_images=pages[1:],resolution=150)
print(len(pages),"頁")
