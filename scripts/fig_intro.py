"""導入の図（2026-11-03）: 錦絵とは何かを一目で示す。(a) 三枚続（1990-032-175）に詞書（青）と札（橙）の位置、(b) 札 1 枚の拡大（西郷隆盛）、(c) 比較用に近世の版本の 1 頁。
機械の検出は描かない。詞書は行ごとの正解枠を、近いもの同士まとめて 1 つの枠にする。題簽（画題の枠）と、その中の文字の枠は除く。
資料: (a) data/img/1990-032-175_{1,2,3}.jpg、正解は data/rows.tsv。(b) IIIF で原寸の範囲を取得 data/fig_intro_src/label_saigo.jpg。
(c) NDL デジタルコレクション『本朝食鑑』（元禄10 [1697]）pid 2569413、canvas 23、公開範囲: PDM（保護期間満了）。data/fig_intro_src/book_c23.jpg。
使い方: python3 fig_intro.py <出力.jpg>"""
import sys,csv,collections
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
plt.rcParams["font.family"]=["Hiragino Sans","DejaVu Sans"]
ITEM="1990-032-175"; BLUE="#0072B2"; ORANGE="#E69F00"
Y0,Y1=210,2440                    # 色票・物差しを除く（画像の画素）
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"):
    if r["item"]==ITEM: rows[r["ci"]].append(r)
def inscr_blocks(L,gap=45,split=False):
    B=[[float(r[k]) for k in "xywh"] for r in L if (r["pred"]=="詞書" or r["text"].startswith("鹿児島へ")) and r["text"] not in ("最期","再乱")]
    B=[[x,y,x+w,y+h] for x,y,w,h in B]; grp=[]
    for b in sorted(B,key=lambda b:b[0]<1500):   # 2 枚目は題簽の下の右の段と左の段を分ける
        if split and False: pass
        hit=[g for g in grp if (not split or (g[0][0]<1500)==(b[0]<1500)) and any(b[0]<c[2]+gap and c[0]<b[2]+gap and b[1]<c[3]+gap and c[1]<b[3]+gap for c in g)]
        new=[b]+[c for g in hit for c in g]; grp=[g for g in grp if g not in hit]+[new]
    return [(min(c[0] for c in g),min(c[1] for c in g),max(c[2] for c in g),max(c[3] for c in g)) for g in grp]
def labels(L):
    out=[]
    for r in L:
        if r["pred"] in("札","札?") and r["text"] not in("西郷","隆盛","弌戦","鹿児島") and not r["text"].startswith("鹿児島へ"):
            x,y,w,h=(float(r[k]) for k in "xywh"); out.append((x,y,x+w,y+h))
    return out
FW,FH=12,4.25
fig=plt.figure(figsize=(FW,FH))
PH=3.5; Y=0.42                    # パネルの高さと下端（インチ）
def ax_at(x,w,h=PH):
    a=fig.add_axes([x/FW,Y/FH,w/FW,h/FH]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_linewidth(0.5); s.set_color("#888")
    return a
# (a) 三枚続
sw=(1676)/(Y1-Y0)*PH; GAP=0.06; wa=3*sw+2*GAP
bar=lambda c:[pe.Stroke(linewidth=2.6,foreground="white"),pe.Normal()]
for i,ci in enumerate("123"):
    a=ax_at(0.1+i*(sw+GAP),sw); im=Image.open(f"../data/img/{ITEM}_{ci}.jpg").convert("RGB"); s=im.size[0]/float(rows[ci][0]["W"])
    a.imshow(im); a.set_xlim(0,im.size[0]); a.set_ylim(Y1,Y0)
    for b in inscr_blocks(rows[ci],split=(ci=='2')):
        a.add_patch(Rectangle((b[0]*s,b[1]*s),(b[2]-b[0])*s,(b[3]-b[1])*s,fill=False,ec="white",lw=2.8))
        a.add_patch(Rectangle((b[0]*s,b[1]*s),(b[2]-b[0])*s,(b[3]-b[1])*s,fill=False,ec=BLUE,lw=1.3))
    for b in labels(rows[ci]):
        a.add_patch(Rectangle((b[0]*s,b[1]*s),(b[2]-b[0])*s,(b[3]-b[1])*s,fill=False,ec="white",lw=2.4))
        a.add_patch(Rectangle((b[0]*s,b[1]*s),(b[2]-b[0])*s,(b[3]-b[1])*s,fill=False,ec=ORANGE,lw=1.1))
    if ci=="1":   # (b) で拡大する札に印
        x,y,w,h=(float(rows["1"][[r["text"] for r in rows["1"]].index("西郷隆盛")][k]) for k in "xywh")
        a.annotate("",xy=((x+w)*s+4,(y+h*0.5)*s),xytext=((x+w)*s+95,(y+h*0.5)*s),arrowprops=dict(arrowstyle="-|>",color="black",lw=1.2,path_effects=[pe.Stroke(linewidth=3,foreground="white"),pe.Normal()]))
fig.text(0.1/FW,(Y+PH+0.08)/FH,"(a) triptych (3 sheets, 1990-032-175)",fontsize=10.5,va="bottom")
# 凡例
lx=0.1; ly=Y-0.2
for c,t,dx in ((BLUE,"inscription",0),(ORANGE,"label",1.45)):
    fig.add_artist(Rectangle(((lx+dx)/FW,(ly-0.07)/FH),0.28/FW,0.14/FH,fill=False,ec=c,lw=1.4,transform=fig.transFigure))
    fig.text((lx+dx+0.36)/FW,ly/FH,t,fontsize=10,va="center")
# (b) 札の拡大
lab=Image.open("../data/fig_intro_src/label_saigo.jpg").convert("RGB"); wb=PH*lab.size[0]/lab.size[1]
xb=0.1+wa+0.25; a=ax_at(xb,wb); a.imshow(lab,interpolation="lanczos")
for sp in a.spines.values(): sp.set_color(ORANGE); sp.set_linewidth(1.4)
fig.text(xb/FW,(Y+PH+0.08)/FH,"(b) enlarged label",fontsize=10.5,va="bottom")
fig.text((xb+wb/2)/FW,(Y-0.2)/FH,"西郷隆盛 (personal name)",fontsize=9.5,ha="center",va="center")
# (c) 版本の 1 頁
bk=Image.open("../data/fig_intro_src/book_c23.jpg").convert("RGB"); bk=bk.crop((1050,100,1810,1310)); wc=PH*bk.size[0]/bk.size[1]
xc=xb+wb+0.45; a=ax_at(xc,wc); a.imshow(bk)
fig.text(xc/FW,(Y+PH+0.08)/FH,"(c) printed book page",fontsize=10.5,va="bottom")
print("width used",xc+wc)
plt.savefig(sys.argv[1],dpi=200,pad_inches=0.05,facecolor="white",pil_kwargs=dict(quality=90))
