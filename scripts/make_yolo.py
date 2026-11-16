"""正解（rows.tsv）から、1 種類「文字の範囲」の YOLO 形式のデータを作る。
資料（item）単位で 学習 70% / 検証 10% / 評価 20% に分ける（同じ資料の面は同じ側に入る）。seed 20261030。
画像は data/img/<item>_<ci>.jpg（長辺 2560px）へのリンク。枠は面の幅・高さで割った値なので縮尺に依らない。
出力: data/yolo/{images,labels}/{train,val,test}/, data/yolo/seinan.yaml, data/split.tsv"""
import csv,collections,random,os
rows=list(csv.DictReader(open(os.environ.get("ROWS","../data/rows.tsv")),delimiter="\t"))  # 環境変数 ROWS で正解を替えられる
byf=collections.defaultdict(list)
for r in rows: byf[(r["item"],r["ci"])].append(r)
items=sorted({k[0] for k in byf}); random.Random(20261030).shuffle(items)
n=len(items); sp={}
for i,it in enumerate(items): sp[it]="train" if i<n*0.7 else "val" if i<n*0.8 else "test"
D=os.path.abspath("../data/yolo")
for s in ("train","val","test"):
    os.makedirs(f"{D}/images/{s}",exist_ok=True); os.makedirs(f"{D}/labels/{s}",exist_ok=True)
cnt=collections.Counter()
with open("../data/split.tsv","w") as f:
    f.write("item\tci\tcanvas\tsplit\n")
    for (it,ci),L in byf.items():
        s=sp[it]; name=f"{it}_{ci}"; src=os.path.abspath(f"../data/img/{name}.jpg")
        dst=f"{D}/images/{s}/{name}.jpg"
        if not os.path.exists(dst): os.symlink(src,dst)
        W,H=float(L[0]["W"]),float(L[0]["H"])
        with open(f"{D}/labels/{s}/{name}.txt","w") as g:
            for r in L:
                x,y,w,h=(float(r[k]) for k in "xywh")
                x0,y0=max(0,x),max(0,y); x1,y1=min(W,x+w),min(H,y+h)
                if x1<=x0 or y1<=y0: continue
                g.write(f"0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n")
        f.write(f"{it}\t{ci}\t{L[0]['canvas']}\t{s}\n"); cnt[s,"face"]+=1; cnt[s,"box"]+=len(L)
open(f"{D}/seinan.yaml","w").write(f"path: {D}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: text\n")
for s in ("train","val","test"): print(s,"資料",sum(1 for v in sp.values() if v==s),"面",cnt[s,"face"],"範囲",cnt[s,"box"])
