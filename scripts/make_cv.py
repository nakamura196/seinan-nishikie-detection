"""資料単位の 5 分割交差検証の分け方を作る（全 739 面に、学習に使っていないモデルの予測を付けるため）。
資料を seed 20261030 で並べ替え、i % 5 を分割の番号にする。分割 k を評価にし、残り 4 分割の資料のうち 1/8 を検証（全体の 10%）、残りを学習にする。
画像と正解は make_yolo.py が作った data/yolo/{images,labels}/<split>/ をそのまま使い、一覧の txt で指す（YOLO は /images/ を /labels/ に読み替えて正解を探す）。
使い方: python3 make_cv.py [data/yolo の絶対パス（既定はこの手元の位置）]
出力: data/cv.tsv（item, ci, canvas, fold）, data/cv/fold<k>_{train,val,test}.txt, data/cv/fold<k>.yaml"""
import sys,csv,random,os,collections
D=os.path.abspath(sys.argv[1] if len(sys.argv)>1 else "../data/yolo")
faces=list(csv.DictReader(open("../data/split.tsv"),delimiter="\t"))
items=sorted({r["item"] for r in faces}); random.Random(20261030).shuffle(items)
fold={it:i%5 for i,it in enumerate(items)}
os.makedirs("../data/cv",exist_ok=True)
with open("../data/cv.tsv","w") as f:
    f.write("item\tci\tcanvas\tfold\n")
    for r in faces: f.write(f"{r['item']}\t{r['ci']}\t{r['canvas']}\t{fold[r['item']]}\n")
for k in range(5):
    rest=[it for it in items if fold[it]!=k]; random.Random(20261030+k).shuffle(rest)
    val=set(rest[:round(len(rest)/8)])
    part=lambda it:"test" if fold[it]==k else "val" if it in val else "train"
    L=collections.defaultdict(list)
    for r in faces: L[part(r["item"])].append(f"{D}/images/{r['split']}/{r['item']}_{r['ci']}.jpg")
    for s,v in L.items(): open(f"../data/cv/fold{k}_{s}.txt","w").write("\n".join(v)+"\n")
    C=os.path.abspath("../data/cv")
    open(f"../data/cv/fold{k}.yaml","w").write(f"path: {C}\ntrain: {C}/fold{k}_train.txt\nval: {C}/fold{k}_val.txt\ntest: {C}/fold{k}_test.txt\nnames:\n  0: text\n")
    print(f"fold{k}", {s:len(v) for s,v in L.items()}, "資料 評価",sum(1 for it in items if fold[it]==k),"検証",len(val))
