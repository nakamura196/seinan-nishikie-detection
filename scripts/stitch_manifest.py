"""位置合わせの結果（../data/register/pairs.tsv）から、続き物の錦絵を 1 つのキャンバスにつないで見せる IIIF Presentation 3 のマニフェストを作る。
各コマの画像を target の #xywh で並べる（コマ k+1 の原点 = コマ k の原点 + ずらし。倍率と回転は 1・0 とみなす）。
重なりのあるコマの本文は Choice: (1) 左の重なりを切った画像（ImageApiSelector の範囲指定、既定）、(2) 元の画像（深く拡大できる）。
2 枚目以降のキャンバスに各コマを元のまま置く。使い方: python3 stitch_manifest.py <資料> <出力.json> <マニフェストの URL>"""
import sys,csv,json,urllib.parse
item,outp,mid=sys.argv[1:4]
rows=[r for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t") if r["item"]==item]
C={}
for r in rows: C[int(r["ci"])]=(r["img"],int(r["W"]),int(r["H"]))
S={(d["item"],int(d["ci"])):d["scale"] for d in map(json.loads,open("../data/main_cv_all.jsonl")) if d["item"]==item}
P={int(r["k"]):r for r in csv.DictReader(open("../data/register/pairs.tsv"),delimiter="\t") if r["item"]==item}
ks=sorted(C); pos={ks[0]:(0.0,0.0)}
for k in ks[:-1]:
    s=S[(item,k)]; x,y=pos[k]; pos[k+1]=(x+float(P[k]["tx"])*s, y+float(P[k]["ty"])*s)
mx=min(p[0] for p in pos.values()); my=min(p[1] for p in pos.values())
pos={k:(round(x-mx),round(y-my)) for k,(x,y) in pos.items()}
CW=max(pos[k][0]+C[k][1] for k in ks); CH=max(pos[k][1]+C[k][2] for k in ks)
def svc(img): return urllib.parse.quote(urllib.parse.unquote(img),safe=":/")
def full(k):
    b=svc(C[k][0]); return {"id":f"{b}/full/max/0/default.jpg","type":"Image","format":"image/jpeg","width":C[k][1],"height":C[k][2],
        "label":{"ja":["元の画像"]},"service":[{"id":b,"type":"ImageService3","profile":"level2"}]}
items=[]; anns=[]
for k in ks:
    x,y=pos[k]; _,W,H=C[k]
    cut=round(float(P[k]["overlap_px"])*S[(item,k)]) if k in P else 0
    # 範囲は ImageApiSelector で書く（cookbook 0299）。URL に範囲を書くと、Kasane などは範囲を読まず画像全体を枠に押し込む
    trim={"type":"SpecificResource","source":full(k),"selector":{"type":"ImageApiSelector","region":f"{cut},0,{W-cut},{H}"},
        "label":{"ja":["重なりを切った画像" if cut else "画像"]}}
    tgt=f"{mid}/canvas/stitched#xywh={x+cut},{y},{W-cut},{H}"
    body={"type":"Choice","items":[trim,full(k)]}
    anns.append({"id":f"{mid}/anno/s{k}","type":"Annotation","motivation":"painting","target":tgt,"body":body})
# 注釈を重ねる: 各コマの注釈の座標にコマの位置を足す。重なりを切った帯に中心がある注釈は、隣のコマにも同じものがあるので落とす
notes=[]; drop=0
for r in rows:
    k=int(r["ci"]); x,y=pos[k]; cut=round(float(P[k]["overlap_px"])*S[(item,k)]) if k in P else 0
    ax,ay,aw,ah=(float(r[c]) for c in "xywh")
    if ax+aw/2<cut: drop+=1; continue
    notes.append({"id":f"{mid}/note/{r['id']}","type":"Annotation","motivation":"commenting",
                  "body":{"type":"TextualBody","value":r["text"],"language":"ja","format":"text/plain"},
                  "target":f"{mid}/canvas/stitched#xywh={round(x+ax)},{round(y+ay)},{round(aw)},{round(ah)}"})
json.dump({"@context":"http://iiif.io/api/presentation/3/context.json","id":f"{mid}-notes.json","type":"AnnotationPage","items":notes},
          open(outp.replace(".json","-notes.json"),"w"),ensure_ascii=False)
print("注釈",len(notes),"件を重ねた（重なりの帯で落とした",drop,"件）")
can=[{"id":f"{mid}/canvas/stitched","type":"Canvas","label":{"ja":["つないだ表示"]},"width":CW,"height":CH,
      "items":[{"id":f"{mid}/page/stitched","type":"AnnotationPage","items":anns}],
      "annotations":[{"id":mid.replace(".json","-notes.json"),"type":"AnnotationPage"}]}]
for k in ks:
    _,W,H=C[k]; can.append({"id":f"{mid}/canvas/p{k}","type":"Canvas","label":{"ja":[f"コマ {k}"]},"width":W,"height":H,
      "items":[{"id":f"{mid}/page/p{k}","type":"AnnotationPage","items":[{"id":f"{mid}/anno/p{k}","type":"Annotation","motivation":"painting","target":f"{mid}/canvas/p{k}","body":full(k)}]}]})
man={"@context":"http://iiif.io/api/presentation/3/context.json","id":mid,"type":"Manifest","label":{"ja":[f"{item}（コマをつないだ試作）"]},
     "viewingDirection":"right-to-left","items":can}
json.dump(man,open(outp,"w"),ensure_ascii=False,indent=1); print("キャンバス",CW,CH,"位置",pos)
