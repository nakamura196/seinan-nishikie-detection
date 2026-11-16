"""Gemini の確かめ（gemini_check.jsonl）を pairs.tsv と突き合わせ、組を分ける。
  重なりあり   : overlap かつ aligned（紙がつながったまま撮られ、隣の紙の端が写り込んでいる。注釈を写してよい組）
  ずれ         : overlap だが aligned でない（重なりはあるが位置合わせが外れた）
  隣の紙なし  : overlap でない（どの写真も紙 1 枚ずつ撮っている。紙が離して置かれ、隣の紙が写っていない。対応は背景に合っていた）
  未判定       : まだ確かめていない、または返答が読めなかった
厳しい条件（register_all の数値: 正しい対応 30 以上・倍率 ±3%・回転 1.5° 以内・重なり 0〜35%）とのクロス表も出す。
→ ../data/register/pair_class.tsv（組ごと）、item_class.tsv（資料ごと: 組の分類の内訳）。使い方: python3 aggregate_register_check.py"""
import csv,json,collections,os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","register")
ans={}
for l in open(f"{D}/gemini_check.jsonl"):
    r=json.loads(l); ans[r["file"][:-4]]=r["answer"]
def strict(r):
    try: return int(r["inliers"])>=30 and abs(float(r["scale"])-1)<=0.03 and abs(float(r["angle"]))<=1.5 and 0<=float(r["overlap_frac"])<=0.35
    except ValueError: return False
def cls(a):
    if not a: return "未判定"
    if not a.get("overlap"): return "隣の紙なし"
    return "重なりあり" if a.get("aligned") else "ずれ"
rows=list(csv.DictReader(open(f"{D}/pairs.tsv"),delimiter="\t"))
out=open(f"{D}/pair_class.tsv","w"); out.write("item\tk\tk2\tstrict\tclass\tshift\tnote\n")
x=collections.Counter(); per=collections.defaultdict(collections.Counter)
for r in rows:
    a=ans.get(f'{r["item"]}_{r["k"]}-{r["k2"]}'); c=cls(a); s=strict(r)
    x[(s,c)]+=1; per[r["item"]][c]+=1
    out.write(f'{r["item"]}\t{r["k"]}\t{r["k2"]}\t{int(s)}\t{c}\t{(a or {}).get("shift","")}\t{(a or {}).get("note","")}\n')
out.close()
C=["重なりあり","ずれ","隣の紙なし","未判定"]
print("組の数",len(rows)); print("厳しい条件\t"+"\t".join(C)+"\t計")
for s in (True,False): print(("満たす" if s else "満たさない")+"\t"+"\t".join(str(x[(s,c)]) for c in C)+f"\t{sum(x[(s,c)] for c in C)}")
print("計\t"+"\t".join(str(sum(x[(s,c)] for s in (True,False))) for c in C))
io=open(f"{D}/item_class.tsv","w"); io.write("item\t"+"\t".join(C)+"\tkind\n"); k=collections.Counter()
for it,p in sorted(per.items()):
    j=[c for c in C[:3] if p[c]]; kind="未判定" if not j else ("重なりあり" if j==["重なりあり"] else "隣の紙なし" if j==["隣の紙なし"] else "混在")
    k[kind]+=1; io.write(it+"\t"+"\t".join(str(p[c]) for c in C)+f"\t{kind}\n")
print("資料ごと（判定済みの組だけで）:",dict(k))
