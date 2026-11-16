"""eval_coverage.py が書いた 2 つの tsv（NDL と学習したモデル、同じ面）を範囲ごとに突き合わせて比べる。
拾えた = 覆い >= 0.5。種類別・字数別・「難しい面」（NDL が 2 範囲以上の面で半分以上落とす面）・NDL の取りこぼしが多い面の上位 30 面での値を出す。
使い方: python3 compare_cov.py <NDL の tsv> <モデルの tsv>"""
import sys,csv,collections
A={r["id"]:r for r in csv.DictReader(open(sys.argv[1]),delimiter="\t")}
B={r["id"]:r for r in csv.DictReader(open(sys.argv[2]),delimiter="\t")}
ids=[i for i in A if i in B]
ok=lambda R,i: float(R[i]["cov"])>=0.5
def line(name,S):
    n=len(S)
    if not n: return
    a=sum(ok(A,i) for i in S); b=sum(ok(B,i) for i in S)
    za=sum(float(A[i]["cov"])==0 for i in S); zb=sum(float(B[i]["cov"])==0 for i in S)
    print(f"{name}\tn={n}\tNDL {a/n:.3f} → model {b/n:.3f}\t覆い0 {za/n:.3f} → {zb/n:.3f}\t落とした {n-a} → {n-b}")
print("面",len({A[i]["canvas"] for i in ids}))
line("全体",ids)
for k in ["詞書","札","札?"]: line(k,[i for i in ids if A[i]["pred"]==k])
L=lambda i: len(A[i]["text"])
for lo,hi,nm in [(1,1,"1字"),(2,3,"2-3字"),(4,10,"4-10字"),(11,30,"11-30字"),(31,10**9,"31字以上")]:
    line(nm,[i for i in ids if lo<=L(i)<=hi])
byc=collections.defaultdict(list)
for i in ids: byc[A[i]["canvas"]].append(i)
hard=[c for c,v in byc.items() if len(v)>=2 and sum(ok(A,i) for i in v)/len(v)<0.5]
line(f"難しい面 {len(hard)} 面",[i for c in hard for i in byc[c]])
top=sorted(byc,key=lambda c:-sum(not ok(A,i) for i in byc[c]))[:30]
line("NDL の取りこぼし上位 30 面",[i for c in top for i in byc[c]])
na=sum(not ok(A,i) for i in ids); print(f"上位 30 面が NDL の取りこぼしに占める割合 {sum(not ok(A,i) for c in top for i in byc[c])/na:.3f}")
x=collections.Counter((ok(A,i),ok(B,i)) for i in ids)
print(f"NDL だけ落とした {x[False,True]}  モデルだけ落とした {x[True,False]}  両方 {x[False,False]}")
print("難しい面の内訳:")
for c in sorted(hard,key=lambda c:-len(byc[c])):
    v=byc[c]; print(f"  {A[v[0]]['item']}\tn={len(v)}\tNDL {sum(ok(A,i) for i in v)/len(v):.2f} → {sum(ok(B,i) for i in v)/len(v):.2f}")
