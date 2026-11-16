"""論文の結果の節の数字を、主設定と比較の設定についてまとめて出す。
設定: NDL / A 前のモデル（全体の画像だけで学習、全体の画像だけ、0.25）/ B 前のモデル + 2×2 で隙間を埋める（0.15）/
C 片でも学習したモデル + 2×2 で隙間を埋める（0.15、主設定）。すべて評価の面（test、全 739 面）。
出すもの: 種類別（拾えた・覆い 0.9 以上・覆い 0）、字数別、難しい面、枠の数と注釈の外の枠、分割ごとの範囲、
NDL と取りこぼしの重なり、取りこぼしの集中、資料単位のブートストラップ（2,000 回、seed 20261101）、覆いの線の感度。
使い方: python3 paper_tables.py（環境変数 ROWS で正解のファイルを替えられる。ROWS=../data/rows_ext.tsv のときは NDL の覆いを ndlkoten_cov_fill0.2_ext.tsv から読み、C の覆いを main_cv_cov_fill0.2_ext.tsv に書く。主設定の枠を ../data/main_cv_all.jsonl、覆いを ../data/main_cv_cov_fill0.2.tsv にも書く）"""
import sys,json,csv,collections
sys.argv=["x"]; import numpy as np
SUF="_ext" if "rows_ext" in __import__("os").environ.get("ROWS","") else ""
exec(open("sweep_tiles.py").read().split("acc=collections")[0])
base={r["id"]:float(r["cov"]) for r in csv.DictReader(open(f"../data/ndlkoten_cov_fill0.2{SUF}.tsv"),delimiter="\t")}
def fillin(full,t2,ov=0.5):
    ar=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1])); out=list(full)
    for b in nms(t2):
        if all(max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))<ov*ar(b) for o in full): out.append(b)
    return out
CF={"A":("","full",0.25),"B":("","fill",0.15),"C":("t2train_","fill",0.15)}
dump=open("../data/main_cv_all.jsonl","w")  # 主設定 C の枠（fig3.py 用）
cov={n:{} for n in CF}; box={n:[0,0] for n in CF}; fold={}; img={}
for n,(tag,mode,th) in CF.items():
    for k in range(5):
        for d in map(json.loads,open(f"../data/tiles/{tag}fold{k}.jsonl")):
            if d["role"]!="test": continue
            L=rows[(d["item"],d["ci"])]; W,H=d["W"]//K+1,d["H"]//K+1; G=np.zeros((H,W),bool)
            for r in L:
                x,y,w,h=(float(r[c])/K for c in "xywh"); G[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
            if mode=="full": bs=[b for b in nms(d["full"]) if b[4]>=th]
            else: bs=fillin(nms([b for b in d["full"] if b[4]>=th]),[b for b in d["t2"] if b[4]>=th])
            if n=="C":
                from PIL import Image; s=d["W"]/Image.open(f"../data/img/{d['item']}_{d['ci']}.jpg").size[0]
                dump.write(json.dumps({"item":d["item"],"ci":str(d["ci"]),"W":d["W"],"H":d["H"],"scale":s,"boxes":[list(map(float,b)) for b in bs]})+"\n")
            P=np.zeros_like(G)
            for b in bs:
                m=G[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]; box[n][0]+=1
                if m.size==0 or m.mean()<0.1: box[n][1]+=1
                if m.size==0 or m.mean()<FILL: continue
                P[int(b[1]/K):int(b[3]/K)+1,int(b[0]/K):int(b[2]/K)+1]=True
            for r in L:
                x,y,w,h=(float(r[c])/K for c in "xywh"); m=P[int(y):int(y+h)+1,int(x):int(x+w)+1]
                cov[n][r["id"]]=float(m.mean()) if m.size else 0.0; fold[r["id"]]=k; img[r["id"]]=(r["item"],r["ci"])
dump.close()
with open(f"../data/main_cv_cov_fill0.2{SUF}.tsv","w") as f:
    f.write("id\tcov\n"); [f.write(f"{i}\t{v:.3f}\n") for i,v in cov["C"].items()]
cov={"NDL":base,**cov}; S=list(cov)
R={r["id"]:r for r in csv.DictReader(open(__import__("os").environ.get("ROWS","../data/rows.tsv")),delimiter="\t")}; ids=sorted(R)
pc=lambda n,v,t=0.5:100*np.mean([cov[n][i]>=t for i in v])
def L(t): n=len(t); return "1" if n==1 else "2-3" if n<=3 else "4-10" if n<=10 else "11-30" if n<=30 else "31+"
print("== 種類別: 範囲数 / 拾えた / 覆い 0.9 以上 / 覆い 0 （NDL A B C の順）")
for g,v in [("全体",ids),*[(c,[i for i in ids if R[i]["pred"]==c]) for c in ("詞書","札","札?")]]:
    print(f"  {g} n={len(v)} 拾えた",*[f"{pc(n,v):.1f}" for n in S],"| 0.9",*[f"{pc(n,v,0.9):.1f}" for n in S],"| 0",*[f"{100*np.mean([cov[n][i]==0 for i in v]):.1f}" for n in S])
print("== 字数別 拾えた（NDL A B C）")
for b in ("1","2-3","4-10","11-30","31+"):
    v=[i for i in ids if L(R[i]["text"])==b]; print(f"  {b} n={len(v)}",*[f"{pc(n,v):.1f}" for n in S])
byimg=collections.defaultdict(list)
for i in ids: byimg[img[i]].append(i)
hard6=[i for k,v in byimg.items() if len(v)>=2 and np.mean([base[x]>=.5 for x in v])<.5 for i in v]
top30=[i for k in sorted(byimg,key=lambda k:-sum(base[x]<.5 for x in byimg[k]))[:30] for i in byimg[k]]
print("== 難しい面（NDL A B C）")
for g,v in [("難6面",hard6),("上位30面",top30),("格子 2012-06-03 p1",byimg[("2012-06-03","1")])]: print(f"  {g} n={len(v)}",*[f"{pc(n,v):.1f}" for n in S])
print("== 取りこぼし数・枠の数・注釈の外の枠（重なり 10% 未満）")
for n in S:
    miss=[i for i in ids if cov[n][i]<0.5]; per=collections.Counter(img[i] for i in miss)
    s=f"  {n} 取りこぼし {len(miss)}、面 {len(per)}、上位 30 面に {100*sum(c for _,c in per.most_common(30))/max(1,len(miss)):.0f}%"
    if n in box: s+=f"、枠 {box[n][0]}、外 {box[n][1]}（{100*box[n][1]/box[n][0]:.1f}%）"
    print(s)
print("== 分割ごとの拾えた（NDL / C）")
for k in range(5):
    v=[i for i in ids if fold[i]==k]; print(f"  fold{k} n={len(v)}",f"{pc('NDL',v):.1f}",f"{pc('C',v):.1f}")
mN={i for i in ids if base[i]<.5}; mC={i for i in ids if cov["C"][i]<.5}
print(f"== NDL の取りこぼし {len(mN)} のうち C が拾う {len(mN-mC)}、C だけが落とす {len(mC-mN)}、両方 {len(mN&mC)}")
print("== 覆いの線の感度（NDL A B C）")
for t in (0.3,0.5,0.7,0.9): print(f"  {t}",*[f"{pc(n,ids,t):.1f}" for n in S])
item={i:R[i]["item"] for i in ids}; byi=collections.defaultdict(list)
for i in ids: byi[item[i]].append(i)
its=sorted(byi); hit={n:{it:np.array([cov[n][i]>=0.5 for i in byi[it]]) for it in its} for n in S}
rng=np.random.default_rng(20261101); res=collections.defaultdict(list)
for _ in range(2000):
    s=rng.choice(its,len(its)); tot=sum(len(byi[it]) for it in s); v={n:sum(hit[n][it].sum() for it in s)/tot for n in S}
    for n in S: res[n].append(v[n])
    for a,b in (("A","NDL"),("C","NDL"),("C","A"),("C","B")): res[f"{a} − {b}"].append(v[a]-v[b])
print(f"== 資料 {len(its)} 件のブートストラップ 2,000 回（平均 [95% 区間]）")
for n,v in res.items(): print(f"  {n}: {100*np.mean(v):.1f} [{100*np.percentile(v,2.5):.1f}, {100*np.percentile(v,97.5):.1f}]")
