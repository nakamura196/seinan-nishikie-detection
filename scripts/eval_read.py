"""read_regions.py の読みを、正解の文字と突き合わせる（拾えた枠が実際に読めるか）。
枠は検出の評価と同じく、面積の 20% 以上が正解の範囲にかかるものだけを使う（1/8 格子）。
(1) 面ごとの文字の再現率: 正解の文字の多重集合のうち、使った枠の読みに含まれた割合（並び順は見ない）
(2) 範囲ごと: 各枠を、いちばん重なる正解の範囲に割り当て、上から（縦書きは右から）つないだ読みと正解を比べる。
    文字の正しさ = max(0, 1 − 編集距離/正解の字数) を字数で重み付けした平均、完全一致の割合
ONLY=<系> を与えると、その系の読みがある面だけで全系を数える（途中までの結果を同じ面で比べるため）。
CONT=0.7 などを環境変数で与えると、別の大きい枠に面積の 7 割以上包まれた枠を落としてから数える（全体と片の枠の二重読みを除くため）。
正解の「／」（行の区切り）と空白は除く。STRIP=1 で、正解の括弧の中（ルビ・校訂の注記）も除く。
使い方: python3 eval_read.py [系 ...]（既定 gt ndl old new）"""
import sys,json,csv,collections
import numpy as np
K=8; FILL=0.2
import os
CONT=float(os.environ.get("CONT","0"))  # >0 なら、面積のこの割合以上が別の（大きい）枠に包まれた枠を落とす
S=sys.argv[1:] or ["gt","ndl","old","new"]
norm=lambda t:"".join(t.split()).replace("／","")
import re
# STRIP=1: 正解の括弧の中（ルビ「（をつと）」や校訂の注記「[讃カ]」）を除いて文字を数える。領域の分け方（1 字など）は norm のまま
STRIP=os.environ.get("STRIP")=="1"
BR=re.compile(r"（[^）]*）|\([^)]*\)|［[^］]*］|\[[^\]]*\]|《[^》]*》")
normc=(lambda t:norm(BR.sub("",t))) if STRIP else norm
def ed(a,b):
    p=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        q=[i]
        for j,cb in enumerate(b,1): q.append(min(p[j]+1,q[j-1]+1,p[j-1]+(ca!=cb)))
        p=q
    return p[-1]
R={r["id"]:r for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t")}
byp=collections.defaultdict(list)
for i,r in R.items(): byp[(r["item"],r["ci"])].append(i)
base={r["id"]:float(r["cov"]) for r in csv.DictReader(open("../data/ndlkoten_cov_fill0.2.tsv"),delimiter="\t")}
hard=[i for v in byp.values() if len(v)>=2 and np.mean([base[x]>=.5 for x in v])<.5 for i in v]
top=sorted(byp,key=lambda k:-sum(base[x]<.5 for x in byp[k]))[:30]
G={"全体":list(R),"詞書":[i for i in R if R[i]["pred"]=="詞書"],"札":[i for i in R if R[i]["pred"]!="詞書"],
   "1字":[i for i in R if len(norm(R[i]["text"]))==1],"2-3字":[i for i in R if 2<=len(norm(R[i]["text"]))<=3],
   "難6面":hard,"上位30面":[i for k in top for i in byp[k]],"格子":byp[("2012-06-03","1")]}
ONLY=os.environ.get("ONLY"); only={tuple(json.loads(l)["key"]) for l in open(f"../data/read/{ONLY}.jsonl")} if ONLY else None
res={}
for s in S:
    acc={}; prec=[0,0]; pagehit=collections.Counter(); pagetot=collections.Counter()
    for d in map(json.loads,open(f"../data/read/{s}.jsonl")):
        key=tuple(d["key"]); ids=byp[key]
        if only is not None and key not in only: continue
        rects={i:[float(R[i][c]) for c in "xywh"] for i in ids}
        W=max(int((x+w)/K)+2 for x,y,w,h in rects.values()); H=max(int((y+h)/K)+2 for x,y,w,h in rects.values())
        W=max(W,max([int(b[2]/K)+2 for b in d["boxes"]]+[0])); H=max(H,max([int(b[3]/K)+2 for b in d["boxes"]]+[0]))
        M=np.full((H,W),-1); Gm=np.zeros((H,W),bool)
        for n,i in enumerate(ids):
            x,y,w,h=(v/K for v in rects[i]); M[int(y):int(y+h)+1,int(x):int(x+w)+1]=n; Gm[int(y):int(y+h)+1,int(x):int(x+w)+1]=True
        asg=collections.defaultdict(list); used=[]
        BT=list(zip(d["boxes"],d["text"]))
        if CONT>0 and s!="gt":
            ar=lambda b:max(0,b[2]-b[0])*max(0,b[3]-b[1]); keep=[]
            for n,(b,t) in enumerate(BT):
                inside=any(ar(o)>ar(b) and max(0,min(b[2],o[2])-max(b[0],o[0]))*max(0,min(b[3],o[3])-max(b[1],o[1]))>=CONT*ar(b) for m,(o,_) in enumerate(BT) if m!=n)
                if not inside: keep.append((b,t))
            BT=keep
        for b,t in BT:
            sl=(slice(int(b[1]/K),int(b[3]/K)+1),slice(int(b[0]/K),int(b[2]/K)+1)); g=Gm[sl]
            if s!="gt" and (g.size==0 or g.mean()<FILL): continue
            used.append(t)
            if s=="gt": asg[b[5]].append((b,t)); continue
            m=M[sl][M[sl]>=0]
            if m.size: asg[ids[np.bincount(m).argmax()]].append((b,t))
        gtc=collections.Counter("".join(normc(R[i]["text"]) for i in ids)); pc=collections.Counter("".join(used))
        hit=sum((gtc&pc).values())
        for i in ids:
            L=sorted(asg.get(i,[]),key=lambda bt:(-bt[0][0],bt[0][1]) if bt[0][3]-bt[0][1]>bt[0][2]-bt[0][0] else (bt[0][1],bt[0][0]))
            pr="".join(t for _,t in L); g=normc(R[i]["text"])
            acc[i]=(len(g),max(0,1-ed(pr,g)/max(1,len(g))),pr==g,sum((collections.Counter(g)&collections.Counter(pr)).values()))
        prec[0]+=hit; prec[1]+=sum(gtc.values()); prec.append(sum(pc.values()))
    res[s]=acc
    print(f"== {s}  面ごとの文字の再現率 {prec[0]/prec[1]:.3f}（{prec[0]}/{prec[1]} 字）  適合率 {prec[0]/max(1,sum(prec[2:])):.3f}（読んだ {sum(prec[2:])} 字）")
for gname,ids in G.items():
    line=[]
    for s in S:
        a=[res[s][i] for i in ids if i in res[s]]; n=sum(x[0] for x in a)
        line.append(f"{s} 正しさ {sum(x[0]*x[1] for x in a)/n:.3f} 一致 {np.mean([x[2] for x in a]):.3f}")
    print(f"{gname}(n={len(ids)})  "+" | ".join(line))
