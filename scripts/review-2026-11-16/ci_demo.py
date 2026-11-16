"""Values for the confidence-interval explainer slides (papers/seinan-nishikie-detection/explainer-ci). Same measure as paper_numbers.py.
Usage: python3 ci_demo.py <out.json>"""
import os,sys,json,collections
import numpy as np
sys.argv=[sys.argv[0]]+sys.argv[1:]; out=os.path.abspath(sys.argv[1])
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"paper_numbers.py")).read()
exec(src.split("# C's upper bound")[0])
G,_,_=evaluate(f"{D}/gemini4_det.jsonl"); C,_,_=evaluate(f"{D}/main_cv_all.jsonl"); Lt,_,_=evaluate(f"{D}/ndlkoten_det.jsonl",0.3)
g={i:G[i]>=.5 for i in ids}; c={i:C[i]>=.5 for i in ids}; l={i:Lt[i]>=.5 for i in ids}
by=collections.defaultdict(list)
for i in ids: by[R[i]["item"]].append(i)
its=sorted(by)
print("all",len(ids),"G",sum(g.values()),"C",sum(c.values()),"diff",sum(g.values())-sum(c.values()),round(100*(sum(g.values())-sum(c.values()))/len(ids),3))
for it in ("2013-24-1","1990-032-206","2012-06-03"):
    print(it,len(by[it]),sum(g[i] for i in by[it]),sum(c[i] for i in by[it]),sum(g[i]-c[i] for i in by[it]))
A={it:(sum(g[i] for i in by[it]),sum(c[i] for i in by[it]),sum(l[i] for i in by[it]),len(by[it])) for it in its}
rng=np.random.default_rng(20261116); reps=[]; first=[]
for k in range(2000):
    S=rng.choice(its,len(its)); n=sum(A[s][3] for s in S); gg=sum(A[s][0] for s in S); cc=sum(A[s][1] for s in S)
    reps.append(100*(gg-cc)/n)
    if k<3: cnt=collections.Counter(S); first.append((n,gg,cc,round(100*(gg-cc)/n,2),cnt["2013-24-1"],cnt["1990-032-206"],len(cnt),315-len(cnt),cnt.most_common(1)))
r=np.sort(reps); print("first3",first)
print("2.5%: 50th",round(r[49],3),"51st",round(r[50],3),"| 97.5%: 1950th",round(r[1949],3),"1951st",round(r[1950],3),"pct",np.percentile(reps,[2.5,97.5]).round(3),"below0",int((r<0).sum()))
rng=np.random.default_rng(20261116); gl=[]
for k in range(2000):
    S=rng.choice(its,len(its)); n=sum(A[s][3] for s in S); gl.append(100*(sum(A[s][0] for s in S)-sum(A[s][2] for s in S))/n)
print("G-Lite",round(100*(sum(g.values())-sum(l.values()))/len(ids),2),np.percentile(gl,[2.5,97.5]).round(2),"min",round(min(gl),2))
rng=np.random.default_rng(1); a=np.array([g[i] for i in ids],float)-np.array([c[i] for i in ids],float)
rb=[100*a[rng.integers(0,len(a),len(a))].mean() for _ in range(2000)]; print("region-level",np.percentile(rb,[2.5,97.5]).round(2))
hist=collections.Counter(np.floor(np.array(reps)/0.25).astype(int).tolist())
json.dump({"reps":reps,"hist":{str(k*0.25):v for k,v in sorted(hist.items())}},open(out,"w"))
print("hist range",min(hist)*0.25,max(hist)*0.25+0.25,"max count",max(hist.values()))
