exec(open("decomp.py").read().split("# per print diff")[0])
items=sorted({R[i]["item"] for i in ids}); byit=collections.defaultdict(list)
for i in ids: byit[R[i]["item"]].append(i)
rng=np.random.default_rng(2)
def boot(a,b,sel):
    its=sorted({R[i]["item"] for i in sel}); by=collections.defaultdict(list)
    for i in sel: by[R[i]["item"]].append(i)
    f=lambda S:100*(np.mean([a[i] for it in S for i in by[it]])-np.mean([b[i] for it in S for i in by[it]]))
    bs=[f(rng.choice(its,len(its))) for _ in range(2000)]; return round(f(its),2),np.round(np.percentile(bs,[2.5,97.5]),2),len(sel),len(its)
dense={"1990-032-206","2013-24-1"}
rest=[i for i in ids if R[i]["item"] not in dense]
print("G-C w/o dense",boot(g,c,rest))
one=[i for i in ids if len(R[i]["text"])==1]
for key in [(0.05,0.5),(0.05,0.7),(0.1,0.5),(0.15,0.5),(0.25,0.5)]:
    cb=fres[key]; print(key,"1char C",round(100*np.mean([cb[i] for i in one]),1),"G-C",boot(g,cb,one), "all G-C",boot(g,cb,ids)[:2])
# 2-3 chars and label
two=[i for i in ids if 2<=len(R[i]["text"])<=3]
cb=fres[(0.05,0.7)]; print("2-3 G",round(100*np.mean([g[i] for i in two]),1),"Cbest",round(100*np.mean([cb[i] for i in two]),1))
lab=[i for i in ids if R[i]["pred"]=="札"]; print("label G",round(100*np.mean([g[i] for i in lab]),1),"Cbest",round(100*np.mean([cb[i] for i in lab]),1))
# Gemini 3 runs vs Cbest
for fn in ["gemini4_det_rep2.jsonl","gemini4_det_rep3.jsonl"]:
    gg=ok(evaluate(f"{D}/{fn}")); print(fn,"all",boot(gg,cb,ids)[:2],"1char",boot(gg,cb,one)[:2])
