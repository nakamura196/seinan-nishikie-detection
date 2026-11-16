exec(open("d2.py").read().split("one=[i")[0])
cb=fres[(0.05,0.7)]; print("G-Cbest w/o dense",boot(g,cb,rest))
# distribution: how many prints where G>C, C>G
items=sorted({R[i]["item"] for i in ids}); byit=collections.defaultdict(list)
for i in ids: byit[R[i]["item"]].append(i)
d={it:sum(g[i]-c[i] for i in byit[it]) for it in items}
print("prints G>C",sum(v>0 for v in d.values()),"C>G",sum(v<0 for v in d.values()),"equal",sum(v==0 for v in d.values()))
# Lite gain concentration: images
