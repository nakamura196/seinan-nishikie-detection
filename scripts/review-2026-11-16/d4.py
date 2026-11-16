exec(open("d2.py").read().split("one=[i")[0])
sub=json.load(open(f"{D}/vlm_subset.json")); s90={tuple(k) for v in sub.values() for k in v}
r649=[i for i in ids if (R[i]["item"],str(R[i]["ci"])) not in s90]
print("649 imgs regions",len(r649),"G",round(100*np.mean([g[i] for i in r649]),2),"C",round(100*np.mean([c[i] for i in r649]),2),"L",round(100*np.mean([l[i] for i in r649]),2))
print("G-C 649",boot(g,c,r649)[:2]); print("G-L 649",boot(g,l,r649)[:2])
one=[i for i in r649 if len(R[i]["text"])==1]; print("1char 649 n",len(one),"G-C",boot(g,c,one))
# prints overlapping: items in 90 also have other canvases in 649?
it90={k[0] for k in s90}; print("prints in 90",len(it90), "649-regions on those prints",sum(R[i]["item"] in it90 for i in r649))
