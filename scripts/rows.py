import os,json,csv,collections,re,sys
sys.path.insert(0,".")
from load import load
MAP=json.load(open(os.environ["MANIFEST_MAP"]))
media=collections.defaultdict(list)
for r in csv.DictReader(open(os.environ["MEDIA_CSV"])): media[r["item_id"]].append(r)
def txt(b):
    v=b.get("value","") if isinstance(b,dict) else ""
    v=re.sub("<[^>]+>","",v); return re.sub(r"\([^)]*\)","",v).strip()
def hira(t): return sum(1 for ch in t if 'ぁ'<=ch<='ゟ')/max(1,len(t))
rows=[]
for a in load():
    if a["_src"]!="page": continue
    if "OCR" in (a.get("tags") or []): continue  # 2026-11-16: OCR の出力を取り込んだ印の付いた領域（257）は正解から除く。元の表は rows_all.tsv
    fs=[x for x in a["target"]["selector"] if x["type"]=="FragmentSelector"][0]["value"]
    x,y,w,h=map(float,fs.split("=")[1].split(","))
    u=re.search(r"iiif/2/([0-9a-f-]+)/canvas/p(\d+)",a["_canvas"]); item=MAP[u.group(1)]; ci=int(u.group(2))
    md=media[item][ci-1]
    rows.append(dict(id=a["id"],item=item,ci=ci,canvas=a["_canvas"],img=md["url"].replace("/info.json",""),W=int(md["width"]),H=int(md["height"]),
        user=a["_user"] or "(無記名)",text=txt(a["body"]),x=x,y=y,w=w,h=h))
byc=collections.defaultdict(list)
for r in rows: byc[r["canvas"]].append(r)
for L in byc.values():
    for r in L:
        r["nb"]=sum(1 for o in L if o is not r and max(o["x"]-(r["x"]+r["w"]), r["x"]-(o["x"]+o["w"]))<0.8*max(r["w"],o["w"])
                    and min(r["y"]+r["h"],o["y"]+o["h"])-max(r["y"],o["y"])>0.5*min(r["h"],o["h"]))
iso=collections.defaultdict(set)
for r in rows:
    t=r["text"]
    if r["nb"]==0 and 2<=len(t)<=8 and hira(t)<0.25: iso[t].add(r["canvas"])
names={t for t,v in iso.items() if len(v)>=2}
for r in rows:
    t=r["text"]; core=re.split(r"[　 ]",t)[-1]
    if (t in names or core in names) and len(t)<=16: k="札"
    elif r["nb"]==0 and 2<=len(t)<=10 and hira(t)<0.25: k="札"
    elif r["nb"]<=1 and 2<=len(t)<=10 and hira(t)<0.25: k="札?"
    else: k="詞書"
    r["pred"]=k
if __name__=="__main__":
    print(len(rows), collections.Counter(r["pred"] for r in rows), "札のある面",len({r["canvas"] for r in rows if r["pred"]=="札"}))
    with open("../data/rows.tsv","w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),delimiter="\t"); w.writeheader(); w.writerows(rows)
