import os,json,gzip,collections,re,sys
P=os.environ["ANNOTATION_BACKUP"]  # Firestore export of IIIF Annotator (2026-10-25); not distributed
def load(p=P):
    d=json.load(gzip.open(p)); out={}
    for v in d["annotations"].values():
        a=v.get("_data",v)
        if "drupal.lab" in str(a.get("manifestId","")) or "drupal.lab" in json.dumps(a.get("target",{})):
            out[a["id"]]=dict(a, _user=a.get("userName"), _uid=a.get("userId"), _canvas=a.get("canvasId") or a["target"].get("source",{}).get("id"), _src="ann")
    for v in d["annotationPages"].values():
        pg=v.get("_data",v)
        if "drupal.lab" not in str(pg.get("manifestId","")): continue
        for it in pg.get("items",[]):
            out.setdefault(it["id"],dict(it,_user=pg.get("userName"),_uid=pg.get("userId"),_canvas=pg.get("canvasId"),_manifest=pg.get("manifestId"),_src="page"))
    return list(out.values())
if __name__=="__main__":
    s=load()
    print(len(s), collections.Counter(a["_src"] for a in s), "面",len({a["_canvas"] for a in s}))
    print(collections.Counter((a["_user"] or "(無記名)", (a["_uid"] or "")[:6]) for a in s))
