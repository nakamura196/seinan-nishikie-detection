"""NDL の候補（ndl_candidates_kept.tsv のうち query=日清）の IIIF マニフェストを引き、インターネットで公開されているか・コマの数・画像の大きさを確かめる（2026-11-02）。
公開されていないものは {"checkResult": "NG"} が返る。マニフェストは data/external/manifests/<pid>.json に保存する。
出力: data/external/ndl_check.tsv（pid・題名・公開・コマ数・幅・高さ）
使い方: python3 ndl_manifests.py"""
import csv, json, os, time, urllib.request
D = "../data/external"; os.makedirs(f"{D}/manifests", exist_ok=True)
cand = [c for c in csv.DictReader(open(f"{D}/ndl_candidates_kept.tsv"), delimiter="\t") if c["query"] == "日清"]
out = []
for c in cand:
    fp = f"{D}/manifests/{c['pid']}.json"
    if not os.path.exists(fp):
        try: body = urllib.request.urlopen(f"https://dl.ndl.go.jp/api/iiif/{c['pid']}/manifest.json", timeout=30).read()
        except Exception as e: body = json.dumps({"checkResult": "NG", "err": str(e)}).encode()
        open(fp, "wb").write(body); time.sleep(1)
    m = json.load(open(fp))
    cs = m["sequences"][0]["canvases"] if "sequences" in m else []
    out.append(dict(pid=c["pid"], title=c["title"], public=int(bool(cs)), n=len(cs),
                    w=cs[0]["width"] if cs else "", h=cs[0]["height"] if cs else ""))
with open(f"{D}/ndl_check.tsv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(out)
print(len(out), "公開", sum(o["public"] for o in out))
