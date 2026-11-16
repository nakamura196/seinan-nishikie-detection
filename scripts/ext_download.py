"""外部の評価用（NDL の日清戦争錦絵）の画像を IIIF マニフェストから取得する（2026-11-03）。
対象は ../data/external/ndl_check.tsv の public=1・n>0 の全点・全コマ。西南戦争の画像と同じく長辺 2560px（run_ndlkoten_det.py と同じ指定）。
画像は ../data/external/images/<pid>_<コマ番号>.jpg、一覧は ../data/external/images.tsv（item・ci・canvas・実際の幅と高さ・元の幅と高さ）。
機械の出力は ../data/external/machine/ に分けて置く（正解を人が描くとき目に入らないように。画像には枠を描かない）。
使い方: python3 ext_download.py"""
import csv, json, os, time, urllib.request
from PIL import Image
D = "../data/external"; os.makedirs(f"{D}/images", exist_ok=True)
rows = []
for c in csv.DictReader(open(f"{D}/ndl_check.tsv"), delimiter="\t"):
    if c["public"] != "1" or int(c["n"]) == 0: continue
    for ci, cv in enumerate(json.load(open(f"{D}/manifests/{c['pid']}.json"))["sequences"][0]["canvases"], 1):
        W, H = cv["width"], cv["height"]; svc = cv["images"][0]["resource"]["service"]["@id"]
        p = f"{D}/images/{c['pid']}_{ci}.jpg"
        if not os.path.exists(p):
            url = f"{svc}/full/" + ("max" if max(W, H) <= 2560 else "2560," if W >= H else ",2560") + "/0/default.jpg"
            for t in range(3):
                try: open(p, "wb").write(urllib.request.urlopen(url, timeout=120).read()); break
                except Exception as e: print("再試行", c["pid"], ci, e); time.sleep(5)
            time.sleep(0.5)
        w, h = Image.open(p).size
        rows.append(dict(item=c["pid"], ci=ci, canvas=cv["@id"], w=w, h=h, W=W, H=H, title=c["title"]))
with open(f"{D}/images.tsv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t"); wr.writeheader(); wr.writerows(rows)
print(len(set(r["item"] for r in rows)), "点", len(rows), "面")
