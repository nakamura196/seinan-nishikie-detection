"""重なりのあるコマの組で、隣のコマの注釈を重なりの帯に写して正解を補う。
対象は data/register/pair_class.tsv の class=重なりあり（Gemini の確かめで、重なりがあり位置も合っている組）だけ。
変換は data/register/pairs.tsv（コマ k2 の data/img の座標 → コマ k の data/img の座標の相似変換）。
両向きに写す（k2 → k の左端の帯、k → k2 の右端の帯）。写すのは元の注釈だけ（写したものをさらに写さない）。
残す条件: 写した枠の面積の半分以上がコマの中に入る（はみ出した分は切る）、
写した先の画像が元の枠の中身と同じ（灰色の濃淡の相関 >= NCC。重なりが片側にしか無い組で、背景に写すのを防ぐ）、
かつ、写した先の既存の注釈と重ならない（写した枠の面積の 30% 以上が既存の注釈に覆われていたら、同じものとみなして足さない）。
出力: data/rows_ext.tsv（rows.tsv と同じ列に src を足す。元の注釈は空、写したものは copy:<元の id>）、
data/register/ext_check/<資料>_<コマ>.jpg（写した枠を緑、元の枠を赤で描いた確認図。--viz を付けたとき）
使い方: python3 rows_ext.py [--viz]"""
import sys, csv, collections
import numpy as np, cv2
from PIL import Image

D = "../data"
rows = list(csv.DictReader(open(f"{D}/rows.tsv"), delimiter="\t"))
byf = collections.defaultdict(list)
for r in rows: byf[(r["item"], r["ci"])].append(r)
cls = {(r["item"], r["k"], r["k2"]): r["class"] for r in csv.DictReader(open(f"{D}/register/pair_class.tsv"), delimiter="\t")}
pairs = [r for r in csv.DictReader(open(f"{D}/register/pairs.tsv"), delimiter="\t") if cls.get((r["item"], r["k"], r["k2"])) == "重なりあり"]

size = {}
def scale(item, ci):
    """元画像の座標 / data/img の座標。元画像の大きさは注釈の W,H（注釈の無いコマは他の注釈から引けないので None）"""
    L = byf.get((item, ci))
    if not L: return None
    if (item, ci) not in size: size[item, ci] = Image.open(f"{D}/img/{item}_{ci}.jpg").size
    return float(L[0]["W"]) / size[item, ci][0]

NCC = 0.5
def gray(it, ci): return cv2.cvtColor(cv2.imread(f"{D}/img/{it}_{ci}.jpg"), cv2.COLOR_BGR2GRAY).astype(np.float32)

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def S(s): return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1.0]])

def inter(a, b): return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

nccs = []; out = []; cnt = collections.Counter(); faces = collections.Counter(); meta = {}
for r in rows: out.append(dict(r, src=""))
for p in pairs:
    it, k, k2 = p["item"], p["k"], p["k2"]
    s1, s2 = scale(it, k), scale(it, k2)
    if s1 is None or s2 is None:
        cnt["注釈の無いコマを含む組（飛ばした）"] += 1; continue  # 元画像の大きさが分からない
    th = np.radians(float(p["angle"])); sc = float(p["scale"])
    M = np.array([[sc * np.cos(th), -sc * np.sin(th), float(p["tx"])], [sc * np.sin(th), sc * np.cos(th), float(p["ty"])], [0, 0, 1]])
    Mo = S(s1) @ M @ np.linalg.inv(S(s2))  # 元画像の座標どうし: k2 → k
    for src, dst, T in ((k2, k, Mo), (k, k2, np.linalg.inv(Mo))):
        # 写した先の画像を、元のコマの data/img の座標に引き戻す（valid: 写した先のコマの中に入る画素）
        Ti = (np.linalg.inv(S(scale(it, dst))) @ T @ S(scale(it, src)))[:2]
        gs, gd = gray(it, src), gray(it, dst); hs, ws = gs.shape
        back = cv2.warpAffine(gd, Ti, (ws, hs), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        valid = cv2.warpAffine(np.ones_like(gd), Ti, (ws, hs), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST) > 0
        ss = scale(it, src)
        tgt = byf[(it, dst)]; W, H = float(tgt[0]["W"]), float(tgt[0]["H"])
        have = [(float(o["x"]), float(o["y"]), float(o["x"]) + float(o["w"]), float(o["y"]) + float(o["h"])) for o in tgt]
        for a in byf[(it, src)]:
            x, y, w, h = (float(a[c]) for c in "xywh")
            q = T @ np.array([[x, x + w, x, x + w], [y, y, y + h, y + h], [1, 1, 1, 1]])
            b = (q[0].min(), q[1].min(), q[0].max(), q[1].max()); ar = (b[2] - b[0]) * (b[3] - b[1])
            c = (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
            if c[2] <= c[0] or c[3] <= c[1]: continue
            if (c[2] - c[0]) * (c[3] - c[1]) < 0.5 * ar: cnt["半分以上がコマの外"] += 1; continue
            X0, Y0, X1, Y1 = (int(round(v / ss)) for v in (x, y, x + w, y + h)); X0, Y0 = max(0, X0), max(0, Y0)
            m = valid[Y0:Y1, X0:X1]
            v = ncc(gs[Y0:Y1, X0:X1][m], back[Y0:Y1, X0:X1][m]) if m.sum() > 50 else 0.0
            nccs.append((src == k, v))
            if v < NCC: cnt["中身が合わない（片側だけの重なりなど）"] += 1; continue
            if sum(inter(c, o) for o in have) >= 0.3 * (c[2] - c[0]) * (c[3] - c[1]): cnt["既存の注釈と重なる"] += 1; continue
            cnt["写した"] += 1; faces[(it, dst)] += 1
            out.append(dict(a, id=f"{a['id']}@{it}_{dst}", ci=dst, canvas=tgt[0]["canvas"], img=tgt[0]["img"], W=tgt[0]["W"], H=tgt[0]["H"],
                            x=c[0], y=c[1], w=c[2] - c[0], h=c[3] - c[1], src=f"copy:{a['id']}"))

with open(f"{D}/rows_ext.tsv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(out)
print(f"組 {len(pairs)}（重なりあり）", dict(cnt))
for d, nm in ((False, "k2→k（左端の帯）"), (True, "k→k2（右端の帯）")):
    v = np.array([n for dd, n in nccs if dd == d]); print(nm, "相関の分布", np.histogram(v, bins=[-1, 0, .2, .4, .5, .6, .8, 1.01])[0])
print(f"元の注釈 {len(rows)} → {len(out)}（写した {len(out) - len(rows)}、面 {len(faces)}）",
      collections.Counter(r["pred"] for r in out if r["src"]))

if "--viz" in sys.argv:
    import os, cv2
    os.makedirs(f"{D}/register/ext_check", exist_ok=True)
    for (it, ci), n in faces.items():
        A = cv2.imread(f"{D}/img/{it}_{ci}.jpg"); s = scale(it, ci)
        for r in out:
            if r["item"] != it or r["ci"] != ci: continue
            x, y, w, h = (float(r[c]) / s for c in "xywh")
            cv2.rectangle(A, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0) if r["src"] else (0, 0, 230), 4 if r["src"] else 2)
        cv2.imwrite(f"{D}/register/ext_check/{it}_{ci}.jpg", cv2.resize(A, None, fx=0.4, fy=0.4), [cv2.IMWRITE_JPEG_QUALITY, 75])
