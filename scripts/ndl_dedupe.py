"""NDL の錦絵の候補（data/external/ndl_candidates.tsv）から、今の 315 点（dcb の item.csv）と重なりうるものを除く（2026-11-02）。
除く条件: (1) 題名の文字の 2-gram の Jaccard が 0.4 以上の資料が 315 点にある、(2) 西南戦争の語（西郷・鹿児島・熊本・薩摩・西南・田原坂・城山・征討・賊）を題名に含む、
(3) 刊年が明治 10〜11 年（西南戦争の年）。外部の評価は「別の戦争・別の題材の錦絵」にするため。
出力: data/external/ndl_candidates_kept.tsv（残した候補、近い 315 点の題名と値も添える）
使い方: python3 ndl_dedupe.py"""
import csv, re, os
item = os.path.expanduser("~/git/work/watanave/dcb/sites/seinan/item.csv")
norm = lambda s: re.sub(r"[\s　・、。「」『』()（）\[\]〔〕\-－ー〈〉.]", "", s or "")
grams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)} or {s}
ours = [(r["title@ja"], grams(norm(r["title@ja"]))) for r in csv.DictReader(open(item))]
cand = list(csv.DictReader(open("../data/external/ndl_candidates.tsv"), delimiter="\t"))
WORDS = "西郷 鹿児島 熊本 薩摩 西南 田原坂 城山 征討 賊".split()
z2h = str.maketrans("０１２３４５６７８９", "0123456789")
kept = []; why = {"題名が近い": 0, "西南戦争の語": 0, "明治10〜11年": 0}
for c in cand:
    g = grams(norm(c["title"]))
    best = max(ours, key=lambda o: len(g & o[1]) / len(g | o[1]))
    j = len(g & best[1]) / len(g | best[1])
    yr = re.search(r"明治\s*(\d+)", c["issued"].translate(z2h))
    if j >= 0.4: why["題名が近い"] += 1; continue
    if any(w in c["title"] for w in WORDS): why["西南戦争の語"] += 1; continue
    if yr and int(yr.group(1)) in (10, 11): why["明治10〜11年"] += 1; continue
    kept.append(dict(c, near=best[0], jaccard=round(j, 2)))
with open("../data/external/ndl_candidates_kept.tsv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(kept[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(kept)
print(len(cand), "→", len(kept), why)
