"""Gemini 4 の検出結果を、学習した検出器 C と並べて、人が読みやすい PDF にする（本人の目視用、2026-11-02、作り直し）。
1 ページ 1 面、A4 横。上に「この面で見てほしいこと」を 1 行、下に色見本つきの凡例。
図は紙の部分（正解と枠の範囲）だけを切り出す。拾えた正解の線は描かず、拾えなかった正解だけを赤く塗る（線が多すぎて見づらかったため）。
出力: ../data/gemini_view/gemini_examples.pdf
使い方: python3 gemini_examples_pdf.py"""
import json, csv, collections, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
K = 8; FILL = 0.2; D = "../data"
R = list(csv.DictReader(open(f"{D}/rows.tsv"), delimiter="\t")); by = collections.defaultdict(list)
for r in R: by[(r["item"], str(r["ci"]))].append(r)
def load(p): return {(d["item"], str(d["ci"])): d for d in map(json.loads, open(p))}
GEM, C = load(f"{D}/gemini4_det.jsonl"), load(f"{D}/main_cv_all.jsonl")
def score(d, L):
    W, H = d["W"] // K + 1, d["H"] // K + 1; G = np.zeros((H, W), bool); P = np.zeros_like(G); outs = []
    for r in L:
        x, y, w, h = (float(r[c]) / K for c in "xywh"); G[int(y):int(y + h) + 1, int(x):int(x + w) + 1] = True
    for b in d["boxes"]:
        m = G[int(b[1] / K):int(b[3] / K) + 1, int(b[0] / K):int(b[2] / K) + 1]
        outs.append(m.size == 0 or m.mean() < 0.1)
        if m.size and m.mean() >= FILL: P[int(b[1] / K):int(b[3] / K) + 1, int(b[0] / K):int(b[2] / K) + 1] = True
    hit = {}
    for r in L:
        x, y, w, h = (float(r[c]) / K for c in "xywh"); m = P[int(y):int(y + h) + 1, int(x):int(x + w) + 1]
        hit[r["id"]] = bool(m.size) and m.mean() >= 0.5
    return hit, outs
# 面と、その面で見てほしいこと
PICK = [(("2015-017-19", "1"), "地が黒い面。C は 6 か所とも拾えず、Gemini は 6 か所すべて拾った。"),
        (("2017-013-01", "2"), "難しい面の一つ。Gemini は全部拾い、C は一部を落とした。"),
        (("1990-032-187", "1"), "人名の札が並ぶ面。どちらも全部拾った。Gemini は注釈の無い札「川路少将」も拾っている（橙）。"),
        (("2013-17", "1"), "詞書の多い面。どちらも範囲は拾えているが、Gemini は段落ごとに大きな枠でまとめ、C は 1 行ずつ囲む。"),
        (("1990-032-206", "1"), "文字が細かく詰まった面。Gemini が C より多く落とした（赤）。"),
        (("2013-24-1", "1"), "文字が細かく詰まった面（もう一つ）。Gemini が C より多く落とした（赤）。"),
        (("1990-032-036", "1"), "無作為に選んだ面。ふつうの面では、どちらもほぼ全部拾える。")]
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"; FONT3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
fT, fB, fS = ImageFont.truetype(FONT, 40), ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT3, 26)
PW, PH = 2339, 1654  # A4 横、200dpi
BLUE, GREEN, ORANGE, RED = (30, 90, 230), (0, 150, 60), (240, 130, 0), (220, 0, 0)
def legend(dr, y):
    items = [(BLUE, "box", "Gemini 4 が検出した枠"), (GREEN, "box", "C が検出した枠"), (ORANGE, "box", "注釈の無い所の枠（実在の文字のことも、絵のこともある）"),
             (RED, "fill", "拾えなかった正解（注釈の範囲）")]
    x = 60
    for col, kind, txt in items:
        if kind == "box": dr.rectangle([x, y, x + 44, y + 30], outline=col, width=5)
        else: dr.rectangle([x, y, x + 44, y + 30], fill=(255, 170, 170), outline=col, width=3)
        dr.text((x + 56, y - 2), txt, fill="black", font=fS); x += 56 + int(dr.textlength(txt, font=fS)) + 50
def panel(im, s, d, L, col, crop):
    hit, outs = score(d, L); a = im.copy(); ov = Image.new("RGBA", a.size, (0, 0, 0, 0)); do = ImageDraw.Draw(ov)
    for r in L:
        if hit[r["id"]]: continue
        x, y, w, h = (float(r[c]) / s for c in "xywh"); do.rectangle([x, y, x + w, y + h], fill=(255, 0, 0, 90), outline=RED + (255,), width=4)
    a = Image.alpha_composite(a.convert("RGBA"), ov).convert("RGB"); dr = ImageDraw.Draw(a)
    for b, o in zip(d["boxes"], outs): dr.rectangle([b[0] / s, b[1] / s, b[2] / s, b[3] / s], outline=ORANGE if o else col, width=6)
    return a.crop(crop), sum(hit.values()), len(hit), len(d["boxes"]), sum(outs)
pages = []
cover = Image.new("RGB", (PW, PH), "white"); dr = ImageDraw.Draw(cover)
lines = ["Gemini 4（学習なし）と、学習した検出器 C の検出結果の例", "",
         "西南戦争錦絵 全 739 面での結果（拾えた割合 = 注釈の範囲の半分以上が枠に覆われた割合）",
         "　Gemini 4　96.2%　　C　95.7%　　NDL古典籍OCR-Lite　87.1%", "", "このあとのページの見方",
         "　・1 ページに 1 面。Gemini 4 を左（横長の面は上）、C を右（横長の面は下）に、同じ大きさで並べています。",
         "　・枠の色は下の凡例のとおりです。拾えた正解には線を引いていません。赤く塗った所だけが、拾えなかった正解です。",
         "　・各ページの上に、その面で見てほしいことを 1 行で書いています。",
         "　・数字は「拾えた正解の数 / 正解の数」「検出した枠の数」「そのうち注釈の無い所の枠の数」です。"]
y = 120
for i, t in enumerate(lines): dr.text((80, y), t, fill="black", font=fT if i == 0 else fB); y += 80 if i == 0 else 62
legend(dr, PH - 120); pages.append(cover)
for k, note in PICK:
    im = Image.open(f"{D}/img/{k[0]}_{k[1]}.jpg").convert("RGB"); s = GEM[k]["W"] / im.width
    xs = [float(r["x"]) / s for r in by[k]] + [b[0] / s for S in (GEM, C) for b in S[k]["boxes"]]
    ys = [float(r["y"]) / s for r in by[k]] + [b[1] / s for S in (GEM, C) for b in S[k]["boxes"]]
    xe = [(float(r["x"]) + float(r["w"])) / s for r in by[k]] + [b[2] / s for S in (GEM, C) for b in S[k]["boxes"]]
    ye = [(float(r["y"]) + float(r["h"])) / s for r in by[k]] + [b[3] / s for S in (GEM, C) for b in S[k]["boxes"]]
    m = 60; crop = (max(0, int(min(xs)) - m), max(0, int(min(ys)) - m), min(im.width, int(max(xe)) + m), min(im.height, int(max(ye)) + m))
    pg = Image.new("RGB", (PW, PH), "white"); dr = ImageDraw.Draw(pg)
    dr.text((60, 30), f"{k[0]}　{k[1]} 枚目", fill="black", font=fT); dr.text((60, 95), note, fill="black", font=fB)
    top, bottom = 170, PH - 150
    wide = (crop[2] - crop[0]) > (crop[3] - crop[1])  # 横長の面は上下に、縦長の面は左右に並べる
    for i, (name, S, col) in enumerate((("Gemini 4（学習なし）", GEM, BLUE), ("C（注釈で学習した検出器）", C, GREEN))):
        a, h_, n, nb, no = panel(im, s, S[k], by[k], col, crop)
        if wide:
            hh = (bottom - top) // 2; a.thumbnail((PW - 120, hh - 55)); x0, y0 = 60, top + i * hh
        else:
            bw = (PW - 180) // 2; a.thumbnail((bw, bottom - top - 55)); x0, y0 = 60 + i * (bw + 60), top
        dr.text((x0, y0), f"{name}　拾えた {h_}/{n}　枠 {nb}（うち注釈の無い所 {no}）", fill=col, font=fB)
        pg.paste(a, (x0, y0 + 45))
    legend(dr, PH - 110); pages.append(pg)
os.makedirs(f"{D}/gemini_view", exist_ok=True)
pages[0].save(f"{D}/gemini_view/gemini_examples.pdf", save_all=True, append_images=pages[1:], resolution=200)
print(len(pages), "ページ")
