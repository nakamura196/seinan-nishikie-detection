"""Gemini 4 が枠と一緒に返した読み（label）を、eval_read.py が読める形（data/read/gemini_own.jsonl）に直す（2026-11-02）。
read_regions.py の gemini（Gemini の枠を NDL の認識器で読み直したもの）と並べると、Gemini の強みが「探す」側か「読む」側かを分けて見られる。
返答の raw から box_2d と label の組を抜く（run_gemini4_agy.py の枠と同じ順・同じ座標の直し方）。label の無い枠は空の読み。
2026-11-02: 元の指示文では label に "text" がそのまま返り、読みが 6 面にしか無かった。書き起こしを頼んだ指示文（VER=read）の結果は
python3 gemini_own_read.py ../data/gemini4_read_det.jsonl gemini_read で data/read/gemini_read.jsonl に直す（キーは transcription）。
使い方: python3 gemini_own_read.py [入力 jsonl] [系の名前] → python3 eval_read.py gt ndl fill gemini gemini_read"""
import sys
import json, re
src = sys.argv[1] if len(sys.argv) > 1 else "../data/gemini4_det.jsonl"
name = sys.argv[2] if len(sys.argv) > 2 else "gemini_own"
out = open(f"../data/read/{name}.jsonl", "w"); n = 0
for d in map(json.loads, open(src)):
    W, H = d["W"], d["H"]; boxes = []; text = []
    for m in re.finditer(r'\{[^{}]*?"box_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\][^{}]*?\}', d["raw"]):
        y0, x0, y1, x1 = (float(v) / 1000 for v in m.groups())
        boxes.append([min(x0, x1) * W, min(y0, y1) * H, max(x0, x1) * W, max(y0, y1) * H, 1.0])
        lab = re.search(r'"(?:label|transcription)"\s*:\s*"((?:[^"\\]|\\.)*)"', m.group(0))
        text.append(json.loads('"' + lab.group(1) + '"') if lab else "")
    if len(boxes) != len(d["boxes"]): print("枠の数が合わない", d["item"], d["ci"], len(boxes), len(d["boxes"]))
    out.write(json.dumps(dict(key=[d["item"], str(d["ci"])], canvas=d["canvas"], boxes=boxes, text=text), ensure_ascii=False) + "\n"); n += 1
print(n, "面")
