#!/usr/bin/env bash
# mdx-gpu で、元の画像 + 2×2 の片で 5 分割を学習し直す（2026-10-31）。1 本ずつ順に回す（2 本並べると 40GB に収まらず、計算の一部が CPU に回されたため）。
# 設定は前の 5 分割と同じ（yolo11s、1280、80 回、patience 20、batch 16）。違いは学習画像に片を足したことだけ。
# 使い方: cd ~/seinan-detection/scripts && nohup bash run_cv_t2.sh > ../data/cv_t2.log 2>&1 &
set -euo pipefail
PY=~/islamic-train/.venv/bin/python
$PY make_tiles.py ~/seinan-detection/data
run(){ echo "== t2 fold$1 start $(date)"; $PY train_yolo.py cv-t2-fold$1 yolo11s.pt 1280 80 0 16 ../cv/fold$1_t2.yaml > ../data/cv_t2_fold$1.log 2>&1; echo "== t2 fold$1 done $(date)"; }
for k in 0 1 2 3 4; do run $k; done
echo "== all done $(date)"
