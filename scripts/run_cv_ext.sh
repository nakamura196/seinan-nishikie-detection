#!/usr/bin/env bash
# mdx-gpu で、重なりの帯に注釈を写して補った正解（data/rows_ext.tsv）で、主設定 C（元の画像 + 2×2 の片）の 5 分割を学習し直す（2026-11-02）。
# 設定は run_cv_t2.sh と同じ（yolo11s、1280、80 回、patience 20、batch 16、1 本ずつ順に）。違いは正解だけ。
# make_yolo.py が mdx-gpu 側の data/yolo/labels を補った正解で書き直す（分け方 split.tsv・cv.tsv は面が同じなので変わらない）。
# 前提: 手元から scripts/ と data/rows_ext.tsv を ~/seinan-detection/ に送ってあること。
# 使い方: cd ~/seinan-detection/scripts && nohup bash run_cv_ext.sh > ../data/cv_ext.log 2>&1 &
set -euo pipefail
PY=~/islamic-train/.venv/bin/python
ROWS=../data/rows_ext.tsv $PY make_yolo.py
$PY make_tiles.py ~/seinan-detection/data
run(){ echo "== ext fold$1 start $(date)"; $PY train_yolo.py cv-ext-fold$1 yolo11s.pt 1280 80 0 16 ../cv/fold$1_t2.yaml > ../data/cv_ext_fold$1.log 2>&1; echo "== ext fold$1 done $(date)"; }
for k in 0 1 2 3 4; do run $k; done
echo "== all done $(date)"
