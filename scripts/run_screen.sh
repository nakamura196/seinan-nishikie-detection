#!/usr/bin/env bash
# mdx-gpu で、格子の面（2012-06-03 p1、fold 3）を伸ばせるかの下見（2026-11-01）。fold 3 だけで 2 つの設定を順に学習する。
#  (1) cv-t2m-fold3: yolo11m、1280、batch 16（モデルを大きくする）
#  (2) cv-t2s1920-fold3: yolo11s、1920、batch 8（画像を大きくする。1920 で batch 16 は 40GB に収まらないおそれ）
# 学習画像は前と同じ（元の画像 + 2×2 の片、data/cv/fold3_t2.yaml）。80 回、patience 20。効いた方だけ 5 分割に広げる。
# 使い方: cd ~/seinan-detection/scripts && nohup bash run_screen.sh > ../data/screen.log 2>&1 &
set -euo pipefail
PY=~/islamic-train/.venv/bin/python
run(){ echo "== $1 start $(date)"; $PY train_yolo.py $1 $2 $3 80 0 $4 ../cv/fold3_t2.yaml > ../data/$1.log 2>&1; echo "== $1 done $(date)"; }
run cv-t2m-fold3 yolo11m.pt 1280 16
run cv-t2s1920-fold3 yolo11s.pt 1920 8
echo "== all done $(date)"
