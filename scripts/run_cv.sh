#!/usr/bin/env bash
# mdx-gpu で 5 分割を順に学習する。使い方: cd ~/seinan-detection/scripts && nohup bash run_cv.sh > ../data/cv.log 2>&1 &
set -euo pipefail
PY=~/islamic-train/.venv/bin/python
$PY make_cv.py ~/seinan-detection/data/yolo
for k in 0 1 2 3 4; do
  echo "== fold$k start $(date)"
  $PY train_yolo.py cv-fold$k yolo11s.pt 1280 80 0 16 ../cv/fold$k.yaml
  echo "== fold$k done $(date)"
done
echo "== all done $(date)"
