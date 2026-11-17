# seinan-nishikie-detection

Evaluation code and machine outputs for the paper
"Text Detection in Nishiki-e Prints: A Book-Trained Layout Model, a Training-Free Multimodal Model and a Detector Trained on Transcription Annotations"
(Satoru Nakamura, Miku Takahashi, Hiroshi Hakoishi, Taizo Yamada; draft for DH2027).

The paper compares three ways to locate text regions (inscriptions, name labels, titles, signatures)
in 739 canvases of 316 nishiki-e prints of the Satsuma Rebellion (1877):

- NDL Kotenseki OCR-Lite and NDL Kotenseki OCR ver. 3 (layout models trained on books),
- Gemini 4 and Claude Opus 5.5 (multimodal models, no training),
- YOLO11s trained on the transcription annotations of the prints (5-fold cross-validation by item).

## What is in this repository

| Path | Contents |
|---|---|
| `scripts/` | All scripts used for the experiments (detection, training, evaluation, tables, figures). Comments are in Japanese. |
| `scripts/review-2026-11-16/` | Scripts that print the numbers quoted in the paper but not printed by `paper_tables.py`, and their outputs (`out-2026-11-16-ocr-excluded/`). |
| `data/*.jsonl` | Detected regions per canvas, one JSON object per line: `canvas` (IIIF Canvas URI), `item`, `ci` (canvas index), `W`, `H`, `boxes` (`[x0, y0, x1, y1, score]` in full-image pixels). VLM outputs also keep the raw answer (`raw`). |
| `data/*_cov*.tsv` | Coverage of each annotated region by the detected regions (`id` = annotation ID, `cov` = covered fraction). |
| `data/cv.tsv`, `data/split.tsv` | Fold assignment (by item) and the train/test split. |
| `data/vlm_subset.json` | The 90 canvases used for the VLM comparison (30 hardest for NDL Lite + 60 random, seed 20261101). |
| `data/tiles/` | Detections of each fold model with low-confidence boxes kept (threshold sweep). |
| `data/read/` | Text read by NDL Kotenseki OCR-Lite inside the detected regions (Section 4.3). |
| `data/claude_raw/` | Raw answers of Claude Opus 5.5, one file per canvas. |
| `data/external/` | External test set (NDL Digital Collections, Sino-Japanese War prints): image list and machine outputs. |
| `data/register/` | Registration of overlapping canvases of multi-sheet prints (used to extend the annotations in `rows_ext.py`). |
| `data/outside_check/` | Manual check of detected regions that overlap no annotation ("outside boxes", Section 5.2): 100 random boxes each for YOLO11s (`main_C/s100.tsv`, drawn with seed 20261116 from an earlier sample of 150) and Gemini 4 (`gemini4/s100.tsv`). Coordinates are in full-image pixels. `judge` holds the class proposed by Claude Opus 5.5 (`claude:`) and the class decided by the first author (`author:`): `scene` (unannotated text in the picture), `margin` (marginal publication text, signatures, seals), `picture` (no text), `overlap` (a strip annotated on the overlapping neighbouring sheet), `unclear`. `combined/outside_check_author.tsv` holds the same 200 judgements in one table (`overlap` = fraction covered by annotations copied from the neighbouring sheet). The samples were drawn before the 257 OCR-tagged regions were excluded. |
| `data/paper_tables.txt` | Output of `paper_tables.py`. |

## Multimodal models: instructions and run dates

The instruction given to the models is in `scripts/run_gemini4_agy.py` (`PROMPT`; the `VER=read` variant also asks for a transcription).
Claude Opus 5.5 received the same instruction (`scripts/claude_raw_to_det.py`).

| Output | Model | Run date |
|---|---|---|
| `data/gemini4_det.jsonl` | Gemini 4 (`gemini-4-argon-high`, via Google Antigravity CLI) | 2026-11-02 |
| `data/gemini4_det_rep2.jsonl` | same, second run | 2026-11-02 |
| `data/gemini4_det_rep3.jsonl` | same, third run | 2026-11-03 |
| `data/gemini4_read_det.jsonl` | same, with transcription | 2026-11-02 |
| `data/vlm_claude-opus-5-5_det.jsonl` | Claude Opus 5.5 (Claude Code) | 2026-11-02 |
| `data/external/machine/gemini4_det.jsonl` | Gemini 4, external set | 2026-11-03 |

## What is not in this repository

- **The images.** They are published by the Historiographical Institute, The University of Tokyo, via IIIF;
  the Canvas URIs in the outputs point to them.
- **The annotations** (regions and transcriptions, `rows.tsv`), and outputs derived only from them
  (the trained weights, readings inside the annotated regions). Whether to release them is being discussed with the co-authors.
  The evaluation scripts read `data/rows.tsv` and cannot be run without it.
- The original export of the annotation tool (IIIF Annotator).

## Reproducing

Python 3.11. The detection scripts need the respective tools
(NDL Kotenseki OCR-Lite: set `NDLKOTEN_LITE_SRC` to its `src` directory; Ultralytics YOLO11; Surya 0.17.1).
`load.py` and `rows.py` build `data/rows.tsv` from the annotation export
(`ANNOTATION_BACKUP`, `MANIFEST_MAP`, `MEDIA_CSV`).

With `data/rows.tsv` in place, run from `scripts/`:

```
python3 paper_tables.py
python3 vlm_subset.py
python3 eval_read.py ndl old new   # the "gt" row (reading inside the annotated regions) needs the annotations
python3 review-2026-11-16/paper_numbers.py
```

## License

Code: MIT License (see `LICENSE`). Machine outputs in `data/`: CC BY 4.0.
