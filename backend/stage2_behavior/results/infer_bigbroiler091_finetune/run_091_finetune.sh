#!/bin/bash
# Uji video 091 (top-view) dengan model FINE-TUNE v2.
# Prosedur SAMA seperti baseline 091 (commit 76b7499), hanya checkpoint diganti
# models (Stage 2) -> models_finetune (fine-tune v2), dan inference langsung
# pada tracks TERFILTER (17 track, 6.119 klip) — tanpa mengulang tracking
# Stage 1 dan tanpa filter ulang.
# Dijalankan SETELAH evaluasi test set klip selesai (CPU VM terbatas);
# kalau mati SIGTERM, jalankan lagi script ini — inference resume dari JSONL.
set -u
cd ~/workspace/pbl-ayam-obs-rtsp/chicken-anomaly-system
VENV=~/workspace/.venv-rtdetr/bin/python
VIDEO=~/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_720p_10fps.mp4
TRACKS=backend/stage2_behavior/results/infer_bigbroiler091/merged_tracks_bigbroiler091.filtered.json
GT=~/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_3_32_1920_008_0_10.csv
OUT=backend/stage2_behavior/results/infer_bigbroiler091_finetune
mkdir -p "$OUT"

echo "=== [1/4] Inference Stage 2 (model fine-tune, 6.119 klip) ==="
$VENV backend/stage2_behavior/infer_video_batched.py \
  --video "$VIDEO" --tracks "$TRACKS" \
  --checkpoint backend/stage2_behavior/models_finetune \
  --out-dir "$OUT" --stride 8 --margin 1.3 --resume \
  > "$OUT/infer_full.log" 2>&1 || { echo "INFERENCE FAILED rc=$?"; exit 1; }
echo "inference OK: $($VENV -c "import json;print(json.load(open('$OUT/track_predictions.json'))['n_clips'])" ) klip"

echo "=== [2/4] Render video berlabel ==="
$VENV backend/stage2_behavior/render_annotated_video.py \
  --video "$VIDEO" \
  --tracks "$TRACKS" \
  --predictions "$OUT/track_predictions.json" \
  --out "$OUT/bigbroiler091_annotated_finetune.mp4" \
  > "$OUT/render_full.log" 2>&1 || { echo "RENDER FAILED rc=$?"; exit 1; }
echo "render OK"

echo "=== [3/4] Banding GT (metode sama seperti baseline) ==="
$VENV backend/stage2_behavior/compare_gt.py \
  --tracks "$TRACKS" \
  --predictions "$OUT/track_predictions.json" \
  --gt "$GT" --map-explor-forag ACTIVE \
  --out "$OUT/gt_comparison.json" \
  > "$OUT/gt_compare.log" 2>&1 || { echo "GT COMPARE FAILED rc=$?"; exit 1; }
echo "gt compare OK"

echo "=== [4/4] Metrik per paruh (paruh-2 = angka jujur) ==="
$VENV backend/stage2_behavior/analyze_gt_halves.py \
  --tracks "$TRACKS" \
  --predictions "$OUT/track_predictions.json" \
  --gt "$GT" \
  --out "$OUT/gt_halves.json" \
  > "$OUT/gt_halves.log" 2>&1 || { echo "HALVES FAILED rc=$?"; exit 1; }
cat "$OUT/gt_halves.log"
echo "ALL DONE"
