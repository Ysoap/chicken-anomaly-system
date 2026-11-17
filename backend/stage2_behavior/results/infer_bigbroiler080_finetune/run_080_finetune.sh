#!/bin/bash
# Uji video 080 (sudut miring) dengan model FINE-TUNE v2.
# Prosedur SAMA seperti baseline 080 (commit 11fc5a9), hanya checkpoint diganti
# models (Stage 2) -> models_finetune (fine-tune v2).
# Dijalankan SETELAH uji video 091 fine-tune selesai (CPU VM terbatas).
# Langkah: 1) inference 7.432 klip (52 ID, termasuk silo ID 40),
#          2) buang klip silo ID 40 -> 6.936 klip,
#          3) render video berlabel, 4) banding GT (dengan --force, seperti baseline).
set -u
cd ~/workspace/pbl-ayam-obs-rtsp/chicken-anomaly-system
VENV=~/workspace/.venv-rtdetr/bin/python
VIDEO=~/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_080_720p_10fps.mp4
TRACKS=backend/stage2_behavior/results/infer_bigbroiler080/tracks_bigbroiler080.filtered.json
GT=~/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_080_2_40_2688_024_0_10.csv
OUT=backend/stage2_behavior/results/infer_bigbroiler080_finetune

echo "=== [1/4] Inference Stage 2 (model fine-tune) ==="
$VENV backend/stage2_behavior/infer_video_batched.py \
  --video "$VIDEO" --tracks "$TRACKS" \
  --checkpoint backend/stage2_behavior/models_finetune \
  --out-dir "$OUT" --stride 8 --margin 1.3 --resume \
  > "$OUT/infer_full.log" 2>&1 || { echo "INFERENCE FAILED rc=$?"; exit 1; }
echo "inference OK: $(python3 -c "import json;print(json.load(open('$OUT/track_predictions.json'))['n_clips'])" ) klip"

echo "=== [2/4] Buang klip silo ID 40 ==="
cp backend/stage2_behavior/results/infer_bigbroiler080/tracks_bigbroiler080.filtered_nofp.json "$OUT/" \
  || { echo "COPY NOFP TRACKS FAILED"; exit 1; }
$VENV "$OUT/filter_silo40.py" || { echo "FILTER SILO FAILED"; exit 1; }

echo "=== [3/4] Render video berlabel ==="
$VENV backend/stage2_behavior/render_annotated_video.py \
  --video "$VIDEO" \
  --tracks "$OUT/tracks_bigbroiler080.filtered_nofp.json" \
  --predictions "$OUT/track_predictions_filtered.json" \
  --out "$OUT/bigbroiler080_annotated_filtered.mp4" \
  > "$OUT/render_full.log" 2>&1 || { echo "RENDER FAILED rc=$?"; exit 1; }
echo "render OK"

echo "=== [4/4] Banding GT (pakai --force, seperti baseline) ==="
$VENV backend/stage2_behavior/compare_gt.py \
  --tracks "$OUT/tracks_bigbroiler080.filtered_nofp.json" \
  --predictions "$OUT/track_predictions_filtered.json" \
  --gt "$GT" --map-explor-forag ACTIVE --force \
  --out "$OUT/gt_comparison.json" \
  > "$OUT/gt_compare.log" 2>&1 || { echo "GT COMPARE FAILED rc=$?"; exit 1; }
echo "gt compare OK"
python3 -c "import json;d=json.load(open('$OUT/gt_comparison.json'));print('accuracy:',d['accuracy_matched'],'matched:',d['n_clips_matched'],'/',d['n_clips_total'],'coverage:',round(d['coverage'],4));print('per_class:',{k:(round(v['recall'],4),round(v['precision'],4)) for k,v in d['per_class'].items()})"
echo "ALL DONE"
