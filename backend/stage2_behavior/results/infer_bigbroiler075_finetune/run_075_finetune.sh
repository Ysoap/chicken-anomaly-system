#!/bin/bash
# Uji video 075 (sudut miring, HELD-OUT) dengan model FINE-TUNE v2.
# Prosedur SAMA seperti uji 080 fine-tune (commit 51d5d6c), hanya video +
# tracks + GT diganti 080 -> 075; tracking Stage 1 SUDAH dikerjakan dulu
# (run_075_track.sh: 6.066 frame, 61.878 box, 100 ID mentah -> 62 ID
# sesudah filter statis 10 px).
# 075 TIDAK dipakai di training fine-tune v2 (train: 091, 080, 067, 057,
# 060, 064) — jadi angka GT di sini ADALAH ukuran generalisasi yang valid,
# tidak seperti 080 yang terkontaminasi data latih.
# Langkah: 1) inference 7.325 klip (62 ID, termasuk drinker ID 33+112),
#          2) buang drinker -> tracks nofp (60 ID) + 6.851 klip,
#          3) render video berlabel, 4) banding GT (--force, seperti 080),
#          5) uji sensitivitas (IoU 0,5 + Explor_Forag -> INGESTIVE).
set -u
cd ~/workspace/pbl-ayam-obs-rtsp/chicken-anomaly-system
VENV=~/workspace/.venv-rtdetr/bin/python
VIDEO=~/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_075_720p_10fps.mp4
TRACKS=backend/stage2_behavior/results/infer_bigbroiler075_finetune/tracks_bigbroiler075.filtered.json
GT=~/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_075_2_38_1920_026_0_10.csv
OUT=backend/stage2_behavior/results/infer_bigbroiler075_finetune

echo "=== [1/5] Inference Stage 2 (model fine-tune) ==="
$VENV backend/stage2_behavior/infer_video_batched.py \
  --video "$VIDEO" --tracks "$TRACKS" \
  --checkpoint backend/stage2_behavior/models_finetune \
  --out-dir "$OUT" --stride 8 --margin 1.3 --resume \
  > "$OUT/infer_full.log" 2>&1 || { echo "INFERENCE FAILED rc=$?"; exit 1; }
echo "inference OK: $(python3 -c "import json;print(json.load(open('$OUT/track_predictions.json'))['n_clips'])" ) klip"

echo "=== [2/5] Buang drinker ID 33 + 112 ==="
$VENV "$OUT/remove_drinker_tracks.py" || { echo "REMOVE DRINKER TRACKS FAILED"; exit 1; }
$VENV "$OUT/filter_drinker.py" || { echo "FILTER DRINKER FAILED"; exit 1; }

echo "=== [3/5] Render video berlabel ==="
$VENV backend/stage2_behavior/render_annotated_video.py \
  --video "$VIDEO" \
  --tracks "$OUT/tracks_bigbroiler075.filtered_nofp.json" \
  --predictions "$OUT/track_predictions_filtered.json" \
  --out "$OUT/bigbroiler075_annotated_filtered.mp4" \
  > "$OUT/render_full.log" 2>&1 || { echo "RENDER FAILED rc=$?"; exit 1; }
echo "render OK"

echo "=== [4/5] Banding GT (pakai --force, seperti 080) ==="
$VENV backend/stage2_behavior/compare_gt.py \
  --tracks "$OUT/tracks_bigbroiler075.filtered_nofp.json" \
  --predictions "$OUT/track_predictions_filtered.json" \
  --gt "$GT" --map-explor-forag ACTIVE --force \
  --out "$OUT/gt_comparison.json" \
  > "$OUT/gt_compare.log" 2>&1 || { echo "GT COMPARE FAILED rc=$?"; exit 1; }
echo "gt compare OK"
python3 -c "import json;d=json.load(open('$OUT/gt_comparison.json'));print('accuracy:',d['accuracy_matched'],'matched:',d['n_clips_matched'],'/',d['n_clips_total'],'coverage:',round(d['coverage'],4));print('per_class:',{k:(round(v['recall'],4) if v['recall'] is not None else None,round(v['precision'],4) if v['precision'] is not None else None) for k,v in d['per_class'].items()})"

echo "=== [5/5] Uji sensitivitas ==="
$VENV backend/stage2_behavior/compare_gt.py \
  --tracks "$OUT/tracks_bigbroiler075.filtered_nofp.json" \
  --predictions "$OUT/track_predictions_filtered.json" \
  --gt "$GT" --map-explor-forag ACTIVE --force --iou-thresh 0.5 \
  --out "$OUT/gt_comparison_iou05.json" \
  > "$OUT/gt_compare_iou05.log" 2>&1 || { echo "GT IOU05 FAILED rc=$?"; exit 1; }
$VENV backend/stage2_behavior/compare_gt.py \
  --tracks "$OUT/tracks_bigbroiler075.filtered_nofp.json" \
  --predictions "$OUT/track_predictions_filtered.json" \
  --gt "$GT" --map-explor-forag INGESTIVE --force \
  --out "$OUT/gt_comparison_explor_ing.json" \
  > "$OUT/gt_compare_explor_ing.log" 2>&1 || { echo "GT EXPLOR-ING FAILED rc=$?"; exit 1; }
python3 -c "import json;a=json.load(open('$OUT/gt_comparison_iou05.json'));b=json.load(open('$OUT/gt_comparison_explor_ing.json'));print('iou05 accuracy:',a['accuracy_matched'],'matched:',a['n_clips_matched']);print('explor->ing accuracy:',b['accuracy_matched'],'matched:',b['n_clips_matched'])"
echo "ALL DONE"
