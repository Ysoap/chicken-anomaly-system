#!/bin/bash
# Tracking Stage 1 video 075 miring (Big Broiler, setup 2, broiler 38 hari).
# Video pipeline-ready: oblique_bigbroiler_075_720p_10fps.mp4 — 6.066 frame
# @10 fps, 1280x720. Threshold 0.6, track_buffer 30, filter track statis
# default AKTIF (ambang gerak pusat bbox 10 px) — sama seperti 091/080.
# Checkpoint tiap 600 frame ke tracks_bigbroiler075.json (+ .tracker.pkl);
# kalau file sudah ada, lanjutkan dari situ (--resume-from).
# Log di-append ke track_full.log (bukan overwrite) supaya riwayat restart
# tidak hilang.
set -u
cd ~/workspace/pbl-ayam-obs-rtsp/chicken-anomaly-system
VENV=~/workspace/.venv-rtdetr/bin/python
VIDEO=~/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_075_720p_10fps.mp4
OUT=backend/stage2_behavior/results/infer_bigbroiler075_finetune
TRACKS="$OUT/tracks_bigbroiler075.json"
echo "=== TRACK 075 mulai $(date '+%F %T %Z') ===" >> "$OUT/track_full.log"
EXTRA=""
if [ -f "$TRACKS" ]; then
  echo "resume dari $TRACKS" >> "$OUT/track_full.log"
  EXTRA="--resume-from $TRACKS"
fi
$VENV backend/stage1_detection/track_video.py \
  --video "$VIDEO" --out "$TRACKS" --thresh 0.6 $EXTRA \
  >> "$OUT/track_full.log" 2>&1
RC=$?
echo "=== TRACK 075 selesai rc=$RC $(date '+%F %T %Z') ===" >> "$OUT/track_full.log"
exit $RC
