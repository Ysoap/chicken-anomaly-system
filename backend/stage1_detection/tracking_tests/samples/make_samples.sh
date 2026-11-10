#!/bin/bash
# Rakit ulang 4 video sample uji tracking multi-video dari ChickenDet.zip.
# Dipakai saat uji 10 Nov 2026 (lihat tracking_tests/README.md).
# Jalankan dari direktori mana pun; butuh file ChickenDet.zip dan ffmpeg.
# Hasil: 4 file MP4 10 fps (1400x900, CRF 23) — versi yang dipakai saat
# tracking dijalankan. Versi ringkas (960 px, CRF 28) yang tersimpan di
# folder ini dibuat dari hasil script ini.
set -e
ZIP="${1:-ChickenDet.zip}"
OUT="${2:-$(dirname "$0")}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for seq in fCam16.20230221_130001 fCam13.20230210_203000 fCam15.20230213_191500 fCam6.20230215_221500; do
  mkdir -p "$TMP/$seq"
  unzip -o -q "$ZIP" "images/Validation/${seq}_*.jpg" -d "$TMP/$seq"
  mv "$TMP/$seq/images/Validation/"*.jpg "$TMP/$seq/"
  ffmpeg -y -v error -framerate 10 -i "$TMP/$seq/${seq}_%04d.jpg" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/$seq.mp4"
  echo "$seq.mp4: $(du -h "$OUT/$seq.mp4" | cut -f1)"
done
