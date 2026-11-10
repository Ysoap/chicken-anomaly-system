# Uji Tracking Multi-Video (ByteTrack) — 10 Nov 2026

Semua artefak uji tracking ByteTrack di beberapa video sample ada di folder
ini (bukan di `examples/` / `results/` — folder itu hanya untuk uji
tunggal pertama). Total folder ini ±18 MB.

## Video sample (`samples/`)

4 sekuens 72 frame dari split Validation ChickenDet, dirakit jadi video
MP4 10 fps dari urutan gambar. Dipilih supaya kamera, tanggal/waktu, dan
kepadatan ayamnya berbeda:

| Sekuens | Kamera / waktu | GT ayam/frame (rata-rata) |
|---|---|---|
| `fCam16.20230221_130001.mp4` | fCam16, 21 Feb 2026 13:00 (siang) | 33,3 — paling padat |
| `fCam13.20230210_203000.mp4` | fCam13, 10 Feb 2026 20:30 (malam) | 30,0 |
| `fCam6.20230215_221500.mp4` | fCam6, 15 Feb 2026 22:15 (malam) | 25,8 |
| `fCam15.20230213_191500.mp4` | fCam15, 13 Feb 2026 19:15 (senja) | 21,7 — paling sepi |

Video di sini adalah **re-encode ringkas** dari rakitan asli (lebar 960 px,
CRF 28) supaya ukuran repo tetap wajar. Rakitan asli (1400×900) bisa
dibuat ulang persis dari `ChickenDet.zip` dengan `make_samples.sh`.

## Cara menjalankan ulang

```bash
# dari backend/stage1_detection/
python inference.py --track --video tracking_tests/samples/fCam16.20230221_130001.mp4 --fps 10 --track-threshold 0.3
```

Parameter sama seperti uji ini: threshold deteksi 0,3, ByteTrack default
(match IoU 0,8, buffer 30 frame), CPU ±2,5 menit per video.

## Hasil (`summaries/`)

`summaries/multi_video_tracking_summary.json` — ringkasan per video
(frame, deteksi/frame, ID unik, ID frame 0 yang bertahan, median umur
track, dst). `summaries/tracks_<sekuens>.json` — track ID per frame
(mentah, hasil `inference.py --track`). Ringkasan lengkap + analisis
perbandingan ada di `../README.md` bagian "Uji tracking multi-video".

## Output visual (`outputs/`)

Hanya **1 video beranotasi penuh** yang disimpan
(`tracked_fCam16.20230221_130001.mp4`, sekuens paling stabil — re-encode
ringkas CRF 28, ±4 MB). Untuk 3 video lain, yang disimpan hanya frame
contoh beranotasi (frame 000 + 048, JPG): box + `#TrackID skor`
terlihat benar di semua video yang dicek.

## Catatan (gotcha)

`inference.py --track` menulis ringkasan ke `results/tracking_summary.json`
dengan path tetap — jadi kalau menjalankan beberapa video berurutan,
file itu akan tertimpa oleh video terakhir. Ringkasan per video yang
benar adalah yang dihitung dari `tracks_<sekuens>.json`
(lihat `summaries/multi_video_tracking_summary.json`), bukan file itu.
