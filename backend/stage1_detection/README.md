# Stage 1: Detection & Tracking

Deteksi ayam per frame (RT-DETRv2) + tracking per ekor (ByteTrack).
Stage ini adalah "mata" sistem: output-nya (bbox + Track ID) dipakai Stage 2
untuk meng-crop tiap ayam dan mengklasifikasi perilakunya.

## Yang benar-benar dipakai (bukan spesifikasi awal)

Dokumen awal repo ini menspesifikasikan **RT-DETRv2-L** (~110M parameter,
via Hugging Face `transformers`). Yang benar-benar dilatih dan terbukti
jalan adalah **RT-DETRv2-S** — varian kecil, backbone PResNet-18, dari repo
resmi `lyuwenyu/RT-DETR`, config `rtdetrv2_r18vd_120e_coco.yml`, tuning dari
checkpoint COCO resmi (`rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`, AP 48.1).

Alasan praktis: training jalan di **Google Colab gratis (Tesla T4)** — varian
S cukup ringan untuk selesai 50 epoch di sana, dan hasilnya melampaui target
dokumen (AP 0,9261 vs target mAP >0,85). Kalau nanti tersedia GPU RTX 3060
seperti di spesifikasi, varian L bisa dicoba ulang — tapi S sudah memenuhi
kebutuhan Stage 1.

## Dataset

**ChickenDet** (ChickenVerse, CC BY-NC-SA 4.0) — 6.539 gambar / 153.764
instance, 1 kelas (`Chicken`). Split resmi: train 5.222 / val 1.067 / test 250
(test = kandang komersial held-out, hanya evaluasi akhir).

**Temuan kritis:** semua anotasi asli bertanda `iscrowd=1`, dan kode dataset
resmi RT-DETR membuang anotasi `iscrowd=1` — tanpa perbaikan, training jalan
dengan 0 box. Perbaikannya: salin JSON ke `*_fixed.json` dengan `iscrowd` -> 0
saja (bbox/kategori tidak diubah). Lihat `train.py` (`prepare_dataset`).

## Training

Dijalankan 1–10 Nov 2026 di Colab T4 (gratis), dari notebook
`notebooks/Colab-ChickenDet-RTDETRv2-Official.ipynb` (notebook itu yang
dijalankan, lengkap dengan output-nya). `train.py` di folder ini adalah port
standalone dari notebook tersebut (langkah yang sama, bisa jalan di mesin
sendiri / Colab tanpa Drive).

- 50 epoch (epoch di log dihitung 0–49), batch train 16 / val 32, AMP aktif.
- Checkpoint disync ke Drive tiap 120 detik; sesi Colab yang putus / limit
  GPU habis dilanjutkan dengan resume dari `last.pth` (`-r`).
- Pace terukur ±11–13 menit/epoch. Total ±9 jam terbagi beberapa sesi
  (limit GPU gratis Colab habis 5x selama training).

### Hasil (10 Nov 2026, final)

| Metrik | Nilai |
|---|---|
| AP (mAP, 1 kelas) terbaik | **0,9261** (epoch 49) |
| AP50 | 0,9876 |
| train_loss (epoch 49) | 3,6517 |
| Model final | `models/best.pth` (bobot EMA, 322 MB, via Git LFS) |

Log lengkap 65 baris (termasuk epoch yang terulang karena resume):
`training_log.txt`. Epoch 49 = epoch ke-50 (epoch dihitung dari 0) — tidak
ada "epoch 50" sebagai angka.

## Evaluasi test set (10 Nov 2026)

`eval_test_set.py` menjalankan `models/best.pth` (bobot EMA, threshold skor
0,05, maks 300 box/gambar) di **semua 250 gambar test** (kandang komersial
held-out, 12.620 anotasi) dan menghitung metrik dengan pycocotools COCOeval.
Hasil lengkap: `results/test_set_eval.json` (CPU, ±5 menit).

| Metrik | Nilai |
|---|---|
| AP (mAP 0.50:0.95) | **0,8306** |
| AP50 | 0,9674 |
| AP75 | 0,8925 |
| AP small / medium / large | 0,4541 / 0,8355 / 0,9083 |

**Dibaca jujur:** AP test set 0,8306 sedikit di **bawah target dokumen
(mAP > 0,85)** — target itu tercapai di val (0,9261), tapi test split lebih
sulit: kandang komersial yang belum pernah dilihat model, lebih padat
(median 48 ayam/gambar vs ~33 di val), dan ayam kecil/jauh lemah
(AP small 0,4541). AP50 0,9674 tetap tinggi, jadi box-nya cukup akurat
untuk kebutuhan tracking (IoU 0,5). Peningkatan yang masuk akal ke depan:
augmentasi / resolusi input lebih besar untuk objek kecil, atau tambah data
latih dari kandang sejenis — bukan menaikkan threshold secara artifisial.

```bash
python eval_test_set.py --zip /path/ke/ChickenDet.zip \
    --repo-dir /path/ke/RT-DETR/rtdetrv2_pytorch
```

## Cek prediksi contoh

`inference.py` menjalankan `models/best.pth` (bobot EMA — sama seperti saat
evaluasi AP) pada gambar Validation acak, menggambar box prediksi (merah +
skor) dan ground-truth (hijau). Hasil 10 Nov 2026 (threshold 0,5, 5 gambar):

| Gambar | GT | Prediksi |
|---|---|---|
| fCam16.20230220_214500_0011.jpg | 33 | 33 (skor 0,56–0,99) |
| fCam16.20230220_214500_0071.jpg | 34 | 34 (skor 0,93–0,99) |
| fCam16.20230221_130001_0045.jpg | 33 | 33 (skor 0,90–0,99) |
| frame_01_2.49s.jpg | 10 | 10 (skor 0,93–0,98) |
| frame_42_615.36s.jpg | 8 | 8 (skor 0,70–0,94) |

Box merah menempel rapat ke box hijau di pengecekan visual (lihat
`examples/`). Satu catatan: di `frame_42` (anak ayam kecil di bawah lampu
merah) ada 1 ayam kecil yang sebagian tertutup terlewat + 1 false positive
di terpal (skor 0,70). Kesimpulan: **tahap deteksi valid, siap lanjut ke
tahap perilaku/tracking.**

```bash
python inference.py --zip /path/ke/ChickenDet.zip
python inference.py --zip /path/ke/ChickenDet.zip --samples 3 --threshold 0.6
```

## Tracking (ByteTrack) — selesai 10 Nov 2026

`inference.py --track` menjalankan deteksi per frame + **ByteTrack** dari
library `supervision` 0.25.1 (versi terakhir yang masih membawa tracker;
supervision ≥ 0.26 menghapus modul trackers — jadi jangan upgrade
supervision tanpa mengganti tracker). Tidak ada tracker yang ditulis
sendiri. Output: video beranotasi (box + `#TrackID skor`) di `examples/`,
track per frame di `results/tracks.json`, ringkasan di
`results/tracking_summary.json`.

```bash
# dari urutan frame (mis. frame video yang diekstrak)
python inference.py --track --frames-dir /path/ke/frame/ --fps 10
# atau langsung dari file video
python inference.py --track --video /path/ke/video.mp4 --fps 10
```

### Hasil uji (10 Nov 2026)

Video uji: 72 frame validasi berurutan, kamera fCam16
(`fCam16.20230220_214500_0000–0071.jpg`, 1400×900, ~33 ayam/frame),
deteksi threshold 0,3, ByteTrack default (match IoU 0,8, buffer 30 frame).
Jalan di CPU (±2,5 menit).

| Metrik | Nilai |
|---|---|
| Frame | 72 |
| Rata-rata deteksi / frame | 33,96 |
| Track aktif per frame | ~29–35 (frame 0: 33, frame 71: 33) |
| Total Track ID unik | 91 |
| ID baru muncul setelah frame 0 | 58 |
| ID frame 0 yang bertahan sampai frame 71 | 10 dari 33 |
| Median umur track | 33 frame |
| Track hidup ≥ 10 / ≥ 30 frame | 70 / 49 |

**Dibaca jujur:** tracking jalan dan ID-nya konsisten untuk sebagian besar
ayam (median umur track 33 dari 72 frame, 49 track hidup ≥ 30 frame), tapi
**ID switch masih sering** — hanya 10 dari 33 ID awal yang bertahan sampai
akhir klip. Penyebabnya: kandang padat + oklusi berat antar ayam yang
mirip, ayam bergerak cepat (ada yang motion-blur sampai tidak terdeteksi
di satu frame), dan ByteTrack tanpa fitur penampilan (hanya IoU + Kalman).
Untuk Stage 2 ini cukup sebagai fondasi (klip per ayam dipotong dari track),
tapi perilaku yang butuh identitas jangka panjang akan terpotong-potong;
perbaikan ke depan: kamera/frame-rate lebih tinggi, atau tracker dengan
ReID (mis. BoT-SORT).

**Eksperimen tuning (dicatat, bukan dipakai):** match IoU diturunkan ke 0,6
dicoba 10 Nov — hasilnya *lebih buruk* (154 ID unik, median umur track
11,5 frame, hanya 4 ID awal bertahan). Jadi default 0,8 dipertahankan.

Contoh frame beranotasi: `examples/track_frame_000.png`,
`track_frame_024.png`, `track_frame_048.png`, `track_frame_071.png`.

## Model

`models/best.pth` (322.403.181 byte, md5 `6f8457e56c9147d1ba8efc14a7ba05c2`)
disimpan via **Git LFS** (`.gitignore` repo ini memblokir `*.pth`, jadi ada
pengecualian khusus untuk file ini). Setelah clone:

```bash
git lfs pull
```

## Yang belum dikerjakan

- ~~ByteTrack (tracking + Track ID konsisten per ekor)~~ — selesai 10 Nov 2026
  (lihat "Tracking (ByteTrack)" di atas; ID switch masih sering, catatannya
  ada di sana).
- ~~Evaluasi di test set (250 gambar held-out)~~ — selesai 10 Nov 2026
  (AP 0,8306 / AP50 0,9674; AP sedikit di bawah target 0,85 — lihat
  "Evaluasi test set").
- Uji FPS/latensi di hardware target (RTX 3060) belum dilakukan — angka 30 FPS
  di dokumen masih target, bukan hasil ukur. Di CPU VM (±1,2 detik/gambar
  untuk deteksi) jelas bukan hardware target.
- Perbaikan AP objek kecil (AP small 0,4541 di test set) belum dicoba.

## Struktur folder

```
backend/stage1_detection/
├── train.py            # training standalone (port notebook Colab)
├── inference.py        # inference + gambar box prediksi vs GT + mode --track (ByteTrack)
├── eval_test_set.py    # evaluasi test set 250 gambar (COCOeval)
├── configs/
│   └── rt_detr_config.yaml   # konfigurasi training yang dipakai
├── models/
│   └── best.pth        # model final (Git LFS)
├── notebooks/
│   └── Colab-ChickenDet-RTDETRv2-Official.ipynb  # notebook training asli
├── examples/           # contoh hasil prediksi (2 gambar) + video/frame tracking
├── results/            # test_set_eval.json + tracks.json + tracking_summary.json
└── training_log.txt    # log epoch/AP dari training
```
