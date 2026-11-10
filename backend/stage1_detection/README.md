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

## Model

`models/best.pth` (322.403.181 byte, md5 `6f8457e56c9147d1ba8efc14a7ba05c2`)
disimpan via **Git LFS** (`.gitignore` repo ini memblokir `*.pth`, jadi ada
pengecualian khusus untuk file ini). Setelah clone:

```bash
git lfs pull
```

## Yang belum dikerjakan

- **ByteTrack** (tracking + Track ID konsisten per ekor) belum terintegrasi —
  `inference.py` saat ini murni deteksi per frame. Ini langkah berikut Stage 1.
- Evaluasi di **test set** (250 gambar held-out) belum dilakukan.
- Uji FPS/latensi di hardware target (RTX 3060) belum dilakukan — angka 30 FPS
  di dokumen masih target, bukan hasil ukur.

## Struktur folder

```
backend/stage1_detection/
├── train.py            # training standalone (port notebook Colab)
├── inference.py        # inference + gambar box prediksi vs GT
├── configs/
│   └── rt_detr_config.yaml   # konfigurasi training yang dipakai
├── models/
│   └── best.pth        # model final (Git LFS)
├── notebooks/
│   └── Colab-ChickenDet-RTDETRv2-Official.ipynb  # notebook training asli
├── examples/           # contoh hasil prediksi (2 gambar)
└── training_log.txt    # log epoch/AP dari training
```
