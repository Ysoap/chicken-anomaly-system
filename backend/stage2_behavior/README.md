# Stage 2: Behavior Classification (VideoMAE)

Klasifikasi perilaku ayam per ekor dari klip video: inputnya tracklet
per ayam dari Stage 1 (deteksi RT-DETRv2-S + ByteTrack) → klip 16 frame
→ **VideoMAE-Small** → label perilaku + confidence. Outputnya dipakai
Stage 3 (aturan durasi perilaku → aman / dicurigai / sakit).

**Status 10 Nov 2026: SETUP SELESAI, TRAINING BELUM.** Yang sudah ada:
data prep + manifest + split (angka riil, terverifikasi), clip pipeline
(teruji di video sample Stage 1), notebook training Colab, dan smoke test
plumbing. Yang belum: fine-tuning VideoMAE (jalan di Colab oleh user),
evaluasi test set, dan ambang perilaku Stage 3.

## Yang benar-benar dipakai (bukan spesifikasi awal)

Dokumen awal repo menspesifikasikan VideoMAE-Small fine-tune 4 kelas
(RESTING/INGESTIVE/ACTIVE/AGONISTIC) dari 15 kelas ChickenAct versi
whitepaper. Dua hal dari spesifikasi itu **tidak sesuai kenyataan**
dan sudah diperbaiki di sini (bukan disembunyikan):

1. **Checkpoint awal:** `MCG-NJU/videomae-small` (self-supervised saja)
   **TIDAK ADA** di Hugging Face. Yang tersedia untuk varian Small
   adalah `MCG-NJU/videomae-small-finetuned-kinetics` (ViT-Small,
   ~22M parameter, hidden 384, 12 layer, input 16 frame @ 224x224,
   tubelet 2, patch 16 — dicek langsung dari `config.json` di HF).
   Itulah titik awalnya: bobot backbone ditransfer, classifier head
   diinisialisasi ulang untuk kelas perilaku (`ignore_mismatched_sizes`).
2. **15 kelas ChickenAct yang sebenarnya BERBEDA dari daftar
   whitepaper** (diperiksa langsung dari `ChickenAct.zip`, md5
   terverifikasi — lihat bagian Dataset). Akibatnya pemetaan 15→4
   direvisi total; lihat bagian Taksonomi.

## Taksonomi — keputusan (10 Nov 2026)

Dipakai versi whitepaper/part 2 soal **kategori**-nya (RESTING /
INGESTIVE / ACTIVE / AGONISTIC — 4 kategori relevansi kesejahteraan
yang dipakai Stage 3 untuk diagnosis), tapi **isi tiap kategori
mengikuti nama kelas yang benar-benar ada di data**, bukan daftar
whitepaper. Alasan: whitepaper menulis kelas yang tidak ada di zip
(`sitting`, `pecking_other`, `fighting`, `head_shaking`,
`erratic movement / distress`), sementara zip berisi kelas lain
(`feeding`, `litter_pecking`, `litter_scratching`, `stretching`,
`head_scratching`, `body_shaking`, `pooping`).

Pemetaan yang dipakai (`data_prep/label_mapping.py`):

| Kelas sistem | Kelas ChickenAct (jumlah klip) | Total |
|---|---|---|
| INGESTIVE (0) | feeding 1.924, drinking 1.373, litter_pecking 858, litter_scratching 577 | **4.732** |
| ACTIVE (1) | walking 1.515, running 629, wing_flapping 322 | **2.466** |
| RESTING (2) | resting 2.419, preening 1.917, stretching 1.202, standing 872, dust_bathing 530, head_scratching 500, body_shaking 352 | **7.792** |
| AGONISTIC (3) | — tidak ada data — | **0** |

- `feeding` = eating (whitepaper); `litter_pecking` = pecking ground;
  `litter_scratching` = foraging (mengais litter cari pakan — komponen
  appetitive dari feeding, sesuai etogram ayam standar).
- `stretching`, `head_scratching`, `body_shaking` = comfort / perawatan
  tubuh (seperti preening) → RESTING.
- **Pooping (260 klip, 1,7%) DIKECUALIKAN dari training** — peristiwa
  fisiologis sesaat (buang kotoran), bukan state perilaku kesejahteraan;
  tidak masuk ke satu pun dari 4 kategori. Dicatat di
  `label_mapping.EXCLUDED_CLASSES` dan di laporan distribusi.
- **AGONISTIC (id 3) DICADANGKAN — tidak ada data latih.** ChickenAct
  tidak berisi satu pun klip fighting / pecking_other / distress, jadi
  classifier Stage 2 HANYA melatih **3 kelas** (INGESTIVE/ACTIVE/
  RESTING). `train.py` menurunkan `num_labels` dari manifest (bukan
  hardcode), jadi kalau nanti ada data agresi, pipeline yang sama
  melatih 4 kelas tanpa perubahan kode. **Konsekuensi untuk Stage 3:**
  aturan diagnosis yang butuh "proporsi Agonistic tinggi" (pecking
  outbreak) untuk sementara TIDAK bisa mengandalkan output VideoMAE —
  harus dari fitur lain (variansi kecepatan, overlap bbox) atau data
  tambahan. Ini pekerjaan terbuka, dicatat di bawah.

Dua versi taksonomi lain yang beredar di repo (di sebagian dokumen
Stage 2: `eating/drinking/walking/resting`; varian penamaan di
`pipeline.py`) **tidak dipakai**. Nama whitepaper tetap dikenali
sebagai alias di `label_mapping.py` supaya tidak pecah bila muncul di
data lain.

## Dataset

**ChickenAct** (ChickenVerse, CC BY-NC-SA 4.0, Zenodo record 20672799,
DOI 10.5281/zenodo.20672799) — 15.250 klip video MP4 single-bird,
15 kelas. Klip dihasilkan pipeline detect-track-crop (RT-DETRv2 +
Norfair) lalu dilabeli manual; tiap klip adalah crop kecil di sekitar
ayam (bukan frame kamera utuh).

- File: `ChickenAct.zip`, **1.974.664.090 byte**,
  **md5 `fa5962dfd9cf99df1b08df613891322c`** (terverifikasi 10 Nov 2026,
  sekali unduh dari `https://zenodo.org/api/records/20672799/files/ChickenAct.zip/content`).
  Salinan lokal terverifikasi: `~/workspace/pbl-ayam-obs-rtsp/ChickenAct.zip`.
- Properti klip (dicek dari contoh per kelas): MPEG-4, **25 fps**,
  **40–47 frame (~1,6–1,9 detik)**, resolusi bervariasi per klip
  (~100–210 px, crop di sekitar ayam). Detail: `data_prep/prepared/clip_check.json`.
- Split resmi ChickenAct (train 11.278 / val 1.829 / test 2.143)
  **tidak bisa direproduksi dari zip** (tidak ada manifest split di
  dalamnya), jadi dipakai split acak stratified kami sendiri dengan
  proporsi yang sama (seed 42): **train 11.094 / val 1.799 / test 2.097**
  (dari 14.990 klip yang masuk training). Test = held-out, hanya untuk
  evaluasi akhir — seperti test set Stage 1.
- **Catatan imbalance:** RESTING 52% dari data latih (7.792/14.990),
  ACTIVE hanya 16%. Model bisa bias ke RESTING; opsi ke depan: class
  weight / sampling seimbang (belum diterapkan — dicatat terbuka).
- **Upload ke Drive akun lama GAGAL + DIBATALKAN.** Percobaan upload
  2x ke Drive `pandunugrahaputra@gmail.com` gagal (HTTP 500 — CLI
  Drive hanya mendukung multipart upload, file 2 GB konsisten putus;
  bukan kuota). Lalu user memutuskan **ganti akun Google untuk
  Colab/Drive Stage 2 ke `pandumanis72@gmail.com`** supaya Drive akun
  lama (sisa ~4,1 GB) tidak penuh — upload ke akun lama di-pause dan
  tidak akan dilakukan. Per 10 Nov 2026 ~15:04 WIB, akun baru sudah
  terhubung (limit 400 GB) dan ChickenAct.zip sedang di-upload ke
  `PBL-Ayam/dataset/` akun baru; salinan lokal terverifikasi tetap ada
  di VM sebagai cadangan.

### Data prep

```bash
# dari backend/stage2_behavior/data_prep/
python prepare_chickenact.py --zip /path/ke/ChickenAct.zip \
    --extract-dir /path/ke/ChickenAct \
    --out-dir prepared \
    --expected-md5 fa5962dfd9cf99df1b08df613891322c
```

Output di `data_prep/prepared/` (sudah ter-commit): `manifest.json`
(14.990 entri: path relatif, label15, label4, label4_id, split),
`class_distribution.json`, `clip_check.json`. Script gagal dengan pesan
jelas kalau ada folder kelas yang tidak terpetakan (jangan ditebak).

```bash
# Ekstrak 16 frame/klip (sekali saja; training baca JPEG, cepat)
python extract_frames.py --manifest prepared/manifest.json \
    --data-root /path/ke/ChickenAct \
    --out-dir /path/ke/frames --workers 4
# -> frames_manifest.json (dipakai train.py). Klip <16 frame di-padding
#    dengan frame terakhir (tercatat di manifest).
```

## Training (Colab — oleh user)

Notebook: `notebooks/Colab-ChickenAct-VideoMAE.ipynb` (12 cell, pola
yang sama seperti notebook Stage 1: mount Drive, config di satu cell,
sync checkpoint tiap 120 detik, bisa resume).

- **Akun: `pandumanis72@gmail.com`** (BUKAN akun lama). Saat mount
  Drive, pilih akun itu. Checkpoint di-sync ke
  `MyDrive/PBL-Ayam/runs/stage2-videomae/` di Drive akun baru.
  Akun baru ini **sudah terhubung** (limit 400 GB, sisa ~384 GB) dan
  struktur `PBL-Ayam/{dataset, runs/{stage1-rtdetrv2-s,
  stage1-v1-backup, stage2-videomae}, notebooks}` sudah disiapkan;
  file-file Stage 1 sudah pindah ke sana (akun lama tetap utuh sebagai
  backup). Tidak ada file ID Drive yang di-hardcode.
- **Dataset — dua cara** (notebook menangani keduanya):
  - **(A)** `ChickenAct.zip` sudah ada di Drive akun baru (upload
    sekali dari komputer ke folder mana saja di Drive itu — notebook
    menemukannya berdasarkan **nama file**): dipakai dari Drive.
    (Per 10 Nov 2026 ~15:04 WIB, ChickenAct.zip sedang di-upload ke
    `PBL-Ayam/dataset/` akun baru — kalau sudah selesai, cara ini
    langsung bisa dipakai.)
  - **(B)** Belum ada di Drive: notebook mengunduh langsung dari
    **Zenodo** ke sesi Colab (±10 menit, lalu verifikasi ukuran + md5).
    Kamu tidak wajib upload file 2 GB secara manual. Kalau
    `SAVE_ZIP_TO_DRIVE = True`, zip hasil unduhan juga disalin ke Drive
    akun baru supaya sesi berikut bisa pakai cara (A).
  - Cara memindah zip ke akun baru (bila mau cara A dan belum ada):
    unduh `ChickenAct.zip` dari Zenodo (atau dari salinan lokal
    terverifikasi) lalu upload ke `drive.google.com` dengan akun
    `pandumanis72` — folder bebas, notebook mencari berdasarkan nama file.
- **Config default:** VideoMAE-Small dari
  `MCG-NJU/videomae-small-finetuned-kinetics`, 10 epoch, LR 5e-5
  (cosine, warmup 10%), batch train 16 / eval 32, fp16 (T4), seed 42.
  Perkiraan kasar: 1–3 jam untuk 10 epoch di T4 gratis (pantau epoch
  pertama); kalau sesi putus / limit habis, jalankan lagi notebook dari
  awal — training resume dari checkpoint Drive (optimizer di-restart,
  dicatat di notebook).
- **Standalone** (tanpa Drive — mesin sendiri / Colab):
  `python train.py --frames-manifest <path>/frames_manifest.json --data-root <path>/frames --output-dir runs/stage2_v1 --epochs 10`
  (tambah `--resume auto` untuk lanjut dari checkpoint).
- Output: checkpoint Trainer per epoch + model terbaik (`best/`) +
  `train_metrics.json` (akurasi val + per kelas).
- **Masalah 10 Nov 2026 (sudah diperbaiki):** run pertama training mati
  di awal — `TypeError: TrainingArguments got an unexpected keyword
  argument 'warmup_ratio'`, exit code 1, tanpa checkpoint; cell notebook
  tetap ke-centang dan print "Training selesai" walau gagal.
  Penyebab: `warmup_ratio` baru ada di transformers 4.46, tapi pin
  notebook (`>=4.40`) tidak memaksa upgrade kalau versi lama sudah
  terpasang di Colab; transformers 5.x juga menghapus keyword itu.
  Perbaikan (commit `a346e64`): `train.py` (`make_training_args()`)
  mencoba `warmup_ratio` dulu, kalau ditolak ulangi tanpa warmup —
  jadi jalan di transformers lama, 4.46+, maupun 5.x; pin notebook
  jadi `>=4.46`; cell training sekarang raise `RuntimeError` kalau
  exit code != 0 (gagal keras, bukan pura-pura selesai).

## Inference & clip pipeline

```bash
# Satu klip (mis. klip ChickenAct):
python inference.py --clip /path/ke/klip.mp4 --checkpoint <model>

# Alur penuh di video kandang (butuh tracks.json Stage 1):
python inference.py --video /path/ke/video.mp4 --tracks /path/ke/tracks.json \
    --checkpoint <model> --out-dir out/
# -> track_predictions.json: label + confidence per klip, voting mayoritas
#    + rata-rata probabilitas per track.
```

`clip_builder.py` membangun klip 16 frame dari tracklet Stage 1:
crop persegi di sekitar bbox (margin 1,3x) → resize 224x224. Ada satu
gotcha yang sudah ditangani: bbox di `tracks.json` bisa berada di ruang
koordinat asli frame (1400x900) sementara videonya re-encode lebih
kecil (960x618) — builder mendeteksi dan menskalakan otomatis (dan
mencatat skalanya di log).

**Uji clip builder (10 Nov 2026):** video sample fCam16 Stage 1
(72 frame @10 fps) + `tracks_fCam16.json` (85 ID unik) → **120 klip**
terbentuk (jendela 16 frame, stride 8); contact sheet dicek visual —
ayam berada di tengah crop di semua 16 frame. Uji lengkap di
`tests/test_stage2.py`.

**PERINGATAN — label sebelum fine-tuning tidak ada artinya:** kalau
`--checkpoint` masih checkpoint pretrained Kinetics (bukan hasil
fine-tune ChickenAct), classifier head-nya acak untuk kelas perilaku —
label yang keluar hanya smoke test plumbing. `inference.py` mencetak
peringatan ini setiap kali checkpoint pretrained dipakai.

## Tes

```bash
# dari root repo:
python -m pytest tests/test_stage2.py -v
```

9 test (10 Nov 2026, **10/10 lolos**): mapping 15→4 (kelas asli +
pooping dikecualikan + alias whitepaper), sampling frame, preprocessing
(bentuk [16,3,224,224]), clip builder (video sintetis + skala
koordinat), data prep (manifest + split + eksklusi), extract frames,
dan plumbing VideoMAE (load checkpoint Kinetics di CPU + klasifikasi 1
klip sintetis; ±6,5 menit di CPU karena unduh model ~90 MB sekali).

Smoke test inference end-to-end (10 Nov 2026): `inference.py --video
+ --tracks` di video sample fCam16 (120 klip, 43 track) jalan sampai
selesai (±8 menit CPU); distribusi label nyaris seragam (INGESTIVE 87,
RESTING 29, ACTIVE 3, AGONISTIC 1) — sesuai ekspektasi classifier head
yang masih acak; label belum bermakna sebelum fine-tune.

## Evaluasi test set (11 Nov 2026)

Training selesai 11 Nov 00:15 WIB (10/10 epoch, val terbaik 92,94%).
Model terbaik (`models/`, best checkpoint dari Drive
`runs/stage2-videomae/best/`) dievaluasi di **2.097 klip test held-out**
— split yang SAMA dengan training (manifest seed 42; test tidak pernah
dipakai training/val) — dengan `eval_test_set.py` (batch 8, CPU,
±84 menit di VM; hasil lengkap `results/test_set_eval.json`).

- **Akurasi test: 91,99%** (1.929/2.097 benar) vs val 92,94% —
  selisih hanya ~1 poin: model menggeneralisasi dengan baik ke data
  yang belum pernah dilihat; tidak ada jurang val/test yang besar.
- Per kelas: **ACTIVE 95,07%** (328/345 — paling mudah dibedakan),
  **INGESTIVE 91,84%** (608/662), **RESTING 91,10%** (993/1.090 —
  paling rendah walau datanya paling banyak).
- Confusion matrix (baris = benar, kolom = prediksi): kesalahan
  terbesar **INGESTIVE ↔ RESTING** — 46 klip INGESTIVE diprediksi
  RESTING dan 79 klip RESTING diprediksi INGESTIVE; total 125 dari
  168 kesalahan (~74%). Masuk akal secara perilaku: litter
  pecking/scratching (kepala menunduk mematuk) vs preening/resting
  (kepala menunduk, diam) sulit dibedakan dalam klip 16 frame, dan
  RESTING adalah kelas paling heterogen (7 sub-perilaku). ACTIVE
  hampir tidak pernah tertukar (hanya 17 dari 345 klip salah).
- Catatan jujur: train loss akhir ~0 (model menghafal data train),
  eval loss naik pelan antar epoch (0,38 → 0,44) — tanda overfitting
  ringan — tapi pemilihan model terbaik memakai akurasi val, jadi
  checkpoint yang disimpan tetap yang terbaik; angka test di atas
  yang membuktikan overfitting itu tidak merusak performa di data
  baru. RESTING 52% data tidak membuat akurasinya paling tinggi —
  imbalance bukan penjelasan untuk kelemahan RESTING; heterogenitas
  kelasnya yang lebih menentukan.

## Uji inference video sample (11 Nov 2026)

Video `fCam16_sample.mp4` (siang, 72 frame @10 fps, 7,2 dtk) + tracks
Stage 1 (`tracks_fCam16.json`, 85 ID unik), model fine-tune `models/`,
`inference.py --video` di CPU (~7 menit, 120 klip):

- **120 klip dari 43 track terklasifikasi** — 42 ID lainnya tidak punya
  jendela 16 frame berurutan penuh (ID switch / track pendek), jadi
  tidak dapat klip dan tidak terklasifikasi. Itu keterbatasan tracking,
  bukan model.
- Distribusi klip: **RESTING 66, ACTIVE 46, INGESTIVE 8**; mayoritas
  per track: RESTING 29, ACTIVE 12, INGESTIVE 2. Confidence rata-rata
  klip 0,97 (tinggi — ingat, ini confidence model sendiri, bukan
  akurasi; video ini tidak punya ground truth perilaku).
- Video berlabel: `results/fCam16_annotated.mp4` (960x618, 7,2 dtk,
  10 MB) — kotak + `ID <n> <LABEL> <conf>` per frame; warna hijau
  INGESTIVE, oranye ACTIVE, biru RESTING; track tanpa klip hanya
  `ID <n>` (kotak putih). Dibuat dengan `render_annotated_video.py`
  (label per frame dari klip yang mencakup frame itu, fallback label
  mayoritas track). Hasil: `results/infer_fcam16/track_predictions.json`.
- Cek visual (frame 0/20/40/60): kotak nempel di ayam yang benar
  (skala bbox 1400x900 → 960x618 diterapkan, sama seperti clip
  builder); campuran label masuk akal — INGESTIVE muncul di dekat
  tempat pakan, bukan semua satu label. **Catatan jujur:** tidak ada
  ground truth untuk video ini, jadi kebenaran label per track tidak
  bisa dihitung — yang terverifikasi baru: plumbing end-to-end jalan
  (Stage 1 → klip → VideoMAE → label), kotak sejajar, dan distribusi
  label tidak kolaps ke satu kelas. Satu track sempat terbelah
  (RESTING 1 vs ACTIVE 1, mean prob 0,50 vs 0,50) — label mayoritas
  di track seperti itu tidak stabil dan jangan dipakai mentah-mentah
  di Stage 3 tanpa ambang confidence.

## Yang belum dikerjakan

- **Fine-tuning VideoMAE** — SELESAI 11 Nov 00:15 WIB (10 epoch,
  val 92,94%); **evaluasi test set** — SELESAI 11 Nov (91,99%,
  lihat di atas); **model hasil fine-tune** — TERSIMPAN di `models/`
  (Git LFS, `model.safetensors` 87 MB + `config.json`).
- **Uji `inference.py` end-to-end di video sample** — SELESAI 11 Nov
  (lihat di atas: 120 klip, 43 track, video berlabel
  `results/fCam16_annotated.mp4`).
- **Kelas AGONISTIC** — tidak ada data latih di ChickenAct (lihat
  Taksonomi); deteksi agresi Stage 3 butuh data tambahan atau fitur
  non-VideoMAE.
- **Imbalance RESTING 52%** — pertimbangkan class weight / sampling
  seimbang bila akurasi per kelas timpang.
- **Ambang perilaku Stage 3** (durasi → aman/dicurigai/sakit) — masih
  generik, menunggu distribusi data + keputusan user.
