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

## Gabung fragmen track / stitching (11 Nov 2026)

Tujuan: mengurangi kotak putih di video berlabel — track yang tidak
punya jendela 16 frame penuh (ID switch / track pendek) tidak dapat
klip dan tidak terklasifikasi. `merge_track_fragments.py` menyambung
run-run (rangkaian frame berurutan satu ID) yang kemungkinan ayam
yang sama, dengan ambang KONSERVATIF: gap ≤ 4 frame, jarak pusat bbox
≤ 0,6× ukuran bbox, rasio luas bbox 0,6–1,67, plus ratio test 0,75
(dua kandidat yang hampir sama dekat → ditolak karena ambigu).
Hasil: `results/infer_fcam16_merged/merged_tracks_fCam16.json` +
`merge_report.json`; inference ulang + render ulang video berlabel
(`results/fCam16_annotated_merged.mp4`).

- **Diterima: 4 merge dari 46 kandidat** — ID 89→76, 28→91, 64→127,
  39→122 (2 ditolak ambigu; 40 dilewati, 39 di antaranya karena frame
  tumpang tindih dengan komponen target). ID: 85 → 81; track
  terklasifikasi 43 → 42 (dua ID terklasifikasi bergabung jadi satu:
  28+91); klip tetap **120** (distribusi sama: RESTING 66, ACTIVE 46,
  INGESTIVE 8); **box-frame berlabel 1.908 → 1.913, kotak putih
  364 → 359** — hanya 5 box-frame yang dapat label (89, 127, 122
  masing-masing 1–3 frame lewat label fallback track gabungannya).
  Tidak ada track yang label mayoritasnya berubah sesudah merge
  (28+91 sama-sama RESTING), dan cek visual frame 0/20/40/60 tidak
  menemukan label yang jelas salah akibat sambungan.
- **Catatan jujur — hasilnya kecil, dan itu temuan yang penting.**
  Komposisi 359 box-frame putih yang tersisa: 252 (70%) fragmen tepi
  video (ayam masuk/keluar frame di awal/akhir klip 7,2 dtk — tidak
  ada pasangan di luar video, stitching TIDAK BISA memperbaikinya),
  19 blip 1–3 frame, 88 fragmen menengah (sebagian besar terhalang
  tumpang tindih waktu). Jadi di klip pendek yang padat ini, langit-langit
  method 2 memang rendah.
- **Yang dicoba dulu dan GAGAL (dicatat supaya tidak diulang):**
  varian run-level — run-run satu ID boleh terpecah ke komponen
  berbeda — justru menambah kotak putih (364 → 512 box-frame putih;
  dengan fase perbaikan naive malah 610), karena run ekor/kepala dari
  ID yang tadinya terklasifikasi terpisah menjadi track putih dan
  kehilangan label fallback-nya. Karena itu script final memakai
  komponen ID-utuh: komponen tidak pernah terpecah, hanya bergabung,
  sehingga box-frame berlabel dijamin tidak pernah turun (script
  gagal keras bila jaminan ini dilanggar).

## Label propagation / label warisan (15 Nov 2026)

Tujuan: menandai 5 box-frame yang dapat label lewat stitching sebagai
label warisan (bukan prediksi model), supaya di video demo kelihatan
bedanya. `propagate_labels.py` membaca `merge_report.json` + prediksi
hasil merge + prediksi asli; hanya fragmen yang tadinya putih yang
mewarisi label mayoritas track gabungannya. Hasil:
`results/infer_fcam16_merged/track_predictions_inherited.json` (file
delta — prediksi model tidak diubah) + video
`results/fCam16_annotated_inherited.mp4` (label warisan: teks
"(warisan)", kotak magenta, tanpa angka confidence).

- **Yang diwarisi: 3 fragmen, 5 box-frame** — 89 (frame 43) ←
  RESTING dari 76; 127 (frame 65, 68, 69) ← RESTING dari 64; 122
  (frame 62) ← ACTIVE dari 39. Merge 28→91 dilewati: kedua fragmen
  sudah berlabel sebelum merge (frame 40 sudah berlabel via fallback
  track 28). Kotak putih 359 → 354.
- **Verifikasi:** script gagal keras bila ada frame warisan yang
  sudah berlabel sebelum merge, bila label fallback track gabungan
  tidak sama dengan label warisan, atau bila jumlah box-frame warisan
  tidak sama dengan selisih merge_report (+5). Cek visual frame
  43/62/65 + hitung piksel magenta: 5 frame warisan terkonfirmasi,
  frame kontrol 40 (merge yang dilewati) tidak bertanda.
- **Batasan — baca dulu sebelum memakai: label warisan HANYA untuk
  tampilan video demo.** Labelnya copy dari label mayoritas track
  gabungan — bisa basi (perilaku berubah selama di luar frame) dan
  bisa salah bila sambungan merge-nya keliru. Jangan dipakai
  mentah-mentah di Stage 3 (aturan durasi); bila dipakai, tandai dan
  bobot rendah. Efeknya kecil karena stitching-nya sendiri hanya
  menyambung 4 pasang.

## Pipeline penuh video demo Big Broiler 091 (15 Nov 2026)

Video demo 10 menit, overhead murni (dataset Big Broiler, UGent/DECIDE,
AGPL-3.0): `topview_test/bigbroiler_091_720p_10fps.mp4` — 6000 frame
@10 fps, 1280×720, 8 ayam broiler umur 32 hari + ground truth bbox dan
perilaku per frame (CSV). Alur: tracking Stage 1 (RT-DETRv2-S +
ByteTrack, threshold 0,6) → perbaikan seam + stitching → inference
Stage 2 batched → render. Script baru: `backend/stage1_detection/track_video.py`
(tracking streaming + checkpoint + resume), `infer_video_batched.py`
(inference streaming + batch, resume dari JSONL),
`fix_seam.py` (perbaiki tracks tercemar sambungan resume),
`analyze_bigbroiler091.py` (ringkasan + banding GT), dan opsi
`--max-frames` di `render_annotated_video.py`.

**Koreksi atas klaim lama:** "9/9 ayam terdeteksi" di 3 frame contoh
SALAH — yang benar **8 ayam + 1 kotak listrik (ID 2) yang terdeteksi
sebagai ayam dengan skor 0,97**. GT memang hanya 8 track.

### Hasil tracking (6000 frame, 58.817 box)

- **32 ID mentah → 41 ID setelah perbaikan seam + merge.** Run
  tracking pertama mati (SIGTERM, diduga restart VM) di frame 3600;
  resume memakai tracker BARU sehingga ID sesudah frame 3600 tertukar
  (ayam lama ID 1→5, 6→1, 5→8, dst). `fix_seam.py` me-renumber frame
  ≥3600 (+1000), lalu stitching menyambung 9 pasang seam (9 edge,
  9 diterima, jarak 0,0–0,9 px) → 50 → 41 ID. Tanpa perbaikan ini
  statistik per ID korup (satu ID berisi dua ayam berbeda).
- **Median span 166 frame** — angka ini ketarik ke bawah oleh banyak
  fragmen pendek; 8 ayam utama span 4334–6000 frame (ID 1, 3, 4, 5, 9
  nyaris penuh 6000 frame). Tiap track GT diwakili 1–4 ID kita
  (3557→4 ID, 3558→2, 3559→3, 3555→2) — **ID switch masih ada**,
  terutama di paruh kedua video.
- **Kotak putih: 108 box-frame (0,18%)** dari 58.817 — jauh lebih baik
  dari fCam16 (359 dari 2.272, 15,8%). 12 track tanpa jendela klip
  penuh (total 108 box).

### Hasil inference (7.164 klip, 29 track terklasifikasi)

| Label | Klip | % |
|---|---|---|
| RESTING | 6.571 | 91,7% |
| INGESTIVE | 532 | 7,4% |
| ACTIVE | 61 | 0,9% |

- **FP statis mengotori 14,6% klip.** 12 ID statis (rentang gerak
  pusat ≤3 px, track ≥30 frame): 7.502 box (12,8%) + 886 klip,
  ditambah ID 15 (drinker, 1.059 frame, 131 klip — lolos ambang
  dengan selisih 0,8 px) dan ID 19 (drinker, 239 frame, 27 klip).
  **ID 2 = kotak listrik abu-abu di dinding: 6000 frame penuh,
  749 klip, semuanya RESTING dengan confidence ~1,00.** Tanpa FP:
  RESTING 5.532 (90,4%), INGESTIVE 527 (8,6%), ACTIVE 61 (1,0%) —
  bias RESTING tetap.

### Filter track statis — SELESAI 16 Nov 2026

- Script baru `backend/stage1_detection/filter_static_tracks.py`
  (ambang default 10 px, `--min-displacement` untuk mengubah,
  `--no-filter-static` di `track_video.py` untuk mematikan). Aturan:
  buang track yang pusat bbox-nya berpindah < 10 px sepanjang
  hidupnya. Terpasang di akhir `track_video.py` (menulis
  `<out>.filtered.json`, dipakai inference berikutnya) dan bisa
  dijalankan manual pada hasil lama.
- Hasil di Big Broiler 091: 41 → 17 track (buang 24, termasuk
  ke-14 ID FP statis + ID 1031, 1 klip); box 58.817 → 49.832;
  klip 7.164 → **6.119**. Distribusi sesudah filter: RESTING
  5.531 (90,4%), INGESTIVE 527 (8,6%), ACTIVE 61 (1,0%) — bias
  RESTING tetap, tapi durasi RESTING tidak lagi tercemar benda
  mati. (Selisih 1 klip RESTING vs hitungan manual 5.532: ID 1031,
  102 frame, 1 klip, perpindahan 8,9 px — ikut terbuang filter.)
  Tidak satu pun ayam asli terbuang (ayam paling diam, ID 1032,
  berpindah 31,2 px — 8x ambang FP terbesar 3,8 px).
- File: `results/infer_bigbroiler091/merged_tracks_bigbroiler091.filtered.json`
  (+ `.report.json`) dan `track_predictions_filtered.json`. File
  mentah tidak diubah. Video berlabel dari file terfilter:
  `bigbroiler091_annotated_filtered.mp4` (6000 frame, 49.832 kotak
  — kotak listrik dan bohlam drinker sudah tidak berlabel).
- **Vote terbelah (butuh ambang confidence di Stage 3):** ID 8
  (100 RESTING / 45 INGESTIVE), ID 1011 (141 / 98 / 7 ACTIVE),
  ID 1014 mayoritas INGESTIVE (30 / 15 / 1) — satu-satunya track
  ayam yang mayoritas INGESTIVE. Tidak ada track mayoritas ACTIVE.
- **Confidence rata-rata per label:** RESTING 0,9955, INGESTIVE
  0,9602, ACTIVE 0,9120 — model sangat yakin bahkan saat salah
  (lihat bias di bawah).

### Perbandingan ke ground truth (pemetaan kasar, BUKAN akurasi)

- **8/8 track GT tercocok** (3552–3559 semua ditemukan pipeline).
- Distribusi aksi GT (94.973 baris frame): Eat+Drink 31,8%,
  Locomotion+Explor_Forag+Stretching+Feather_Pecking 8,2%,
  Inactive+Resting+Preening+Panting+Lying 57,6%, Out_Of_Sight 2,4%.
  Model: RESTING 91,7% vs GT ~57% — **bias top-view kuat**:
  dari atas, makan (kepala menunduk) terlihat seperti gumpalan
  statis → RESTING; crop mengikuti ayam sehingga gerak jalan hanya
  terlihat sebagai gerak latar + kaki yang dari atas nyaris tak
  terlihat. Angka 91,99% (test set) itu dari sudut miring — di
  top-view wajar turun, bukan berarti model rusak.
- **ID 1011: ayam asli yang tidak pernah cocok GT** (1.982 frame,
  246 klip, skor rata-rata 0,96; terverifikasi visual di frame 4500
  — sendirian di drinker — dan 5000 — ayam putih distinct di samping
  ID 9). Jarak median ke track GT terdekat 231–349 px, 0 vote dalam
  1.982 frame. Penjelasan paling mungkin: **GT tidak lengkap di
  video akhir** — di frame GT 11250 hanya 4 dari 8 track yang
  dianotasi, dan track 3555 berhenti total di frame GT 11303.
  Jadi "ID tanpa GT" ≠ FP; ada 26 ID tanpa GT, sebagian ayam asli
  yang GT-nya bolong, sebagian FP statis, sebagian fragmen.
- **Batasan — baca dulu:** label perilaku di top-view TANPA ground
  truth perilaku langsung per klip; distribusi label adalah hasil
  model, bukan akurasi. FP statis (kotak listrik, bohlam drinker)
  WAJIB difilter sebelum Stage 3, kalau tidak durasi RESTING
  tercemar ~15%. ID switch masih ada meski kandang sepi dan nyaris
  tanpa oklusi — stitching posisi saja tidak cukup; ReID tetap opsi
  jangka panjang.

### Akurasi label perilaku vs GT — SELESAI 16 Nov 2026

Perbandingan klip-per-klip (bukan cuma distribusi): label prediksi
(setelah filter FP statis: 6.119 klip, 17 track) vs label GT
mayoritas per klip, dicocokkan lewat kotak (IoU ≥ 0,3).
Script: `compare_gt.py`; angka lengkap: `gt_comparison.json`;
ringkasan: `gt_comparison.md`.

- **Penyelarasan frame TERVERIFIKASI**: frame_gt =
  round(frame_10fps × 2,5). Bukti: jarak pusat kotak prediksi vs
  kotak GT pasangannya pada frame ayam GT sedang bergerak — median
  8,9 px di offset 0 vs 34–36 px di ±25 frame (±1 dtk) dan 81–129 px
  di ±250 frame (±10 dtk); median IoU 0,850 di offset 0.
- **Coverage 79,9% (4.889 / 6.119 klip tercocokkan GT)**; 1.230
  klip tak tercocokkan — 246 di antaranya seluruh klip ID 1011
  (ayam asli yang GT-nya bolong), 54% sisanya di 200 detik terakhir
  (GT makin bolong).
- **Akurasi (klip tercocokkan): 61,83% (3.023 / 4.889)** — angka
  91,99% (test set, sudut miring) TIDAK pindah ke top-view.

  | GT \ prediksi | INGESTIVE | ACTIVE | RESTING |
  |---|---|---|---|
  | INGESTIVE (1.549) | 203 | 5 | **1.341** |
  | ACTIVE (471) | 79 | 37 | 355 |
  | RESTING (2.869) | 85 | 1 | 2.783 |

  Recall: INGESTIVE **13,11%**, ACTIVE 7,86%, RESTING 97,00%.
  Precision: INGESTIVE 55,31%, ACTIVE 86,05%, RESTING 62,13%.
- **Dugaan TERBUKTI: 86,57% klip INGESTIVE GT diprediksi RESTING**
  (1.341 / 1.549); sebaliknya cuma 2,96%. Dari atas, makan
  kelihatan seperti gumpalan diam. ACTIVE pun 75,4%-nya diprediksi
  RESTING.
- **Confidence tidak menyelamatkan**: akurasi klip confidence
  ≥ 0,99 cuma 63,1% (4.590 klip) — kesalahannya yakin, bukan ragu.
- **Uji sensitivitas — kesimpulan tidak berubah**: ambang IoU 0,5 →
  akurasi 61,70%, INGESTIVE → RESTING 87,89%; Explor_Forag →
  INGESTIVE (bukan ACTIVE) → akurasi 62,14%, INGESTIVE → RESTING
  87,13%.
- **Implikasi**: Stage 3 tidak bisa memakai label ini apa adanya
  (durasi RESTING akan sangat kehitung lebih). Opsi yang belum
  diputuskan: (a) fine-tune VideoMAE pada klip top-view (GT Big
  Broiler menyediakan aksi per frame), (b) kamera miring untuk
  perilaku + top-view untuk tracking, (c) terima biasnya dengan
  faktor koreksi. Untuk demo 2 Des: sampaikan 91,99% sebagai
  akurasi test-set (sudut miring) dan 61,8% sebagai akurasi lapangan
  top-view — deteksi + tracking kuat, klasifikasi perilaku top-view
  lemah.

### Perbandingan vs fCam16 (72 frame, sudut miring, 33 ayam/frame)

| | fCam16 | Big Broiler 091 |
|---|---|---|
| Frame | 72 | 6.000 |
| Box total | 2.272 | 58.817 |
| ID unik (setelah merge) | 81 | 41 |
| Klip | 120 | 7.164 |
| Track terklasifikasi | 42 | 29 |
| Kotak putih | 359 (15,8%) | 108 (0,18%) |
| Distribusi klip | RESTING 66, ACTIVE 46, INGESTIVE 8 | RESTING 6.571, INGESTIVE 532, ACTIVE 61 |
| Ground truth | tidak ada | 8/8 track tercocok |

### File hasil

`results/infer_bigbroiler091/`: `tracks_bigbroiler091.json` (mentah,
32 ID), `tracks_bigbroiler091_fixed.json` (seam diperbaiki),
`merged_tracks_bigbroiler091.json` (41 ID) + `merge_report.json`,
`clip_results.jsonl` + `track_predictions.json` (7.164 klip),
`merged_tracks_bigbroiler091.filtered.json` +
`track_predictions_filtered.json` (setelah filter FP statis: 17 track,
6.119 klip), `gt_comparison.json` + `gt_comparison.md` (banding vs GT:
akurasi 61,83%), `summary.json` (ringkasan + banding GT), `bigbroiler091_annotated.mp4`
(6000 frame, 600 dtk, 199 MB, 58.817 kotak digambar; gaya sama
seperti fCam16 — kotak + `ID <n> <LABEL> <conf>`, track tanpa klip =
kotak putih hanya ID), `bigbroiler091_annotated_filtered.mp4`
(197 MB, 49.832 kotak, tanpa FP statis), `bigbroiler091_60s_annotated.mp4` (sampel 60
dtk pertama). Log: `infer_full.log`, `infer_full_resume1.log`,
`infer_full_resume2.log`, `track_full_resume.log`,
`render_full.log`.

### Waktu jalan (CPU VM, 1,5 CPU shared)

Tracking ~2 jam total (run 1 mati SIGTERM di frame 3600, resume
~52 mnt); inference ~5,8 jam total dalam 3 run (run 1 mati SIGTERM
di ~klip 2.784, resume 1 mati saat restart runtime di ~klip 5.320,
resume 2 selesai 7.164 klip dalam 101 mnt — resume dari JSONL
bekerja, tidak ada klip yang hilang/duplikat); render ~5 mnt.
Semua interupsi selamat karena hasil antara disimpan berkala
(checkpoint JSON tiap N frame, JSONL per klip).

## Pipeline penuh video sudut miring Big Broiler 080 (16 Nov 2026)

Video 10 menit, setup 2 angled view ~45° (dataset Big Broiler yang sama,
UGent/DECIDE, AGPL-3.0): `topview_test/oblique_bigbroiler_080_720p_10fps.mp4`
— 6008 frame @10 fps, 1280×720, broiler umur 40 hari, 24 track GT.
Dipilih user setelah 091 (top-view) akurasinya cuma 61,83%: hipotesisnya
sudut miring — sama seperti data latih — mengembalikan akurasi mendekati
91,99%. Alur SAMA seperti 091: tracking Stage 1 (threshold 0,6, filter
statis aktif) → inference Stage 2 batched (resume dari JSONL) → filter
prediksi → render → banding GT (`compare_gt.py` + flag `--force`, lihat
di bawah).

### Hasil tracking (6008 frame, 61.849 box)

- **88 ID mentah → 52 setelah filter statis → 51 setelah buang silo.**
  Jauh lebih banyak dari 091 (41 ID): sudut miring bikin ayam keluar-masuk
  frame + oklusi dinding dekat, jadi banyak fragmen. Median span track
  mentah cuma 26,5 frame (terfilter 86,0; 091: 166).
- Run 1 mati SIGTERM di frame ~1900 (checkpoint 1800); resume MEMULIHKAN
  state tracker (`frame_id=1800`) sehingga ID berlanjut tanpa putus —
  **tidak perlu `fix_seam.py`/stitching** (berbeda dengan 091).
- **24/24 track GT tercocok.** Kebanyakan track GT diwakili 1–2 ID kita;
  pengecualian: GT 3313 diwakili 12 ID (ayam yang paling sering switch),
  GT 3308 diwakili 5 ID. ID switch masih ada meski sudut miring.
- **Kotak putih: 257 box-frame (0,45%)** setelah silo dibuang — dari 56.543
  box. (Angka mentah di `summary.json`, 4.255, masih berisi 3.998 kotak silo.)

### Filter track statis — dan satu FP yang LOLOS

- Filter otomatis (ambang 10 px): 88 → 52 ID (buang 36, box 61.849 →
  60.541). Celah bersih: terbuang maks 9,9 px, yang dipertahankan min
  11,9 px — tidak ada ayam asli terbuang.
- **TAPI: ID 40 = silo drinker putih (3.998 frame, skor rata-rata 0,92)
  LOLOS filter.** Terverifikasi visual di frame 3000 dan 5700: kotaknya
  menempel persis pada silo. Penyebab: silo tidak bergerak, tapi kotaknya
  bergoyang saat ayam lewat di depan/belakangnya (oklusi) — tinggi box
  167–265 px, lompatan pusat antar-frame sampai 34,7 px, rentang pusat
  seumur hidup 28,8 × 51,3 px — jauh di atas ambang 10 px. Beda dengan
  kotak listrik 091 (1,0 × 0,4 px, tidak pernah teroklusi).
- Penanganan (prosedur sama seperti 091): inference jalan di tracks
  terfilter (52 ID), lalu **496 klip ID 40 dibuang pada tahap filter
  prediksi** (458 RESTING + 38 INGESTIVE — silo bahkan sempat diprediksi
  INGESTIVE 38 kali). Bukti lengkap: `results/infer_bigbroiler080/fp_removal_080.json`.
  Render memakai `tracks_bigbroiler080.filtered_nofp.json` (51 ID) supaya
  silo tidak muncul sama sekali.
- **Pelajaran buat filter:** ambang perpindahan pusat saja tidak cukup
  untuk benda yang sering teroklusi. Kandidat perbaikan: cek juga variasi
  ukuran box, atau deteksi objek statis dari frame referensi — belum
  dikerjakan.

### Hasil inference (7.432 klip, 27 track terklasifikasi)

| Label | Klip sebelum filter | % | Klip sesudah filter | % |
|---|---|---|---|---|
| RESTING | 6.714 | 90,4% | 6.256 | 90,2% |
| INGESTIVE | 671 | 9,0% | 633 | 9,1% |
| ACTIVE | 47 | 0,6% | 47 | 0,7% |
| **Total** | **7.432** | | **6.936** | |

- 26 track terklasifikasi sesudah filter (1 track tanpa klip penuh:
  15 track terfilter <16 frame). Tidak ada track mayoritas ACTIVE;
  track paling INGESTIVE: ID 15 (197 INGESTIVE / 705 klip).
- Confidence rata-rata: RESTING 0,9929, INGESTIVE 0,9338, ACTIVE 0,9149.
- Verifikasi visual render (frame 3000 & 792): silo tanpa label; kotak +
  label menempel pada ayam yang benar; label INGESTIVE (hijau) pada ayam
  yang kepalanya menunduk makan/minum — masuk akal.

### Akurasi label perilaku vs GT — SELESAI 16 Nov 2026

Perbandingan klip-per-klip seperti 091 (IoU ≥ 0,3, label GT mayoritas).
Script: `compare_gt.py` (+ flag `--force`, lihat catatan di bawah);
angka lengkap: `gt_comparison.json`; ringkasan: `gt_comparison.md`.

- **Akurasi (klip tercocokkan): 80,91% (1.144 / 1.414)** — naik dari 61,83%
  di top-view 091 (+19,1 poin), tapi masih jauh dari 91,99% test-set.

  | GT \ prediksi | INGESTIVE | ACTIVE | RESTING |
  |---|---|---|---|
  | INGESTIVE (212) | 61 | 0 | **151** |
  | ACTIVE (50) | 3 | 5 | 42 |
  | RESTING (1.152) | 59 | 15 | 1.078 |

  Recall: INGESTIVE **28,77%**, ACTIVE 10,00%, RESTING 93,58%.
  Precision: INGESTIVE 49,59%, ACTIVE 25,00%, RESTING 84,82%.
- **INGESTIVE → RESTING masih dominan: 71,23%** (151/212), turun dari
  86,57% di 091 — sudut miring membantu, tapi tidak menyelesaikan.
  Sebaliknya cuma 5,12%.
- **Coverage HANYA 20,4% (1.414 / 6.936 klip)** — GT setup 2 jarang
  (~2,25 track/frame dianotasi; 091: ~8 track/frame, coverage 79,9%).
  5.522 klip tak tercocokkan. **Angka 80,91% kemungkinan OPTIMIS:**
  klip yang tercocokkan adalah yang kebetulan punya padanan GT (ayam yang
  dianotasi cenderung yang mudah diamati; distribusi GT pada klip
  tercocokkan RESTING 81,5% vs GT keseluruhan ~74%).
- Confidence: klip confidence ≥0,99 akurasinya 84,09% (1.068/1.270) —
  lebih baik dari 091 (63,1%), tapi tetap 16% salah walau sangat yakin.
- **Uji sensitivitas — kesimpulan tidak berubah**: ambang IoU 0,5 →
  akurasi 81,17%, ING→REST 71,23%; Explor_Forag → INGESTIVE → akurasi
  80,91%, ING→REST 72,77%.
- **Distribusi aksi GT 080** (26.922 baris): INGESTIVE (Eat+Drink) 13,8%,
  ACTIVE 3,0%, RESTING (Inactive+Panting+Preening+aksi kosong) 74,0%,
  Out_Of_Sight 1,3%, tidak terpetakan 8,0% (Sleeping 3,8%, Short_Leg_Move
  2,7%, dst. — tidak menyumbang suara, pemetaan sama seperti 091).
- **Penyelarasan frame DITERIMA DENGAN CATATAN (`--force`):** kriteria
  ketat script gagal HANYA di m0 ≤ 15 px (m0 = 15,6 px, selisih 0,6 px,
  n≈108 pasangan bergerak); dua syarat rasio lolos (m25 = 50,5 px,
  m250 = 159,9 px); lembah datar 0..+5 frame, naik curam di −5
  (26,6 px) dan ±25 frame. Ambang 15 px dikalibrasi untuk GT rapat 091
  (n≈1.320); pada GT jarang sampel kecil membuat median berisik.
  Bukti IoU pendukung tidak informatif (median 0,000 di semua offset —
  artefak coverage GT jarang, bukan bukti pemetaan salah). Flag `--force`
  ditambahkan ke `compare_gt.py` (perilaku default tetap ketat; hasil
  ditandai `frame_alignment_forced: true` di JSON).
- **Implikasi:** Stage 3 tetap tidak bisa memakai label ini apa adanya
  (RESTING ~90% prediksi vs ~74% GT; INGESTIVE kehitung kurang). Opsi
  yang belum diputuskan: (a) fine-tune VideoMAE pada klip Big Broiler
  (080 + 075 + 091 digabung; GT 080 jarang tapi ada), (b) terima bias
  dengan faktor koreksi, (c) kamera miring untuk perilaku + top-view
  untuk tracking (keduanya terbukti di video ini: tracking 24/24 track
  GT tercocok, klasifikasi perilaku yang lemah). Untuk demo 2 Des:
  sampaikan 91,99% sebagai akurasi test-set, 80,9% sebagai akurasi
  lapangan sudut miring (coverage 20%, kemungkinan optimis), dan 61,8%
  sebagai akurasi lapangan top-view.

### Perbandingan 091 (top-view) vs 080 (sudut miring)

| | 091 top-view | 080 sudut miring |
|---|---|---|
| Frame / box | 6.000 / 58.817 | 6.008 / 61.849 |
| ID mentah → terfilter | 32 → 17 | 88 → 52 (→ 51 tanpa silo) |
| Median span track | 166 | 86 (mentah 26,5) |
| Klip (sesudah filter) | 6.119 | 6.936 |
| Track terklasifikasi | 29 | 26 |
| Kotak putih | 108 (0,18%) | 257 (0,45%, tanpa silo) |
| Distribusi klip | RESTING 90,4%, ING 8,6%, ACT 1,0% | RESTING 90,2%, ING 9,1%, ACT 0,7% |
| GT track tercocok | 8/8 | 24/24 |
| Coverage GT | 79,9% | 20,4% |
| **Akurasi vs GT** | **61,83%** | **80,91%** |
| Recall INGESTIVE | 13,11% | 28,77% |
| INGESTIVE → RESTING | 86,57% | 71,23% |
| FP statis | kotak listrik (filter 10 px cukup) | silo drinker (**lolos** filter 10 px) |

### File hasil

`results/infer_bigbroiler080/`: `tracks_bigbroiler080.json` (mentah,
88 ID), `tracks_bigbroiler080.filtered.json` (52 ID, filter otomatis) +
`tracks_bigbroiler080.filtered_nofp.json` (51 ID, tanpa silo),
`clip_results.jsonl` + `track_predictions.json` (7.432 klip) +
`track_predictions_filtered.json` (6.936 klip, tanpa silo),
`fp_removal_080.json` (bukti FP silo), `summary.json` (ringkasan +
banding GT), `gt_comparison.json` + `gt_comparison.md` (banding vs GT:
akurasi 80,91%, coverage 20,4%) + `gt_comparison_iou05.json` +
`gt_comparison_explor_ing.json` (uji sensitivitas),
`bigbroiler080_annotated_filtered.mp4` (6008 frame, 600 dtk, ~280 MB,
56.543 kotak digambar; gaya sama seperti 091 — kotak +
`ID <n> <LABEL> <conf>`, track tanpa klip = kotak putih hanya ID).
Log: `track_full.log`, `track_resume.log`, `infer_full.log`,
`infer_full_resume1/2/3.log`, `render_full.log`.

### Waktu jalan (CPU VM, 1,5 CPU shared)

Tracking ~2 jam total (run 1 mati SIGTERM di frame ~1900, resume 84 mnt
— resume dari checkpoint + state tracker, tanpa klip/frame hilang);
inference ~5,4 jam total dalam 4 run (run 1 mati SIGTERM di klip 784,
resume 1 mati di klip 3.240, resume 2 mati di klip 5.776, resume 3
selesai 7.432 klip — resume dari JSONL bekerja, 7.432 pasangan
(track_id, start_frame) unik, tidak ada yang hilang/duplikat); render
~5 mnt. Pola interupsi sama seperti 091: restart runtime berkala
(SIGTERM) — semua selamat karena hasil antara disimpan berkala.

## Fine-tune Big Broiler — persiapan (16 Nov 2026)

**Status: training SELESAI 16 Nov 22:57 WIB + evaluasi kedua test set
SELESAI 17 Nov 2026 — fine-tune BERHASIL (lihat hasil di bawah).**
Alasan: akurasi lapangan jauh di bawah test-set — INGESTIVE ketukar
RESTING 86,57% (top-view 091) / 71,23% (miring 080). Model dilatih dari
klip ChickenAct dan belum pernah melihat video kandang beneran;
fine-tune lanjutan ini mengajarinya langsung dari video Big Broiler.

### Dataset: BigBroiler-Finetune.zip (Drive: PBL-Ayam/dataset/, 2.590.811.233 byte, md5 37d836fefbaee1daf666706ccad88cc4) — v2, 16 Nov 2026

> **Di Drive, zip v2 disimpan sebagai 13 part @200 MB**
> (`BigBroiler-Finetune.zip.part-aa` s.d. `.part-am`) — upload zip utuh
> gagal 3x di CLI (HTTP 500, http2 stream error, multipart; gagal juga
> di file tes 1 GB, berhasil di file tes 200 MB). Notebook mengunduh
> ke-13 part, memverifikasi tiap part (ukuran + md5), menggabungkannya
> dengan `cat`, lalu memverifikasi zip utuh (ukuran + md5 di atas).
> Per-part md5 ada di cell konfigurasi notebook + `parts.md5` (lokal).
> Zip v1 lama di Drive di-rename jadi `BigBroiler-Finetune-v1.zip`
> (554.110.228 byte, md5 28a3308ee3b307ac7c0552a3f095a4e3) sebagai backup.

- Klip 16 frame @224x224, crop persegi mengikuti kotak GT tiap frame
  (margin 1,3 — sama seperti `clip_builder.py`), label = suara
  mayoritas aksi GT. Script pembangun:
  `data_prep/build_bigbroiler_clips.py`.
- Klip usable (stride 16, STRICT — ke-16 frame harus punya aksi GT
  valid, seri dibuang):

  | Video | INGESTIVE | ACTIVE | RESTING | Total |
  |---|---|---|---|---|
  | 091 (top-view) | 734 | 138 | 1.271 | 2.143 |
  | 080 (miring) | 79 | 33 | 451 | 563 |
  | 075 (miring) | 4 | 86 | 118 | 208 |
  | 067 (miring, 39 hr) | 784 | 352 | 1.213 | 2.349 |
  | 057 (miring, 29 hr) | 611 | 372 | 2.511 | 3.494 |
  | 060 (miring, 29 hr) | 579 | 86 | 1.725 | 2.390 |
  | 064 (miring, 34 hr) | 501 | 141 | 1.957 | 2.599 |
- Pemetaan aksi GT → 3 kelas = `compare_gt.ACTION_MAP` persis
  (Explor_Forag → ACTIVE). Perluasan HANYA untuk 5 aksi yang tidak ada
  di CSV 091: Sleeping → RESTING, Flapping → ACTIVE, Head_Shaking →
  RESTING, Short_Leg_Move → ACTIVE, Swallowing → INGESTIVE (mengikuti
  logika `label_mapping.py`; angka 091 tidak berubah). Tanpa suara:
  Sneezing, Defecating, Video_Problem, Other.
- **Perluasan v2 (16 Nov 2026) — INGESTIVE train 435 → 2.662 (6,1x):**
  user bertanya kenapa sampel kelas lemah tidak dibuat lebih banyak.
  Jawabannya: sampel datang dari kejadian beneran di video — 075
  praktis tidak ada adegan makan (4 klip), 080 sedikit (79). Jadi
  digali video setup-2 lain: semua 32 CSV GT diunduh dari zip dataset,
  dihitung klip usable-nya dengan logika jendela yang sama persis
  (`topview_test/bb_rank.py`), dipilih 4 teratas (067/057/060/064 —
  INGESTIVE 784/611/579/501, GT coverage 100%). Video dipilih
  berdasarkan CSV dulu, baru videonya diunduh (range-download, 4 file
  ~580–646 MB, CRC32 cocok), dibuat versi 720p 10fps, sudut miring
  diverifikasi dari frame contoh + deteksi Stage 1 cepat (067: 15/12/12
  box; 057: 12/11/13; 060: 14/14/10; 064: 10/11/11 — ada FP silo
  drinker di beberapa frame, tidak menghalangi karena klip latih
  dipotong dari kotak GT, bukan deteksi). 066 TIDAK dipilih (ING cuma
  56 — GT-nya rapat tapi makannya sedikit; Eat 9,3%-nya ternyata kalah
  dari 067 yang E+D-nya 24,6%).
- **075 TIDAK bisa jadi test set utama**: cuma 208 klip usable dan
  INGESTIVE-nya cuma 4 (di 075 praktis tidak ada adegan makan — GT-nya
  didominasi Explor_Forag + Inactive). Rencana awal (075 hold-out
  penuh) diganti fallback yang sudah direncanakan: split temporal.
- Split (seed 42): train/val dari 091 paruh-1 (start < 3000) + 080
  + 067 + 057 + 060 + 064 semua; val 10% stratified per kelas
  (1.253 klip: ING 296 / ACT 106 / REST 851); **test utama = 091
  paruh-2 (1.011 klip: ING 330 / ACT 63 / REST 618) — TIDAK BERUBAH
  dari v1 (clip-id + label terverifikasi identik)**; **test kedua
  = 075 seluruhnya (208 klip) — TIDAK BERUBAH dari v1** sebagai cek
  generalisasi video tak terlihat.
- Imbalance: train mentah ING 2.662 / ACT 953 / REST 7.659 →
  **RESTING train di-downsample ke 2.662** (jumlah INGESTIVE), train
  akhir 6.277 klip. Alasan: paling gampang + andal; `train.py` tidak
  dukung class-weighted loss tanpa ubah kode; ACTIVE dibiarkan (recall
  95% di test-set ChickenAct). Val/test TIDAK di-downsample (metrik
  jujur). 4.997 klip RESTING train yang dibuang tidak masuk manifest.
- Verifikasi: semua 139.984 file frame ada (0 hilang); 6 contact sheet
  dicek visual — crop nempel di ayam yang benar, label masuk akal
  (INGESTIVE 057: ayam menunduk makan di trough; ACTIVE 057: ayam
  berjalan; RESTING 057: ayam duduk diam).
- Dataset v1 (2.355 klip, train 967) digantikan v2 ini. Rencana awal
  (upload zip v2 sebagai revisi baru file yang sama di Drive) GAGAL:
  upload multipart CLI gagal 3x (HTTP 500, http2 stream error) di file
  2,6 GB — file tes 1 GB juga gagal, file tes 200 MB berhasil. Solusi:
  zip di-split jadi 13 part @200 MB (`split -b 200M`, md5 per part
  terverifikasi di Drive), notebook di-update ke skema part (unduh 13
  part → verifikasi tiap part → `cat` → verifikasi zip utuh) dan
  di-upload sebagai revisi baru (file id notebook sama). Zip v1 di
  Drive di-rename jadi `BigBroiler-Finetune-v1.zip` (backup, file id
  sama). Validasi skema part: cell 2+3 notebook dijalankan lokal
  dengan 13 part — gabungan = 2.590.811.233 byte, md5
  37d836fefbaee1daf666706ccad88cc4, cocok dengan zip v2. Manifest +
  dataset_info v1 tersimpan di `bigbroiler_finetune_v1_backup/`
  (lokal, bukan di repo).

### Notebook: notebooks/Colab-BigBroiler-Finetune.ipynb

- Salinan notebook Stage 2 yang disesuaikan: checkpoint awal = model
  Stage 2 dari Drive (`runs/stage2-videomae/best/` — BUKAN Kinetics
  lagi); LR 1e-5; epoch 4; run baru `runs/stage2-bigbroiler-finetune/`;
  dataset hanya dari Drive (tanpa Zenodo); cell training raise
  RuntimeError kalau exit code ≠ 0.
- **Pin `transformers>=4.46,<5`** (beda dari Stage 2): checkpoint
  Stage 2 disimpan dengan transformers 4.x (attention bias bernama
  `q_bias`/`v_bias`); 5.x menamai ulang jadi `query.bias`/`value.bias`
  sehingga 24 vektor bias tidak termuat dan di-init acak (terbukti di
  smoke test VM, transformers 5.22.2). Pin ini memastikan fine-tune
  mulai PERSIS dari model Stage 2.
- Validasi: semua cell kode ter-compile; tanpa path lokal VM / file ID
  Drive; smoke test `train.py` (CPU VM, 6 train + 3 val klip, 1 epoch)
  lolos end-to-end (load model → train → eval → `best/` +
  `train_metrics.json`).
- Perkiraan waktu training: ~393 step/epoch, 4 epoch ~1.570 step,
  kira-kira 30–45 menit di T4. Unduh 13 part + gabung + verifikasi di
  awal menambah kira-kira 5–10 menit (sekali per sesi Colab).

### Cara jalanin (user, di Colab)

1. Buka notebook `Colab-BigBroiler-Finetune.ipynb` dari Drive (akun
   pandumanis72).
2. Runtime → Change runtime type → T4 GPU. Run All.
3. Setelah selesai: evaluasi kedua test set dengan `eval_test_set.py`
   (lihat langkah terakhir di notebook), bandingkan dengan baseline
   sebelum fine-tune (091 paruh-2 ~61,8% top-view; 075 belum pernah
   diukur).

### Ekspektasi jujur

Belum ada yang tahu berapa naiknya akurasi — diukur lawan kedua test
set setelah latih. Yang diharapkan naik: recall INGESTIVE (13,11%
top-view / 28,77% miring sekarang). Kalau tidak naik, itu juga hasil —
dilaporkan apa adanya.

### Training fine-tune — SELESAI 16 Nov 2026, 22:57 WIB (user Run All di Colab)

- 4/4 epoch, ~1.572 step, T4. Val terbaik **92,90%** (epoch 4 =
  terakhir; INGESTIVE 97,64% / ACTIVE 85,85% / RESTING 92,13%;
  eval loss 0,24; train loss akhir 0,39) — praktis sama dengan val
  Stage 2 (92,94%), jadi val saja tidak membuktikan perbaikan; yang
  menentukan adalah test set di bawah.
- Model terbaik di Drive `runs/stage2-bigbroiler-finetune/best/` +
  di repo `models_finetune/` (Git LFS; `model.safetensors` 87.546.468
  byte, md5 `512258442e05c42ea11d180ea3fc92d8` + `config.json` 967 byte,
  md5 `b8b08a3cff5145cd5aaddbb626ccc3d5` + `training_args.bin`).
- Satu error saat Run All pertama (22:20 WIB): cell 6 gagal
  `AssertionError: Model awal harus 3 kelas` — `config.json` Stage 2
  tidak menyimpan `num_labels` (None) padahal `id2label`-nya 3 kelas;
  modelnya benar, cek notebook terlalu ketat. Diperbaiki (hitung dari
  `id2label`), commit `8241eb2`, notebook Drive diperbarui, user Run
  All ulang.
- Catatan muat model: model fine-tune (disimpan transformers 4.57.6)
  dimuat di VM dengan transformers 5.22.2 TANPA bobot hilang/tak
  terduga (`missing_keys`/`unexpected_keys` kosong, smoke test
  lolos) — beda dengan checkpoint Stage 2 di 5.x dulu, yang
  kehilangan 24 vektor bias.

### Evaluasi test set fine-tune — SELESAI 17 Nov 2026 (VM, CPU)

Perintah (sama seperti evaluasi Stage 2): `eval_test_set.py
--checkpoint models_finetune --frames-manifest ... --data-root ...`.
Run utuh 1.011 klip kena SIGTERM di ~35 menit (restart runtime, log
di `/tmp` ikut hilang) — diulang dalam 4 chunk ~253 klip
(~10 menit/chunk, log di workspace) lalu confusion matrix digabung;
test-075 jalan utuh (~8 menit). Hasil gabungan di
`results/bigbroiler_finetune_test_eval.json` (+ 4 file chunk) dan
`results/bigbroiler_finetune_test075_eval.json`.

**Test utama — 091 paruh-2 (1.011 klip, top-view): akurasi 90,50%
(915/1.011)**, dari baseline 61,83% (**+28,67 poin**). Confusion
(baris = GT, kolom = prediksi): INGESTIVE [310, 5, 15] n=330; ACTIVE
[2, 22, 39] n=63; RESTING [19, 16, 583] n=618.

| Kelas | Recall (sebelum → sesudah) | Precision | F1 |
|---|---|---|---|
| INGESTIVE | 13,11% → **93,94%** (+80,83) | 93,66% | 0,9380 |
| ACTIVE | 7,86% → **34,92%** (+27,06) | 51,16% | 0,4151 |
| RESTING | 97,00% → **94,34%** (−2,66) | 91,52% | 0,9291 |

- INGESTIVE → RESTING tinggal 15 klip (dari 1.341 di baseline) —
  masalah utama yang mau diperbaiki, TERATASI.
- ACTIVE tetap lemah: recall 34,92% (39 dari 63 klip ACTIVE diprediksi
  RESTING). ACTIVE train cuma 953 klip (paling sedikit) dan val-nya
  85,85% — val tidak menangkap kelemahan ini karena val dari video
  yang sama dengan train. Ini sisa masalah terbesar.
- Caveat perbandingan: baseline 61,83% diukur `compare_gt` pada
  seluruh video 091 (4.889 klip tercocokkan, klip dari tracking),
  sedangkan angka fine-tune diukur pada 1.011 klip test (paruh-2,
  klip dari kotak GT, aturan STRICT). Metriknya sama (akurasi klip
  vs label GT), videonya sama, tapi himpunan klipnya tidak identik —
  selisih +28,67 poin jauh lebih besar dari bias yang mungkin
  ditimbulkan perbedaan ini.

**Test kedua — 075 seluruhnya (208 klip, sudut miring, video tak
terlihat): akurasi 67,31%** (140/208). Confusion: INGESTIVE [4, 0, 0]
n=4; ACTIVE [15, 48, 23] n=86; RESTING [20, 10, 88] n=118. Per kelas:
INGESTIVE recall 100% (4/4) tapi precision cuma 10,26% — model
memprediksi INGESTIVE 39 kali, 35 di antaranya salah (15 ACTIVE +
20 RESTING). ACTIVE recall 55,81% / precision 82,76%; RESTING recall
74,58% / precision 79,28%. Tidak ada baseline 075 sebelum fine-tune
(belum pernah diukur), jadi angka ini tidak bisa dibandingkan —
yang bisa dibaca: di video baru, model cenderung TERLALU SERING
menyebut INGESTIVE (kebalikan dari sebelum fine-tune, yang hampir
tidak pernah menyebutnya).

**Kesimpulan: fine-tune BERHASIL untuk target utamanya** — recall
INGESTIVE top-view naik 13,11% → 93,94% dan akurasi lapangan top-view
naik 61,83% → 90,50%, mendekati akurasi test-set Stage 2 (91,99%).
Dua catatan jujur: (1) ACTIVE masih lemah (recall 34,92% di test
091, 55,81% di test-075) — butuh data ACTIVE lebih banyak / penanganan
khusus, bukan sekadar tambah epoch; (2) di video baru (075) model
over-prediksi INGESTIVE — perlu kalibrasi/ambang sebelum dipakai
langsung di Stage 3. Angka demo yang jujur sekarang: test-set
91,99%, lapangan top-view 90,50% (fine-tune), lapangan miring
80,91% (model lama — belum diukur ulang dengan model fine-tune).

## Yang belum dikerjakan

- **Fine-tuning VideoMAE** — SELESAI 11 Nov 00:15 WIB (10 epoch,
  val 92,94%); **evaluasi test set** — SELESAI 11 Nov (91,99%,
  lihat di atas); **model hasil fine-tune** — TERSIMPAN di `models/`
  (Git LFS, `model.safetensors` 87 MB + `config.json`).
- **Fine-tune Big Broiler (lanjutan)** — TRAINING SELESAI 16 Nov
  22:57 WIB (val 92,90%) + **evaluasi test set SELESAI 17 Nov**
  (dataset v2: 8.749 klip, train 6.277 — INGESTIVE train 2.662, naik
  6,1x dari v1 — + notebook v2 skema 13 part, lihat di atas):
  **test 091 paruh-2 90,50%** (recall INGESTIVE 93,94%, dari 13,11%),
  **test-075 67,31%**; ACTIVE masih lemah (recall 34,92% di test 091).
  Model di `models_finetune/` (Git LFS).
- **Uji `inference.py` end-to-end di video sample** — SELESAI 11 Nov
  (lihat di atas: 120 klip, 43 track, video berlabel
  `results/fCam16_annotated.mp4`); **stitching fragmen track** —
  SELESAI 11 Nov (lihat di atas: hanya +5 box-frame berlabel; varian
  run-level dicoba dan dibuang karena menambah kotak putih);
  **label propagation (label warisan)** — SELESAI 15 Nov (lihat di
  atas: 3 fragmen, 5 box-frame, display-only); **pipeline penuh
  video demo 10 menit (Big Broiler 091)** — SELESAI 15 Nov (lihat di
  atas: 6.000 frame, 7.164 klip, video berlabel
  `results/infer_bigbroiler091/bigbroiler091_annotated.mp4`).
- **Tracker ber-ReID (BoT-SORT)** — opsi jangka panjang untuk ID
  switch: stitching (method 2) langit-langitnya rendah di klip padat
  (lihat di atas), akar masalahnya tetap di tracker. Belum dikerjakan.
- **Kelas AGONISTIC** — tidak ada data latih di ChickenAct (lihat
  Taksonomi); deteksi agresi Stage 3 butuh data tambahan atau fitur
  non-VideoMAE.
- **Imbalance RESTING 52%** — pertimbangkan class weight / sampling
  seimbang bila akurasi per kelas timpang.
- **Ambang perilaku Stage 3** (durasi → aman/dicurigai/sakit) — masih
  generik, menunggu distribusi data + keputusan user.
