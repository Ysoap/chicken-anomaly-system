# Perbandingan prediksi vs ground truth — Big Broiler 091 (16 Nov 2026)

Bandingkan label perilaku pipeline (VideoMAE-Small fine-tune, set klip
**sesudah filter FP statis**: 6.119 klip, 17 track) dengan ground truth
dataset Big Broiler (94.973 baris, 8 track, aksi per frame). Script:
`backend/stage2_behavior/compare_gt.py`; angka lengkap:
`gt_comparison.json`.

## Hasil utama

- **Coverage: 4.889 / 6.119 klip tercocokkan GT (79,9%)**; 1.230 klip
  (20,1%) tak tercocokkan; 10 klip seri suara mayoritas.
- **Akurasi (klip tercocokkan): 61,83% (3.023 / 4.889).**

### Confusion matrix (baris = GT, kolom = prediksi)

| GT \ prediksi | INGESTIVE | ACTIVE | RESTING | n GT |
|---|---|---|---|---|
| INGESTIVE | 203 | 5 | **1.341** | 1.549 |
| ACTIVE | 79 | 37 | 355 | 471 |
| RESTING | 85 | 1 | 2.783 | 2.869 |

### Akurasi per kelas

| Kelas | Recall | Precision | F1 | n GT |
|---|---|---|---|---|
| INGESTIVE | 13,11% | 55,31% | 0,21 | 1.549 |
| ACTIVE | 7,86% | 86,05% | 0,14 | 471 |
| RESTING | 97,00% | 62,13% | 0,76 | 2.869 |

## Dugaan INGESTIVE → RESTING: TERBUKTI

**86,57% klip INGESTIVE menurut GT diprediksi RESTING** (1.341 dari
1.549). Sebaliknya cuma 2,96% klip RESTING GT yang diprediksi INGESTIVE
— ketukarnya satu arah. Dari atas, ayam lagi makan (kepala menunduk,
badan diam) terlihat seperti gumpalan diam → model menebak RESTING.

ACTIVE juga hampir tidak kedeteksi: recall 7,86% (75,4% klip ACTIVE GT
diprediksi RESTING). Model praktis cuma kenal RESTING di sudut top-view:
recall RESTING 97%, dan 91,6% dari semua prediksi adalah RESTING.

## Confidence tinggi ≠ benar

Akurasi per kelompok confidence prediksi (klip tercocokkan):

| Confidence | Akurasi | n klip |
|---|---|---|
| < 0,9 | 39,3% | 117 |
| 0,9–0,99 | 43,4% | 182 |
| ≥ 0,99 | 63,1% | 4.590 |

94% klip ber-confidence ≥ 0,99, dan kelompok itu pun cuma 63% benar.
Jadi **ambang confidence tidak bisa memperbaiki ini** — kesalahannya
yakin, bukan ragu-ragu.

## Cara banding (dan kenapa sah)

- **Pemetaan aksi GT → 3 kelas** (dihitung ulang dari CSV):
  INGESTIVE = Eat + Drink (31,8%); ACTIVE = Locomotion +
  Explor_Forag + Stretching + Feather_Pecking (8,2%); RESTING =
  Inactive + Resting + Preening + Panting + Lying (aksi kosong,
  57,6%); Out_Of_Sight (2,4%) diabaikan — kotaknya posisi basi.
- **Penyelarasan frame TERVERIFIKASI.** GT 25 fps (15.000 frame) vs
  pipeline 10 fps (6.000 frame), pemetaan nominal
  frame_gt = round(frame_10fps × 2,5). Bukti utama: jarak pusat kotak
  prediksi vs kotak GT pasangannya, hanya pada frame saat ayam GT
  sedang bergerak (aksi Locomotion atau pusat GT bergeser >15 px/frame)
  — ayam diam tidak bisa membedakan offset. Hasil: median 8,9 px di
  offset 0, vs 34–36 px di ±25 frame (±1 dtk) dan 81–129 px di ±250
  frame (±10 dtk); lembah datar ±3 frame (±120 mdtk) di bawah noise
  gerak. Bukti pendukung: median IoU 0,850 di offset 0.
- **Pencocokan lewat KOTAK (IoU ≥ 0,3), bukan ID track** (ID prediksi
  17, ID GT 8 — beda). Label GT klip = suara mayoritas dari 16 frame
  klip (frame Out_Of_Sight / tanpa anotasi / tanpa kotak prediksi
  dilewati).
- **Subset yang dibandingin representatif:** distribusi GT pada klip
  tercocokkan (INGESTIVE 31,7%, ACTIVE 9,6%, RESTING 58,7%) hampir
  sama dengan distribusi GT keseluruhan (31,8% / 8,2% / 57,6%).

## Uji sensitivitas (kesimpulan tidak bergantung pada pilihan ini)

- **Ambang IoU 0,5** (bukan 0,3): akurasi 61,70%, coverage 4.854,
  INGESTIVE → RESTING 87,89%. Sama.
- **Explor_Forag → INGESTIVE** (bukan ACTIVE — di taksonomi ChickenAct,
  mematuk/mencakar litter itu INGESTIVE): akurasi 62,14%, recall
  INGESTIVE 12,58%, INGESTIVE → RESTING 87,13%, recall ACTIVE 12,89%.
  Sama — dugaan tetap terbukti.

## Klip yang tak tercocokkan (1.230; 20,1%)

- ID 1011: seluruh 246 klipnya tak tercocokkan (ayam asli, tapi GT
  tidak pernah menganotasinya — GT bolong, bukan FP).
- 667 dari 1.230 (54%) ada di 200 detik terakhir video, saat GT
  makin bolong (track 3555 berhenti total di frame GT 11303; frame
  GT akhir cuma 1–3 track teranotasi).

## Implikasi

- **Angka 91,99% (test set) TIDAK pindah ke top-view: 61,8% di sini.**
  Test set itu klip sudut miring; di top-view, perilaku yang sama
  kelihatan beda. Untuk demo 2 Des, sampaikan keduanya dengan jujur:
  deteksi + tracking di top-view kuat (kotak putih 0,18%, 8/8 track
  GT tercocok), klasifikasi perilaku di top-view lemah.
- **Stage 3 tidak bisa langsung memakai label ini apa adanya:**
  durasi RESTING akan sangat kehitung lebih (model menumpuk semua
  ke RESTING). Pilihan yang belum diputuskan: (a) fine-tune VideoMAE
  pada klip top-view (GT Big Broiler menyediakan aksi per frame —
  klip latih bisa dibangun), (b) kamera miring untuk perilaku +
  top-view untuk tracking, (c) terima biasnya dengan faktor koreksi.
- **Batasan yang masih berlaku:** pemetaan 23 aksi GT → 3 kelas
  adalah perkiraan (Explor_Forag dan Stretching ambigu — tapi uji
  sensitivitas menunjukkan kesimpulan tidak berubah); GT bolong di
  video akhir; label GT klip adalah suara mayoritas 16 frame (perilaku
  bisa berganti di dalam klip); pencocokan IoU bisa salah sambung
  saat ayam berkerumun di feeder.
