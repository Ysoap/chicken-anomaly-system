# Perbandingan prediksi vs ground truth — Big Broiler 080 dengan model fine-tune v2 (sudut miring)

Video: `oblique_bigbroiler_080_720p_10fps.mp4` — 6.008 frame @10 fps, 1280×720,
setup 2 angled view ~45°, broiler umur 40 hari, 24 track GT (AGPL-3.0).
Prediksi: model fine-tune v2 (`models_finetune`), set klip sesudah filter
(`track_predictions_filtered.json`, 6.936 klip, 26 track; silo drinker ID 40
sudah dibuang — prosedur sama seperti baseline, lihat `fp_removal_080.json`).
Prosedur SAMA seperti baseline 080 (commit `11fc5a9`), hanya checkpoint diganti
`models` (Stage 2) → `models_finetune`; tracking Stage 1 TIDAK diulang —
inference jalan pada `tracks_bigbroiler080.filtered.json` (52 ID).
Script: `compare_gt.py` (sama seperti 091/080 baseline); angka lengkap:
`gt_comparison.json`. Video berlabel:
`bigbroiler080_annotated_filtered.mp4` (6.008 frame, 56.543 kotak — jumlah
kotak sama persis dengan baseline, karena track-nya sama).

## Hasil utama — akurasi TURUN, recall kelas lemah NAIK BESAR

- **Akurasi (klip tercocokkan): 75,81% (1.072 / 1.414)** — TURUN dari baseline
  80,91% (**−5,10 poin**). Coverage SAMA 20,4% (1.414 / 6.936 klip
  tercocokkan — himpunan klip yang tercocokkan identik dengan baseline,
  karena track dan GT-nya sama).
- Tapi per kelas ceritanya terbalik: **recall INGESTIVE 28,77% → 75,47%
  (+46,70 poin)** dan **recall ACTIVE 10,00% → 72,00% (+62,00 poin)**.
  Yang turun: recall RESTING 93,58% → 76,04% (−17,54 poin). Model
  tidak lagi memanggil hampir semua RESTING — tapi sekarang TERLALU
  SERING memanggil ACTIVE/INGESTIVE.
- **Macro-F1 (rata-rata F1 tiga kelas): 0,4656 → 0,5896 (+0,1240)** —
  dengan ukuran yang seimbang antar kelas, model fine-tune LEBIH BAIK.
  Akurasi keseluruhan turun karena himpunan tercocokkan 81,5%-nya RESTING
  (1.152/1.414), jadi akurasi didominasi RESTING — kelas yang recall-nya
  turun.

## Confusion matrix (baris = GT, kolom = prediksi)

| GT \ prediksi | INGESTIVE | ACTIVE | RESTING | n GT |
|---|---|---|---|---|
| INGESTIVE | 160 | 26 | **26** | 212 |
| ACTIVE | 0 | 36 | 14 | 50 |
| RESTING | 113 | **163** | 876 | 1.152 |

| Kelas | Recall (baseline → fine-tune) | Precision (baseline → fine-tune) | F1 (baseline → fine-tune) |
|---|---|---|---|
| INGESTIVE | 28,77% → **75,47%** | 49,59% → 58,61% | 0,36 → **0,66** |
| ACTIVE | 10,00% → **72,00%** | 25,00% → 16,00% | 0,14 → **0,26** |
| RESTING | 93,58% → **76,04%** | 84,82% → 95,63% | 0,89 → 0,85 |

- **INGESTIVE → RESTING tinggal 12,26% (26/212)**, dari 71,23% (151/212)
  — masalah utama baseline TERATASI di sini juga. Sebaliknya RESTING →
  INGESTIVE naik 5,12% → 9,81% (113/1.152).
- **ACTIVE over-prediksi: 163 klip RESTING GT diprediksi ACTIVE
  (14,15%), dari 15 klip (1,30%) di baseline** — precision ACTIVE jatuh
  ke 16,00% (dari 25,00%). Model memprediksi ACTIVE 225 kali pada klip
  tercocokkan (baseline: 20 kali). Verifikasi visual frame 5700: ayam
  yang jelas berbaring/istirahat (ID 6, ID 3, ID 119) berlabel ACTIVE
  0,76–0,95 — pola ini yang menggerus akurasi RESTING.
- Distribusi prediksi keseluruhan (7.432 klip mentah): RESTING 5.040 /
  ACTIVE 902 / INGESTIVE 1.490 — bandingkan baseline: RESTING 6.714 /
  ACTIVE 47 / INGESTIVE 671. Sesudah buang silo: RESTING 4.662 /
  ACTIVE 853 / INGESTIVE 1.421 (baseline: 6.256 / 47 / 633).
- Confidence: klip confidence ≥0,99 akurasinya 91,56% (434/474) — naik
  dari 84,09% di baseline; 0,9–0,99: 79,28% (398/502, dari 55,42%);
  <0,9: 54,79% (240/438, dari 49,18%). Tapi bucket ≥0,99 menyusut
  (474 vs 1.270 klip) — model fine-tune lebih jarang sangat yakin.
  Ambang confidence tetap bukan solusi.

## CAVEAT BESAR — 080 adalah data latih fine-tune (kontaminasi)

- Klip 080 dipakai sebagai TRAINING fine-tune (usable ING 79 / ACT 33 /
  REST 451, sebagian besar masuk train) — jadi angka 75,81% ini diukur
  pada klip yang modelnya SUDAH PERNAH LIHAT (meski potongan klip
  video di sini dipotong dari kotak tracking, bukan kotak GT seperti
  klip latih — adegan dan ayamnya sama). Angka ini **TIDAK BOLEH
  disajikan sebagai bukti perbaikan vs baseline 80,91%** — baseline
  diukur saat 080 belum pernah dilihat model.
- Justru karena itu hasil ini bermakna lain: walau sudah melihat
  datanya, akurasi keseluruhan tetap turun — karena perilaku model
  bergeser (prior seimbang dari training: ING 2.662 / ACT 953 /
  REST 2.662) menjauhi distribusi lapangan yang ~74–81% RESTING.
  Fine-tune memperbaiki recall kelas minoritas dengan harga
  over-prediksi kelas minoritas itu di kelas mayoritas.
- **Bukti perbaikan yang jujur hanya dari 091 paruh-2** (tidak pernah
  dilatih): akurasi video 86,31% (dari baseline paruh-2 65,13%,
  +21,18 poin), INGESTIVE recall 92,47% — lihat
  `results/infer_bigbroiler091_finetune/gt_comparison.json` +
  `gt_comparison.md`.

## Penyelarasan frame — DITERIMA DENGAN CATATAN (`--force`)

Sama persis seperti baseline (track dan GT-nya sama): pemetaan
`frame_gt = round(frame_10fps × 2,5)` diterima dengan `--force` —
kriteria ketat gagal HANYA di m0 ≤ 15 px (m0 = 15,6 px, n=108).
Penjelasan lengkap ada di `results/infer_bigbroiler080/gt_comparison.md`.
JSON mencatat `frame_alignment_verified: false,
frame_alignment_forced: true`.

## Uji sensitivitas — kesimpulan tidak berubah

- Ambang IoU 0,5 (bukan 0,3): akurasi 76,12% (1.071/1.407),
  ING→REST 12,26% (`gt_comparison_iou05.json`).
- Explor_Forag → INGESTIVE (bukan ACTIVE): akurasi 75,46%
  (1.067/1.414), ING→REST 14,73% (`gt_comparison_explor_ing.json`).
- Coverage tetap 20,4% — caveat baseline berlaku di sini juga: angka
  dihitung pada 20% klip yang kebetulan punya padanan GT, kemungkinan
  optimis.

## Verifikasi visual

Frame 3000: kotak + label menempel di ayam yang benar; ID 15/ID 4
INGESTIVE (hijau, 0,99–1,00) pada ayam menunduk di feeder — masuk
akal; ID 3 ACTIVE 0,96 pada ayam berdiri. Frame 792: ID 15/ID 4/ID 2/
ID 5 INGESTIVE (0,89–1,00) pada ayam menunduk di drinker/trough —
masuk akal. Frame 5700: ID 114 INGESTIVE 0,94 pada ayam menunduk di
drinker; tapi ID 6/ID 3/ID 119 ACTIVE (0,76–0,95) pada ayam yang
jelas berbaring — contoh over-prediksi ACTIVE di atas. Silo drinker
TANPA label di ketiga frame (ID 40 sudah dibuang).

## Implikasi

- Fine-tune menggeser bias model dari "hampir semua RESTING" menjadi
  "terlalu sering INGESTIVE/ACTIVE". Di 091 paruh-2 (data tak terlihat)
  pergeseran ini NETTO POSITIF BESAR (+21,18 poin); di 080 (data latih)
  netto akurasinya negatif (−5,10 poin) karena himpunan tercocokkannya
  didominasi RESTING — tapi recall kelas lemahnya tetap naik besar di
  kedua video.
- Untuk Stage 3: label fine-tune JANGAN dipakai mentah-mentah —
  durasi ACTIVE akan kehitung jauh lebih besar (853 klip vs 47 di
  baseline) dan sebagian besarnya salah (precision 16%). Perlu
  kalibrasi: ambang confidence per kelas, atau prior yang dikembalikan
  ke arah distribusi lapangan, sebelum aturan durasi → aman/dicurigai/
  sakit dijalankan.
- Sisa masalah terbesar tetap ACTIVE (recall naik tapi precision
  16,00% — paling rendah dari semua kelas di semua pengukuran) dan
  RESTING → ACTIVE 14,15%. Data ACTIVE train cuma 953 klip (paling
  sedikit) — butuh data ACTIVE lebih banyak / penanganan khusus.
- Angka demo yang jujur (diperbarui): test-set 91,99%; lapangan
  top-view 87,11% (video 091 fine-tune; paruh-2 jujur 86,31%);
  lapangan miring 75,81% (video 080 fine-tune — TERKONTAMINASI data
  latih, bukan ukuran generalisasi); test klip 091 paruh-2 90,50%,
  test-075 67,31%.
