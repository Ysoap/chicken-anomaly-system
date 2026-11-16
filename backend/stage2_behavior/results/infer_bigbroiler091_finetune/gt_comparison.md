# Perbandingan prediksi vs ground truth — Big Broiler 091, model FINE-TUNE v2 (17 Nov 2026)

Bandingkan label perilaku pipeline (VideoMAE-Small hasil fine-tune Big
Broiler v2, set klip **sesudah filter FP statis**: 6.119 klip, 17 track)
dengan ground truth dataset Big Broiler (94.973 baris, 8 track, aksi per
frame). Prosedur SAMA seperti baseline (model Stage 2, 16 Nov 2026):
tracking Stage 1 tidak diulang — inference jalan langsung pada
`merged_tracks_bigbroiler091.filtered.json` (17 track) dengan checkpoint
`models_finetune`; script `compare_gt.py`; angka lengkap:
`gt_comparison.json`. Per paruh: `gt_halves.json` (script
`analyze_gt_halves.py`).

## Hasil utama

- **Coverage: 4.889 / 6.119 klip tercocokkan GT (79,9%)**; 1.230 klip
  (20,1%) tak tercocokkan; 10 klip seri suara mayoritas — SAMA seperti
  baseline (himpunan klip dan pencocokan GT tidak berubah, hanya
  prediksinya yang beda).
- **Akurasi (klip tercocokkan): 87,11% (4.259 / 4.889)**, dari baseline
  61,83% (**+25,28 poin**).

### Confusion matrix (baris = GT, kolom = prediksi)

| GT \ prediksi | INGESTIVE | ACTIVE | RESTING | n GT |
|---|---|---|---|---|
| INGESTIVE | 1.461 | 25 | **63** | 1.549 |
| ACTIVE | 60 | 255 | 156 | 471 |
| RESTING | 173 | 153 | 2.543 | 2.869 |

### Akurasi per kelas

| Kelas | Recall | Precision | F1 | n GT |
|---|---|---|---|---|
| INGESTIVE | **94,32%** | 86,25% | 0,90 | 1.549 |
| ACTIVE | **54,14%** | 58,89% | 0,56 | 471 |
| RESTING | 88,64% | 92,07% | 0,90 | 2.869 |

- INGESTIVE → RESTING tinggal **4,07%** (63/1.549), dari 86,57%
  (1.341/1.549) di baseline — masalah utama TERATASI.
- RESTING → INGESTIVE naik tipis 2,96% → 6,03% (173/2.869): model
  sekarang lebih berani menyebut INGESTIVE.
- ACTIVE membaik (recall 7,86% → 54,14%) tapi tetap kelas terlemah;
  precision ACTIVE turun 86,05% → 58,89% (lebih banyak klip diprediksi
  ACTIVE).
- Distribusi prediksi sekarang mirip GT: RESTING 2.762 / INGESTIVE
  1.694 / ACTIVE 433 (GT: 2.869 / 1.549 / 471). Baseline: RESTING
  4.479 / INGESTIVE 367 / ACTIVE 43.

## Per paruh — paruh-2 adalah angka jujur

Klip 091 paruh-1 (frame < 3000) dipakai sebagai TRAINING fine-tune,
jadi akurasinya terkontaminasi. Paruh-2 (frame ≥ 3000) tidak pernah
dilatih. 9 klip yang mengangkangi batas (start 2992) dikeluarkan dari
kedua paruh.

| | Paruh-1 (terkontaminasi) | Paruh-2 (jujur) |
|---|---|---|
| Akurasi | 87,82% (2.213/2.520) — baseline 58,73% | **86,31% (2.037/2.360) — baseline 65,13% (+21,18 poin)** |
| INGESTIVE recall | 95,85% — baseline 8,66% | **92,47% — baseline 18,47% (+74,00 poin)** |
| ACTIVE recall | 60,43% — baseline 6,96% | 47,92% — baseline 8,75% |
| RESTING recall | 87,49% — baseline 96,13% | 89,76% — baseline 97,88% |
| INGESTIVE → RESTING | 2,73% — baseline 91,10% | **5,68% — baseline 81,11%** |

Paruh-1 dan paruh-2 hanya selisih 1,51 poin akurasi (87,82% vs
86,31%) dan 3,38 poin recall INGESTIVE (95,85% vs 92,47%) — perbaikan
ini generalisasi beneran, bukan hafalan data latih.

## Konsistensi dengan evaluasi test set klip

Evaluasi test set klip (kotak GT, aturan STRICT) pada paruh-2 yang
sama: akurasi 90,50%, recall INGESTIVE 93,94%. Angka video sedikit
lebih rendah (86,31% / 92,47%) karena klip video dipotong dari kotak
tracking (bukan kotak GT) dan label GT-nya suara mayoritas per klip —
wajar, dan keduanya bercerita sama: fine-tune berhasil.
