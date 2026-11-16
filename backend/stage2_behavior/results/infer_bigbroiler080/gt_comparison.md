# Perbandingan prediksi vs ground truth — Big Broiler 080 (sudut miring)

Video: `oblique_bigbroiler_080_720p_10fps.mp4` — 6.008 frame @10 fps, 1280×720,
setup 2 angled view ~45°, broiler umur 40 hari, 24 track GT (AGPL-3.0).
Prediksi: set klip sesudah filter (`track_predictions_filtered.json`, 6.936 klip,
26 track; silo drinker ID 40 sudah dibuang — lihat `fp_removal_080.json`).
Script: `compare_gt.py` (sama seperti 091); angka lengkap: `gt_comparison.json`.

## Hasil utama

- **Akurasi (klip tercocokkan): 80,91% (1.144 / 1.414)** — naik dari 61,83% di
  top-view 091 (+19,1 poin), tapi masih jauh dari 91,99% test-set (sudut miring).
- **Coverage HANYA 20,4% (1.414 / 6.936 klip tercocokkan)** — GT setup 2 jarang:
  hanya ~2,25 track/frame yang dianotasi (091: ~8 track/frame, coverage 79,9%).
  5.522 klip tak tercocokkan, 3 seri suara mayoritas.
- **BACA DENGAN HATI-HATI:** angka 80,91% dihitung pada 20% klip yang kebetulan
  punya padanan GT — kemungkinan BERSIFAT OPTIMIS (ayam yang dianotasi
  cenderung yang mudah diamati; distribusi GT pada klip tercocokkan:
  RESTING 81,5% vs GT keseluruhan ~74%).

## Confusion matrix (baris = GT, kolom = prediksi)

| GT \ prediksi | INGESTIVE | ACTIVE | RESTING | n GT |
|---|---|---|---|---|
| INGESTIVE | 61 | 0 | **151** | 212 |
| ACTIVE | 3 | 5 | 42 | 50 |
| RESTING | 59 | 15 | 1.078 | 1.152 |

| Kelas | Recall | Precision | F1 |
|---|---|---|---|
| INGESTIVE | **28,77%** | 49,59% | 0,36 |
| ACTIVE | **10,00%** | 25,00% | 0,14 |
| RESTING | **93,58%** | 84,82% | 0,89 |

- **INGESTIVE → RESTING masih dominan: 71,23% klip INGESTIVE GT diprediksi
  RESTING (151/212)**, turun dari 86,57% di 091 — sudut miring membantu, tapi
  tidak menyelesaikan. Sebaliknya cuma 5,12%.
- ACTIVE praktis tidak dikenali (recall 10%, n GT cuma 50 klip — sampel kecil).
- Confidence: klip confidence ≥0,99 akurasinya 84,09% (1.068/1.270) — lebih baik
  dari 091 (63,1%), tapi tetap 16% salah walau sangat yakin; 0,9–0,99: 55,4%;
  <0,9: 49,2%. Ambang confidence tetap bukan solusi.

## Penyelarasan frame — DITERIMA DENGAN CATATAN (`--force`)

Pemetaan `frame_gt = round(frame_10fps × 2,5)`:

- Lembah jarak pusat (frame ayam GT sedang bergerak, n≈108): median 15,6 px di
  offset 0; 13,3–14,7 px di +1..+5; 17,4–22,0 px di −1..−3; 26,6 px di −5;
  44,3–50,5 px di ±25 frame (±1 dtk); 92,9–159,9 px di ±250 frame (±10 dtk).
  Bentuk lembah benar (datar di 0..+5, naik curam di offset jauh).
- **Kriteria ketat script GAGAL HANYA di satu syarat: m0 ≤ 15 px (m0 = 15,6 px,
  selisih 0,6 px).** Dua syarat rasio lolos (m25 = 50,5 ≥ 2,5×15,6 = 39,0;
  m250 = 159,9 ≥ 4×15,6 = 62,4). Uji tambahan dengan ambang gerak lebih longgar
  (>8 dan >5 px/frame, n≈127) memberi lembah yang sama.
- Penyebab: ambang mutlak 15 px dikalibrasi untuk GT rapat 091 (n≈1.320 pasangan,
  m0 = 8,9 px). Di GT jarang 080 sampelnya kecil (n≈108) sehingga median
  berisik; minimum lembah di +2/+3 frame (+80–120 mdtk) bedanya cuma ~2 px dari
  offset 0 — di bawah noise gerak, sesuai toleransi ±3 frame yang ditulis
  script itu sendiri.
- Bukti IoU pendukung TIDAK informatif di sini (median 0,000 di SEMUA offset):
  GT hanya menganotasi ~2,25 track/frame, jadi sebagian besar kotak prediksi
  tidak punya padanan GT sama sekali; fraksi IoU≥0,5 ≈ 0,23 di semua offset =
  fraksi kotak yang kebetulan teranotasi. Ini artefak coverage, bukan bukti
  pemetaan salah.
- Karena itu perbandingan dijalankan dengan flag `--force` (baru ditambahkan ke
  `compare_gt.py`, perilaku default tetap ketat). JSON mencatat
  `frame_alignment_verified: false, frame_alignment_forced: true` + tabel
  lengkap, supaya siapa pun bisa menilai sendiri.

## Uji sensitivitas — kesimpulan tidak berubah

- Ambang IoU 0,5 (bukan 0,3): akurasi 81,17% (1.142/1.407), coverage 20,3%,
  ING→REST 71,23% (`gt_comparison_iou05.json`).
- Explor_Forag → INGESTIVE (bukan ACTIVE): akurasi 80,91% (1.144/1.414),
  ING→REST 72,77% (`gt_comparison_explor_ing.json`).

## Distribusi aksi GT 080 (26.922 baris, 24 track)

INGESTIVE (Eat 6,2% + Drink 7,6%) = 13,8%; ACTIVE (Locomotion 1,2% +
Explor_Forag 0,8% + Stretching 0,7% + Feather_Pecking 0,3%) = 3,0%; RESTING
(Inactive 48,3% + Panting 20,3% + Preening 4,5% + aksi kosong 0,9%) = 74,0%;
Out_Of_Sight 1,3%; TIDAK TERPETAKAN 8,0% (Sleeping 3,8%, Short_Leg_Move 2,7%,
Video_Problem 0,7%, Sneezing/Defecating/Flapping/Swallowing/Head_Shaking
~0,8%) — baris-baris ini tidak menyumbang suara GT (pemetaan sama seperti 091
demi keterbandingan).

## Implikasi

- Sudut miring memperbaiki akurasi lapangan (61,8% → 80,9%) tapi TIDAK
  mengembalikan 91,99% test-set. Penyebab sisa: (1) domain shift tetap ada
  (kandang Big Broiler ≠ kandang ChickenAct; ayam lebih besar, litter beda,
  kamera diam), (2) INGESTIVE tetap ketukar RESTING 71% — dari sudut miring
  pun makan/minum sering terlihat diam, (3) ACTIVE hampir tidak dikenali
  (recall 10%).
- Stage 3 tetap tidak bisa memakai label ini apa adanya — durasi RESTING akan
  kehitung lebih (~90% prediksi vs ~74% GT), INGESTIVE kehitung kurang.
- Opsi yang belum diputuskan: (a) fine-tune VideoMAE pada klip Big Broiler
  (GT 080 menyediakan aksi per frame, tapi jarang — 080 + 075 + 091 bisa
  digabung), (b) terima bias dengan faktor koreksi, (c) kamera miring untuk
  perilaku + top-view untuk tracking (sudah terbukti keduanya di video ini:
  tracking 24/24 track GT tercocok, klasifikasi perilaku yang lemah).
- Untuk demo 2 Des: sampaikan 91,99% sebagai akurasi test-set, 80,9% sebagai
  akurasi lapangan sudut miring (coverage 20%, kemungkinan optimis), dan 61,8%
  sebagai akurasi lapangan top-view.
