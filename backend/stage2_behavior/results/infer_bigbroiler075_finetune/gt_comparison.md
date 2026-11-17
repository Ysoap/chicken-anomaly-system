# Perbandingan prediksi vs ground truth — Big Broiler 075 dengan model fine-tune v2 (sudut miring, HELD-OUT)

Video: `oblique_bigbroiler_075_720p_10fps.mp4` — 6.066 frame @10 fps, 1280×720,
setup 2 angled view ~45°, broiler umur 38 hari, 26 track GT (AGPL-3.0).
Prediksi: model fine-tune v2 (`models_finetune`), set klip sesudah filter
(`track_predictions_filtered.json`, 6.851 klip, 43 track terklasifikasi;
drinker ID 33 + ID 112 sudah dibuang — lihat `fp_removal_075.json`).
Tracking Stage 1 dikerjakan dulu khusus untuk video ini (tidak ada pipeline
075 sebelumnya): 6.066 frame, 61.878 box, 100 ID mentah → 62 ID sesudah
filter statis 10 px (38 ID dibuang). Script: `run_075_track.sh` +
`run_075_finetune.sh`; `compare_gt.py` (metode sama seperti 091/080:
IoU ≥ 0,3, pemetaan `frame_gt = round(frame_10fps × 2,5)`). Video berlabel:
`bigbroiler075_annotated_filtered.mp4` (6.066 frame, 57.397 kotak).

## 075 ADALAH VIDEO HELD-OUT — angka ini ukuran generalisasi yang valid

- **075 TIDAK dipakai di training fine-tune v2** (train: 091, 080, 067, 057,
  060, 064). Tidak seperti 080 yang terkontaminasi data latih, angka
  075 diukur pada video yang modelnya BELUM PERNAH LIHAT — jadi ini
  ukuran generalisasi yang sebenarnya.
- **Tidak ada baseline 075 dengan model Stage 2** (pipeline video penuh
  075 belum pernah ada) — jadi angka ini berdiri sendiri: tidak bisa
  dibilang "naik/turun vs Stage 2", hanya bisa dibaca apa adanya dan
  dibandingkan dengan evaluasi klip test-075 (67,31%) yang mengukur
  video yang sama dengan kotak GT.

## Hasil utama — akurasi 62,84%, over-prediksi INGESTIVE berlanjut

- **Akurasi (klip tercocokkan): 62,84% (301 / 479)**. Coverage cuma
  **7,0% (479 / 6.851 klip tercocokkan)** — jauh di bawah 080 (20,4%):
  GT 075 hanya menganotasi ~1,46 kotak/frame (10.243 baris / 7.002
  frame beranotasi) dari ~10 ayam yang terlihat, dan frame beranotasi
  cuma 46,3% (2.811/6.066 frame 10 fps). Perkiraan kasar: (1,46/10) ×
  46,3% ≈ 6,8% — cocok dengan 7,0% yang teramati. Angka dihitung pada
  7% klip yang kebetulan punya padanan GT — berisik, jangan
  over-interpretasi.
- **INGESTIVE GT cuma 14 klip** (tidak ada aksi `Eat` sama sekali di GT
  075; INGESTIVE di sini hanya `Drink`). Recall INGESTIVE 92,86%
  (13/14) terlihat bagus tapi artinya hampir tidak ada — yang penting
  precision-nya: **12,87%**. Model memprediksi INGESTIVE 101× pada
  klip tercocokkan, cuma 13 yang benar.
- Confusion (baris = GT, kolom = prediksi): INGESTIVE [13, 0, 1] n=14;
  ACTIVE [33, 90, 56] n=179; RESTING [55, 33, 198] n=286.

| Kelas | Recall | Precision | F1 | n GT |
|---|---|---|---|---|
| INGESTIVE | **92,86%** | 12,87% | 0,23 | 14 |
| ACTIVE | 50,28% | 73,17% | 0,60 | 179 |
| RESTING | 69,23% | 77,65% | 0,73 | 286 |

- **RESTING → INGESTIVE 19,23% (55/286)** dan ACTIVE → INGESTIVE
  18,44% (33/179) — model terlalu sering memanggil INGESTIVE di video
  baru. Pola yang SAMA terlihat di evaluasi klip test-075 (67,31%):
  INGESTIVE recall 100% (4/4) tapi precision 10,26% (over-prediksi
  39×, 35 salah). Dua pengukuran independen pada video yang sama
  sepakat: **fine-tune over-prediksi INGESTIVE di video tak terlihat**.
- Sebaliknya INGESTIVE → RESTING tinggal 7,14% (1/14) — masalah
  utama baseline (86,57% di 091) tetap teratasi di sini.
- ACTIVE: recall 50,28% (90/179), precision 73,17% — paling seimbang
  dari tiga kelas di video ini; error terbesar ACTIVE → RESTING
  31,28% (56/179).
- RESTING recall 69,23% (198/286) — model memanggil 55 klip RESTING
  sebagai INGESTIVE dan 33 sebagai ACTIVE.
- Macro-F1: (0,23 + 0,60 + 0,73) / 3 = **0,52**.
- Distribusi prediksi keseluruhan (7.325 klip mentah): RESTING 5.428 /
  ACTIVE 665 / INGESTIVE 1.232. Sesudah buang drinker (6.851 klip):
  RESTING 5.116 / ACTIVE 658 / INGESTIVE 1.077.
- Confidence: klip confidence ≥0,99 akurasinya 72,19% (109/151);
  0,9–0,99: 59,86% (88/147); <0,9: 57,46% (104/181). Ambang
  confidence tetap bukan solusi.

## GT yang tidak ikut terbanding — 16,8% baris GT terbuang oleh pemetaan

`compare_gt.py` (ACTION_MAP, sama seperti 091/080) tidak memetakan:
`Sleeping` (887 baris), `Short_Leg_Move` (316), `Other` (192),
`Video_Problem` (16), `Out_Of_Sight` (305) — total 1.716 dari 10.243
baris GT (16,8%) tidak pernah bisa tercocokkan, karena kelasnya `None`.
Catatan: `Sleeping` jelas-jelas RESTING dan `Short_Leg_Move` jelas
ACTIVE (pemetaan itu dipakai di builder dataset fine-tune), tapi
`compare_gt.py` tidak mengenalnya — di 091/080 aksi itu tidak muncul
di GT, jadi tidak pernah jadi masalah. Akibatnya himpunan tercocokkan
075 kehilangan sebagian besar klip istirahat-tidur. Prosedur TIDAK
diubah di tengah jalan supaya tetap sebanding dengan 091/080; kalau
nanti pemetaan diperluas, angka 075 (dan 091/080) harus dihitung ulang
semua.

## Penyelarasan frame — TERVERIFIKASI (bukan --force seperti 080)

Berbeda dengan 080 (m0 = 15,6 px, diterima pakai `--force`), di 075
kriteria ketat LOLOS: pemetaan `frame_gt = round(frame_10fps × 2,5)`
terverifikasi, median jarak pusat pada frame bergerak m0 = 7,2 px
(n=50), frac<10px = 0,66; offset ±25 frame jelas lebih buruk
(15,4 / 21,3 px). JSON mencatat `frame_alignment_verified: true,
frame_alignment_forced: false`. (Flag `--force` tetap dipakai di
perintah karena skrip dijalankan dengan argumen yang sama seperti
080 — tapi verifikasi sudah lolos tanpa perlu pemaksaan.)

## Uji sensitivitas — kesimpulan tidak berubah

- Ambang IoU 0,5 (bukan 0,3): akurasi 63,62% (459 tercocokkan),
  ING→REST tetap kecil (`gt_comparison_iou05.json`).
- Explor_Forag → INGESTIVE (bukan ACTIVE): akurasi 51,77% (479
  tercocokkan) — TURUN 11 poin dari 62,84%
  (`gt_comparison_explor_ing.json`). Artinya model memperlakukan
  Explor_Forag (3.150 baris GT, aksi kedua terbesar di 075) lebih seperti
  ACTIVE ketimbang INGESTIVE: 90 klip ACTIVE GT yang diprediksi
  benar jadi salah kalau labelnya dibalik. Pemetaan UTAMA (Explor_Forag
  → ACTIVE) tetap yang dipakai.

## Verifikasi visual

Frame 600: kotak + label menempel di ayam yang benar; semua label
RESTING (0,82–1,00) pada ayam berbaring — masuk akal. Drinker TANPA
label. Frame 3000: ID 24 INGESTIVE 0,99 (hijau) pada ayam menunduk di
drinker/trough — masuk akal; ID 28 RESTING 0,94 pada ayam berbaring;
drinker TANPA label. Frame 5000: ID 24 INGESTIVE pada ayam berdiri di
trough menunduk; ID 62 INGESTIVE 0,57 di feeder; ID 3 ACTIVE 0,91
pada ayam berdiri; tapi ID 116 ACTIVE 0,55 pada ayam yang tampak
berbaring di tepi kanan — contoh over-prediksi ACTIVE yang konsisten
dengan angka (RESTING → ACTIVE 11,54%). Drinker TANPA label di ketiga
frame (ID 33/112 sudah dibuang).

## Implikasi

- **Fine-tune TIDAK menggeneralisasi sebaik di 091 paruh-2.** Di 091
  (top-view, paruh-2 tak terlihat) akurasi naik +21,18 poin ke
  86,31%; di 075 (miring, tak terlihat) akurasi cuma 62,84% —
  konsisten dengan evaluasi klip test-075 (67,31%). Perbaikan
  generalisasi yang terbukti hanya di top-view; di sudut miring video
  baru, model masih bermasalah — terutama over-prediksi INGESTIVE
  (precision 12,87%) dan RESTING yang ketukar.
- Catatan jujur soal perbandingan: 075 (miring, n=479, coverage 7%,
  ING n=14) dan 091 paruh-2 (top-view, n=2.360, coverage 79,9%)
  adalah dua video yang sangat berbeda — sudut, umur ayam (38 vs 32
  hari), dan kerapatan GT-nya beda. Selisih 86,31% vs 62,84% bukan
  semata "sudut miring lebih susah" — 075 juga jauh lebih sedikit
  datanya untuk diukur. Tapi arahnya jelas: klaim perbaikan harus
  dibatasi ke top-view sampai ada bukti miring yang lebih kuat.
- Untuk Stage 3: label fine-tune JANGAN dipakai mentah-mentah —
  durasi INGESTIVE akan jauh kehitung lebih di video baru (1.077 dari
  6.851 klip diprediksi INGESTIVE, padahal GT-nya hampir tidak ada
  adegan makan) dan sebagian besarnya salah. Perlu kalibrasi
  (ambang confidence per kelas / prior) sebelum aturan durasi →
  aman/dicurigai/sakit dijalankan.
- Sisa masalah terbesar tetap sama: ACTIVE lemah + over-prediksi
  INGESTIVE di video baru. Data ACTIVE train cuma 953 klip (paling
  sedikit) — butuh data ACTIVE lebih banyak / penanganan khusus.
- Angka demo yang jujur (diperbarui): test-set 91,99%; lapangan
  top-view 87,11% (video 091 fine-tune; paruh-2 jujur 86,31%);
  lapangan miring HELD-OUT 62,84% (video 075 fine-tune — ukuran
  generalisasi yang valid, coverage 7%, n=479); lapangan miring
  75,81% (video 080 fine-tune — TERKONTAMINASI data latih, bukan
  ukuran generalisasi); test klip 091 paruh-2 90,50%, test-075 67,31%.
