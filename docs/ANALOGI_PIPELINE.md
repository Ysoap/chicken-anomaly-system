# ?? Analogi Pipeline 3-Stage: Panduan Memahami Sistem

> **Dokumen ini adalah kunci pemahaman tim.** Baca dengan saksama sebelum coding!

---

## ?? Overview

Sistem deteksi anomali ayam ini bekerja dalam 3 tahap berurutan:

```
RTSP Stream (OBS ? MediaMTX)
    ?
[Stage 1: RT-DETR + ByteTrack] ? Deteksi & Tracking
    ? (bounding boxes + track IDs)
[Stage 2: VideoMAE] ? Klasifikasi Perilaku
    ? (behavior labels per chicken)
[Stage 3: LSTM-AE] ? Deteksi Anomali
    ? (anomaly score + diagnosis)
[FastAPI] ? Firebase Realtime DB
    ?
[Mobile App] ? Push Notification ??
```

**Input**: Video RTSP dari kandang ayam (1280x720, 10 FPS)  
**Output**: Skor anomali real-time + diagnosis (Firebase ? Mobile push notification)

---

## ?? Stage 1: RT-DETRv2-L + ByteTrack

### ?? Analogi: Security Guard di Shopping Mall

#### **RT-DETR = Mata Guard (NMS-free Detection)**

**Masalah YOLO Tradisional:**
- Seperti guard yang harus "double-check" kalau 2 orang berdiri dekat: apakah ini 1 orang atau 2 orang?
- Proses double-check ini disebut **Non-Maximum Suppression (NMS)**
- NMS lambat dan kadang salah hapus deteksi yang sebenarnya benar

**Keunggulan RT-DETR:**
- Guard yang bisa lihat 100 orang sekaligus tanpa bingung (transformer-based)
- **No NMS needed** ? deteksi langsung akurat tanpa post-processing
- Kenapa penting? Di kandang ayam yang padat, ayam sering overlap. NMS-free = deteksi lebih lengkap!

**Technical Details:**
- Model: RT-DETRv2-Large (pretrained COCO ? fine-tuned ChickenDet)
- Input: RGB frames 640x640
- Output: Bounding boxes [x1, y1, x2, y2] + confidence scores

---

#### **ByteTrack = Memori Guard (Tracking)**

**Masalah:**
- Ayam jalan di belakang feeder ? detector kehilangan 2 detik
- Bagaimana kita tahu kalau ayam yang muncul lagi adalah ayam yang sama?

**Solusi ByteTrack:**
- Guard ingat "ayam dengan tag merah tadi jalan ke kiri 2 detik lalu, deteksi baru ini kemungkinan ayam yang sama"
- **Tracking low-confidence detections** ? Kalman filter prediksi posisi selama occlusion
- Setiap ayam dapat **Track ID unik** (contoh: chicken_0042)

**Technical Details:**
- Kalman filter memprediksi posisi ayam saat occluded
- IoU matching untuk asosiasi track baru dengan track lama
- Track ID persistent selama ayam masih di frame

**Output Stage 1:**
- Bounding boxes dengan Track ID per ayam
- ~30 FPS pada RTX 3060

---

## ?? Stage 2: VideoMAE-Small (Behavior Classification)

### ?? Analogi: Guru Tari Menilai Murid

#### **Kenapa Tidak Pakai 2D CNN (ResNet)?**

**Masalah:**
- Foto statis tidak bisa bedakan apakah murid sedang waltz atau tango
- Butuh lihat **MOTION** sepanjang waktu (video = dimensi temporal)

**Video Transformer (VideoMAE) Solution:**
- Butuh 16 frames (~1.6 detik) untuk menangkap gerakan lengkap
- Bisa bedakan "kepala turun-naik berulang" (makan) vs "kepala tilt + leher extend" (minum)

---

#### **Magic 90% Masking: Kenapa VideoMAE Powerful?**

**Tanpa Masking (Model Biasa):**
- Model malas ? cuma lihat tekstur (bulu putih, alas krem ? "memorize aja ayam = putih")
- Tidak belajar motion patterns!

**Dengan 90% Masking (VideoMAE):**
- Analogi: Guru tutup mata 90% waktu, tapi tetap tahu jenis tarian dengan **merasakan ritme**
- Technical: Random tube masking ? model rekonstruksi frame yang hilang ? **forced to learn inter-frame dependencies**

**Kenapa CRITICAL untuk Ayam?**
- Ayam putih + alas krem = **low visual contrast** (nyaris identik secara penampilan)
- Perilaku hanya beda di **motion patterns**:
  - **Eating**: Kepala bobbing turun/naik repeatedly
  - **Drinking**: Kepala tilt + neck extension
  - **Walking**: Steady forward motion
  - **Resting**: Minimal movement
- Static appearance hampir sama ? VideoMAE masking **paksa model ignore appearance, fokus motion**!

---

#### **Technical Details Stage 2:**

**Input:**
- 16 frames per track (~1.6s @ 10 FPS)
- Per-chicken video clips (crop dari bounding box Stage 1)

**Label Mapping (15 ? 4 classes):**
- ChickenAct punya 15 kelas ? kita merge jadi 4:
  - pecking, oraging, scratching ? **eating**
  - standing, sitting, sleeping ? **resting**
  - walking, unning ? **walking**
  - drinking ? **drinking**

**Output:**
- Softmax probabilities: {eating: 0.7, drinking: 0.1, walking: 0.1, resting: 0.1}
- ~10 clips/sec inference speed (RTX 3060)

**Output Stage 2:**
- Behavior label per chicken per 1.6 detik

---

## ?? Stage 3: LSTM Autoencoder (Unsupervised Anomaly Detection)

### ?? Analogi: Dokter yang Hapal Detak Jantung Normal

#### **Kenapa Unsupervised?**

**Masalah:**
- Kita tidak punya data labeled "anomaly" (bagaimana kita tahu anomali masa depan seperti apa?)
- Contoh anomali: serangan predator, overcrowding, penyakit menular ? unpredictable!

**Solusi:**
- Train HANYA pada perilaku normal ? apapun yang berbeda = anomaly
- Seperti dokter yang cuma hapal detak jantung sehat ? kalau ada yang aneh langsung tahu!

---

#### **LSTM = Memori Temporal Dokter**

**Autoencoder Biasa:**
- "Detak jantung sekarang 80 BPM" (no context)
- Tidak tahu apakah ini normal atau tidak

**LSTM Autoencoder:**
- "30 detik terakhir rata-rata 70 BPM, tiba-tiba loncat 120 BPM = anomaly!"
- **LSTM cells** ingat sequence 30-60 detik ? detect temporal deviations

---

#### **Cara Kerja LSTM Autoencoder:**

**1. Training (Unsupervised):**
- Kasih model 10 jam video perilaku normal
- Model belajar rekonstruksi pola normal (encode ? decode)
- Reconstruction error rendah untuk pola normal

**2. Inference (Real-time):**
- Kasih model window 30-60 detik baru
- Model coba rekonstruksi
- **Anomaly Score = Reconstruction Error (MSE)**
  - Pola normal: Error rendah (model kenal pola ini)
  - Pola anomali: Error tinggi (model bingung, never seen before!)

**3. Threshold (3-Sigma Rule):**
- Hitung mean (µ) dan std (s) dari training error
- **Threshold = µ + 3s**
- Alarm trigger hanya kalau error > threshold (99.7% confidence ? minimize false alarms)

**Architecture:**
- Encoder: LSTM(12 ? 64 ? 32)
- Decoder: LSTM(32 ? 64 ? 12)
- Input: 12 features, 30-60s window
- Output: Single anomaly score (0-1)

---

#### **12 Custom Features: Analogi Konser Crowd**

Setiap 1 detik, kita ekstrak 12 fitur dari hasil Stage 1+2:

| Feature | Analogi Konser | Normal Range | Anomaly Example |
|---------|---------------|--------------|-----------------|
| **1. Mean Velocity** | Seberapa cepat orang bergerak | 0.5-2 m/s | Stampede = 5 m/s |
| **2. Velocity Std** | Variasi kecepatan | Low std = seragam | High std = chaos |
| **3. Mean NND** | Seberapa rapat orang berdiri | Mosh pit vs spread out | Terlalu rapat/renggang |
| **4. NND Std** | Variasi jarak | Stabil | Tiba-tiba berubah drastis |
| **5. Overlap Ratio** | Stage diving (tumpuk-tumpukan) | 5-10% | 40% = overcrowding |
| **6. Entropy** | Variasi aktivitas | Sebagian dance, sebagian duduk | Semua frozen = 0 entropy |
| **7. Directional Coherence** | Semua hadap stage vs random | 0.6-0.8 | Semua hadap dinding = 0.2 |
| **8. Eating %** | % orang makan | 20-40% | 5% suddenly = masalah pakan |
| **9. Resting %** | % orang duduk | 30-50% | 90% = exhaustion |
| **10. Active Track Count** | Occupancy rate | 50-90% | 10% = mass exodus |
| **11. Track Turnover Rate** | Seberapa sering orang masuk/keluar | 2-3/min | 20/min = panic |
| **12. Motion Intensity** | Energi gerakan rata-rata | Medium | Tiba-tiba semua frozen |

**Extraction Code:** ackend/stage3_anomaly/feature_extractor.py

---

#### **Output Stage 3:**

```json
{
  "anomaly_score": 0.87,
  "threshold": 0.75,
  "is_anomaly": true,
  "diagnosis": {
    "top_features": [
      {"name": "velocity", "zscore": 3.2, "contribution": 0.35},
      {"name": "nnd", "zscore": 2.8, "contribution": 0.28}
    ],
    "summary": "Abnormally high movement speed + crowding detected"
  }
}
```

---

## ?? Multi-Threading Pipeline Design

**File:** ackend/integration/pipeline.py

**Challenge:**
- 3 stages harus jalan bersamaan (parallel)
- Stage 2 dan 3 butuh data dari stage sebelumnya

**Solution: 3 Threads + FIFO Queue**

```
Thread 1 (GPU): Stage 1 Detection (30 FPS)
    ? Queue 1 (bounding boxes + track IDs)
Thread 2 (GPU): Stage 2 Behavior (buffer 16 frames per track)
    ? Queue 2 (behavior labels)
Thread 3 (CPU): Stage 3 Anomaly (aggregate 1s, LSTM window 30-60s)
    ? FastAPI ? Firebase
```

**Buffer Manager:**
- FIFO buffer per Track ID
- Stage 2 menunggu sampai 16 frames terkumpul untuk track tertentu
- Stage 3 menunggu sampai 30-60s data terkumpul

**Communication:**
- Python Queue (thread-safe)
- Azis (Stage 1) push ke Queue 1
- Malik (Stage 2) consume Queue 1, push ke Queue 2
- Ysoap (Stage 3) consume Queue 2, calculate anomaly

---

## ? Performance Targets

| Metric | Target |
|--------|--------|
| Stage 1 FPS | 30 |
| Stage 2 FPS | 10 clips/sec |
| End-to-end latency | <2 seconds |
| False positive rate | <5% |
| Anomaly detection recall | >90% |

---

## ?? Key Takeaways

1. **Stage 1 (RT-DETR + ByteTrack)**: Security guard mata tajam + memori kuat
2. **Stage 2 (VideoMAE 90% masking)**: Guru tari yang fokus motion, bukan kostum
3. **Stage 3 (LSTM-AE unsupervised)**: Dokter yang hapal pola sehat, langsung tahu kalau ada yang aneh
4. **12 Features**: Concert crowd metrics ? deteksi stampede sebelum terlambat

**Next Steps:**
- Azis ? Implementasi Stage 1
- Malik ? Implementasi Stage 2
- Ysoap ? Feature engineering + LSTM-AE
- Pandu ? Mobile monitoring

**Questions?** ? Tag di GitHub Issues atau diskusi standup!

---

*Dokumen ini hidup! Update jika ada insight baru selama implementasi.*