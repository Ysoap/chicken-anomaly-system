"""Utilitas bersama Stage 2: konstanta VideoMAE, preprocessing klip, dan
pemuat model.

Dipakai oleh `train.py` (fine-tuning), `inference.py` (klasifikasi klip
hasil tracking), dan `tests/test_stage2.py`.

Checkpoint awal
---------------
Checkpoint `MCG-NJU/videomae-small` (self-supervised saja) TIDAK ADA di
Hugging Face — yang tersedia untuk varian Small adalah
`MCG-NJU/videomae-small-finetuned-kinetics` (ViT-Small, ~22M parameter,
sudah fine-tune di Kinetics-400). Itulah yang dipakai sebagai titik awal:
bobot backbone-nya ditransfer, sedangkan classifier head diinisialisasi
ulang untuk 4 kelas perilaku (`ignore_mismatched_sizes=True`).

Konfigurasi checkpoint (dicek langsung dari config.json di HF, 10 Nov 2026):
arsitektur VideoMAEForVideoClassification, hidden 384, 12 layer, image
224, num_frames 16, tubelet 2, patch 16, use_mean_pooling. Preprocessing
resmi: resize (sisi pendek -> 224) + center crop 224 + normalisasi mean
ImageNet [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225].
"""

import numpy as np
import torch

# --- Model ---------------------------------------------------------------
HF_CHECKPOINT = "MCG-NJU/videomae-small-finetuned-kinetics"
NUM_FRAMES = 16          # frame per klip (standar VideoMAE)
IMAGE_SIZE = 224         # sisi input VideoMAE (persegi)
TUBELET_SIZE = 2

# --- Taksonomi 4 kelas (keputusan: versi whitepaper / dokumen part 2) -----
CLASS_NAMES_4 = ["INGESTIVE", "ACTIVE", "RESTING", "AGONISTIC"]
CLASS_TO_ID_4 = {name: i for i, name in enumerate(CLASS_NAMES_4)}
ID_TO_CLASS_4 = {i: name for name, i in CLASS_TO_ID_4.items()}

# --- Normalisasi ImageNet (sama seperti preprocessor resmi VideoMAE) -----
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_frames_uniform(n_total, n_sample=NUM_FRAMES):
    """Indeks frame yang diambil merata dari sebuah klip.

    Mengembalikan daftar indeks sepanjang ``n_sample``. Kalau klip lebih
    pendek dari ``n_sample`` frame, frame terakhir diulang (padding) supaya
    panjangnya tetap ``n_sample`` — dan pemanggil bisa mencatat berapa klip
    yang ter-padding.
    """
    if n_total <= 0:
        raise ValueError("klip kosong (0 frame)")
    if n_total >= n_sample:
        idx = np.linspace(0, n_total - 1, n_sample).round().astype(int)
        # linspace bisa menghasilkan duplikat bila n_total sedikit di atas
        # n_sample; itu tidak masalah (frame yang sama dua kali).
        return idx.tolist()
    # Klip pendek: ambil semua frame, lalu ulang frame terakhir.
    idx = list(range(n_total))
    idx += [n_total - 1] * (n_sample - n_total)
    return idx


def preprocess_frames(frames):
    """Ubah daftar frame (HWC, uint8, BGR atau RGB) menjadi tensor VideoMAE.

    Langkahnya sama seperti ``VideoMAEImageProcessor`` resmi: resize sisi
    pendek ke 224, center crop 224x224, /255, lalu normalisasi ImageNet.
    Mengembalikan tensor float32 bentuk [16, 3, 224, 224] (RGB).

    Parameter ``frames``: list/array sepanjang 16, tiap frame HWC uint8.
    Channel diasumsikan RGB; kalau input dari OpenCV (BGR), konversi dulu
    oleh pemanggil (``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)``).
    """
    import cv2

    if len(frames) != NUM_FRAMES:
        raise ValueError(f"preprocess_frames butuh {NUM_FRAMES} frame, "
                         f"dapat {len(frames)}")
    out = np.empty((NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, fr in enumerate(frames):
        h, w = fr.shape[:2]
        scale = IMAGE_SIZE / min(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # center crop 224x224
        top = (new_h - IMAGE_SIZE) // 2
        left = (new_w - IMAGE_SIZE) // 2
        cropped = resized[top:top + IMAGE_SIZE, left:left + IMAGE_SIZE]
        arr = cropped.astype(np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        out[i] = arr.transpose(2, 0, 1)
    return torch.from_numpy(out)


def load_model(checkpoint=HF_CHECKPOINT, num_labels=None, device="cpu"):
    """Load VideoMAEForVideoClassification.

    ``checkpoint`` bisa berupa id Hugging Face atau path folder hasil
    fine-tuning lokal (berisi config + bobot).
    ``num_labels``: kalau diisi (mis. 4 untuk smoke test dari checkpoint
    Kinetics), classifier head diinisialisasi ulang untuk kelas
    ``CLASS_NAMES_4[:num_labels]`` dan bobot backbone tetap dipakai.
    Kalau None (default — untuk checkpoint HASIL FINE-TUNE), config
    checkpoint dipakai apa adanya (jangan menimpa head yang sudah
    dilatih!). Salah memakai argumen ini akan merusak head fine-tune.
    """
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    if num_labels is None:
        model = VideoMAEForVideoClassification.from_pretrained(checkpoint)
    else:
        id2label = {i: CLASS_NAMES_4[i] for i in range(num_labels)}
        label2id = {n: i for i, n in id2label.items()}
        model = VideoMAEForVideoClassification.from_pretrained(
            checkpoint, num_labels=num_labels, id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True)
    model.to(device)
    model.eval()
    try:
        processor = VideoMAEImageProcessor.from_pretrained(checkpoint)
    except Exception:
        processor = None
    return model, processor


@torch.no_grad()
def predict_clip(model, clip_tensor, device="cpu"):
    """Klasifikasi satu klip. ``clip_tensor``: [16, 3, 224, 224].

    Mengembalikan (label_id, label_name, probs[4]) dengan probs = softmax.
    Nama kelas diambil dari config model (id2label); kalau tidak ada,
    dipakai CLASS_NAMES_4.
    """
    x = clip_tensor.unsqueeze(0).to(device)  # [1, 16, 3, 224, 224]
    logits = model(pixel_values=x).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    label_id = int(np.argmax(probs))
    id2label = getattr(model.config, "id2label", None) or {}
    label_name = id2label.get(label_id, id2label.get(str(label_id),
                              ID_TO_CLASS_4.get(label_id, f"class_{label_id}")))
    return label_id, label_name, probs
