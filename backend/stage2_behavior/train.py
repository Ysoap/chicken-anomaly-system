"""Fine-tuning VideoMAE-Small di ChickenAct (4 kelas perilaku).

Titik awal: ``MCG-NJU/videomae-small-finetuned-kinetics`` (ViT-Small,
~22M parameter; lihat ``videomae_utils.py`` soal kenapa checkpoint ini —
``MCG-NJU/videomae-small`` yang self-supervised saja TIDAK ADA di HF).
Classifier head diinisialisasi ulang untuk kelas perilaku
(``ignore_mismatched_sizes=True``); bobot backbone ditransfer. Jumlah
kelas diturunkan dari manifest — saat ini 3 kelas (INGESTIVE / ACTIVE /
RESTING); AGONISTIC dicadangkan tanpa data latih (lihat
``data_prep/label_mapping.py``).

Input: manifest frame hasil ``data_prep/extract_frames.py`` — 16 frame
JPEG per klip (sampling merata dari klip asli) + label 4 kelas. Membaca
JPEG jauh lebih cepat daripada decode mp4 tiap epoch.

Jalan utama: notebook Colab (``notebooks/Colab-ChickenAct-VideoMAE.ipynb``)
yang memanggil script ini + sync checkpoint ke Drive tiap 120 detik.
Script ini juga bisa jalan standalone (mesin sendiri / Colab tanpa Drive):

    python train.py --frames-manifest prepared_frames/frames_manifest.json \\
        --data-root prepared_frames --output-dir runs/stage2_v1 \\
        --epochs 10 --train-batch 16

    # lanjutkan dari checkpoint Trainer (mis. sesi Colab putus):
    python train.py --frames-manifest ... --output-dir runs/stage2_v1 --resume

    # smoke test cepat (100 klip, 1 epoch, CPU):
    python train.py --frames-manifest ... --epochs 1 --max-train-samples 100 \\
        --max-eval-samples 20 --train-batch 2 --eval-batch 2 --no-fp16

Output: checkpoint Trainer per epoch di ``--output-dir`` + model terbaik
(``best/``) + ``train_metrics.json`` (akurasi val terbaik + akurasi per
kelas). Notebook menyalin folder ini ke Drive.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from videomae_utils import (  # noqa: E402
    CLASS_NAMES_4,
    HF_CHECKPOINT,
    IMAGE_SIZE,
    NUM_FRAMES,
    preprocess_frames,
)


class FramesDataset(Dataset):
    """Dataset dari frames_manifest.json: 16 JPEG per klip + label."""

    def __init__(self, manifest_path, data_root, split):
        with open(manifest_path) as f:
            entries = json.load(f)
        self.items = [e for e in entries if e["split"] == split]
        self.data_root = data_root
        if not self.items:
            raise ValueError(f"Tidak ada entri split={split} di {manifest_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import cv2

        e = self.items[idx]
        frames = []
        for rel in e["frame_files"]:
            img = cv2.imread(os.path.join(self.data_root, rel))
            if img is None:
                raise FileNotFoundError(f"Frame hilang: {rel}")
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if len(frames) != NUM_FRAMES:
            raise ValueError(f"Klip {e['clip']} punya {len(frames)} frame, "
                             f"harusnya {NUM_FRAMES}")
        pixel_values = preprocess_frames(frames)
        return {"pixel_values": pixel_values,
                "labels": torch.tensor(e["label4_id"], dtype=torch.long)}


def compute_metrics(eval_pred):
    """Akurasi + akurasi per kelas (numpy saja, tanpa sklearn).

    Hanya kelas yang benar-benar ada di batch eval yang dilaporkan.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    out = {"accuracy": acc}
    for i in sorted(set(labels.tolist())):
        name = CLASS_NAMES_4[i] if i < len(CLASS_NAMES_4) else f"class_{i}"
        mask = labels == i
        out[f"accuracy_{name.lower()}"] = float(
            (preds[mask] == labels[mask]).mean())
        out[f"n_{name.lower()}"] = int(mask.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-manifest", required=True)
    ap.add_argument("--data-root", required=True,
                    help="Folder root frame (path relatif di manifest).")
    ap.add_argument("--checkpoint", default=HF_CHECKPOINT)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-train-samples", type=int, default=0,
                    help="Batasi klip train (0 = semua). Untuk smoke test.")
    ap.add_argument("--max-eval-samples", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="Path checkpoint Trainer untuk resume (atau 'auto').")
    ap.add_argument("--no-fp16", action="store_true",
                    help="Matikan fp16 (mis. saat smoke test di CPU).")
    args = ap.parse_args()

    from transformers import (
        Trainer,
        TrainingArguments,
        VideoMAEForVideoClassification,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} (cuda={torch.cuda.is_available()})")
    if device == "cpu" and not args.no_fp16:
        print("CPU terdeteksi — fp16 dimatikan otomatis.")
        args.no_fp16 = True

    train_ds = FramesDataset(args.frames_manifest, args.data_root, "train")
    eval_ds = FramesDataset(args.frames_manifest, args.data_root, "val")
    if args.max_train_samples:
        train_ds.items = train_ds.items[:args.max_train_samples]
    if args.max_eval_samples:
        eval_ds.items = eval_ds.items[:args.max_eval_samples]
    print(f"Train: {len(train_ds)} klip, val: {len(eval_ds)} klip")

    # Jumlah kelas diturunkan dari manifest (bukan hardcode): saat ini
    # ChickenAct hanya berisi 3 kelas yang ada datanya (id 0,1,2 =
    # INGESTIVE/ACTIVE/RESTING); AGONISTIC (id 3) dicadangkan tanpa data
    # (lihat label_mapping.py). Kalau nanti ada data AGONISTIC, num_labels
    # otomatis menjadi 4 tanpa perubahan kode.
    active_ids = sorted({e["label4_id"] for e in train_ds.items}
                        | {e["label4_id"] for e in eval_ds.items})
    assert active_ids == list(range(len(active_ids))), (
        f"Id label di manifest tidak berurutan 0..N-1: {active_ids}. "
        f"Perbaiki pemetaan label dulu.")
    num_labels = len(active_ids)
    active_names = [CLASS_NAMES_4[i] for i in active_ids]
    print(f"Kelas yang dilatih ({num_labels}): {active_names}")
    id2label = {i: n for i, n in enumerate(active_names)}
    label2id = {n: i for i, n in enumerate(active_names)}
    model = VideoMAEForVideoClassification.from_pretrained(
        args.checkpoint, num_labels=num_labels,
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    print(f"Model: {args.checkpoint} -> {num_labels} kelas "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameter)")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.05,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=not args.no_fp16,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds, compute_metrics=compute_metrics)

    resume_from = None
    if args.resume == "auto":
        ckpts = sorted(
            d for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")) if os.path.isdir(args.output_dir) else []
        resume_from = os.path.join(args.output_dir, ckpts[-1]) if ckpts else None
    elif args.resume:
        resume_from = args.resume
    if resume_from:
        print(f"Resume dari: {resume_from}")

    result = trainer.train(resume_from_checkpoint=resume_from)
    print(f"Training selesai. train_loss akhir: {result.training_loss:.4f}")

    metrics = trainer.evaluate()
    print("Metrik val (model terbaik):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)
    print(f"Model terbaik disimpan: {best_dir}")
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump({"train_loss": result.training_loss,
                   "val_metrics_best": metrics,
                   "checkpoint": args.checkpoint,
                   "num_labels": num_labels,
                   "classes": active_names,
                   "epochs": args.epochs, "lr": args.lr,
                   "train_batch": args.train_batch}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
