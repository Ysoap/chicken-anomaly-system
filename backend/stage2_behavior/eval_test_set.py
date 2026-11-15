"""Evaluasi test set Stage 2 (VideoMAE) pada klip ChickenAct held-out.

Memakai model hasil fine-tuning (``models/`` — best checkpoint dari
training Stage 2, 10 epoch, 3 kelas) dan frame hasil
``data_prep/extract_frames.py`` untuk split test saja. TIDAK memakai
TrainingArguments (jadi tidak kena masalah ``warmup_ratio`` di
transformers 5.x) — inference batch biasa di CPU.

Test set = 2.097 klip held-out dari manifest yang SAMA dengan training
(``data_prep/prepared/manifest.json``, seed 42: train 11.094 / val 1.799 /
test 2.097), jadi angka ini sebanding langsung dengan akurasi val
training (92,94%).

Contoh pakai (dari root repo):

    python backend/stage2_behavior/eval_test_set.py \\
        --checkpoint backend/stage2_behavior/models \\
        --frames-manifest ~/workspace/pbl-ayam-obs-rtsp/stage2_test_frames/frames_manifest.json \\
        --data-root ~/workspace/pbl-ayam-obs-rtsp/stage2_test_frames \\
        --batch-size 8

Output: ``results/test_set_eval.json`` (akurasi keseluruhan + per kelas
+ confusion matrix) di folder script ini.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import FramesDataset  # noqa: E402
from videomae_utils import CLASS_NAMES_4, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None,
                    help="Folder model hasil fine-tune (default: models/ "
                         "di sebelah script ini).")
    ap.add_argument("--frames-manifest", required=True,
                    help="frames_manifest.json hasil extract_frames.py "
                         "(split test).")
    ap.add_argument("--data-root", required=True,
                    help="Folder root frame (path relatif di manifest).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="Path JSON hasil (default: results/test_set_eval.json "
                         "di sebelah script ini).")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint = args.checkpoint or os.path.join(script_dir, "models")
    out_path = args.output or os.path.join(
        script_dir, "results", "test_set_eval.json")

    torch.set_num_threads(2)
    device = "cpu"
    print(f"Load model: {checkpoint} (device={device}) ...")
    model, _processor = load_model(checkpoint, num_labels=None,
                                   device=device)
    id2label = getattr(model.config, "id2label", None) or {}
    class_names = [id2label.get(i, id2label.get(str(i), CLASS_NAMES_4[i]))
                   for i in range(len(id2label))] if id2label \
        else list(CLASS_NAMES_4)
    print(f"Model siap. Kelas: {class_names}")

    ds = FramesDataset(args.frames_manifest, args.data_root, "test")
    print(f"Test set: {len(ds)} klip")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    y_true, y_pred = [], []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["pixel_values"].to(device)
            logits = model(pixel_values=x).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(batch["labels"].cpu().numpy().tolist())
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                done = (i + 1) * args.batch_size
                el = time.time() - t0
                print(f"  {min(done, len(ds))}/{len(ds)} klip "
                      f"({el:.0f} dtk, "
                      f"{min(done, len(ds)) / max(el, 1e-9):.1f} klip/dtk)")
    elapsed = time.time() - t0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(class_names)
    acc = float((y_true == y_pred).mean())
    per_class = {}
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    for c in range(n_classes):
        n = int((y_true == c).sum())
        correct = int(conf[c, c])
        per_class[class_names[c]] = {
            "n_clips": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else None,
        }

    result = {
        "checkpoint": checkpoint,
        "n_clips": len(ds),
        "split": "test",
        "classes": class_names,
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "rows_true_cols_pred": conf.tolist(),
            "labels": class_names,
        },
        "batch_size": args.batch_size,
        "device": device,
        "elapsed_seconds": round(elapsed, 1),
        "clips_per_second": round(len(ds) / elapsed, 2),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTest accuracy: {acc:.4f} ({len(ds)} klip, {elapsed:.0f} dtk)")
    for name, v in per_class.items():
        acc_str = f"{v['accuracy']:.4f}" if v["accuracy"] is not None else "n/a"
        print(f"  {name}: {acc_str} "
              f"({v['correct']}/{v['n_clips']})")
    print(f"Hasil tersimpan: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
