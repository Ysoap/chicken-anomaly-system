"""Inference Stage 2: klasifikasi perilaku ayam per track (VideoMAE).

Alur: video -> (Stage 1) deteksi + tracking -> tracklet -> klip 16 frame
(``clip_builder.py``) -> VideoMAE -> label perilaku + confidence per
track. Training memakai klip ChickenAct langsung; inference di video
kandang memakai klip hasil crop tracklet.

    python inference.py --video /path/ke/video.mp4 --tracks /path/ke/tracks.json \\
        --checkpoint /path/ke/stage2_finetuned --out-dir out/
    python inference.py --clip /path/ke/klip.mp4 \\
        --checkpoint MCG-NJU/videomae-small-finetuned-kinetics

PERINGATAN — baca sebelum memakai hasil: kalau ``--checkpoint`` masih
checkpoint pretrained Kinetics (``MCG-NJU/videomae-small-finetuned-kinetics``,
bukan hasil fine-tune ChickenAct), classifier head-nya diinisialisasi
acak untuk 4 kelas — label yang keluar TIDAK ADA ARTINYA dan hanya
membuktikan plumbing jalan (smoke test). Label baru bermakna setelah
fine-tuning di ChickenAct selesai (lihat ``train.py`` + notebook) dan
checkpoint hasil fine-tune yang dipakai. Script ini mencetak peringatan
itu setiap kali checkpoint pretrained dipakai.

Input tracking: ``tracks.json`` dari Stage 1
(``backend/stage1_detection/inference.py --track``). Bobot Stage 1 tidak
dibutuhkan di sini — deteksi + tracking diasumsikan sudah jalan dan
hasilnya sudah ada sebagai file.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data_prep"))
from clip_builder import build_clips, load_tracks, read_video_frames  # noqa: E402
from clip_extractor import extract_clip_frames  # noqa: E402
from videomae_utils import (  # noqa: E402
    CLASS_NAMES_4,
    HF_CHECKPOINT,
    NUM_FRAMES,
    load_model,
    predict_clip,
    preprocess_frames,
)


def is_pretrained_checkpoint(checkpoint):
    """True kalau checkpoint adalah titik awal Kinetics (belum fine-tune)."""
    name = os.path.basename(str(checkpoint).rstrip("/"))
    return name in ("videomae-small-finetuned-kinetics",
                    "videomae-small-finetuned-ssv2",
                    "videomae-base",
                    "videomae-base-finetuned-kinetics") or checkpoint == HF_CHECKPOINT


def class_names_of(model):
    """Nama kelas model sesuai config-nya (urut id)."""
    id2label = getattr(model.config, "id2label", None) or {}
    if id2label:
        return [id2label.get(i, id2label.get(str(i), f"class_{i}"))
                for i in range(len(id2label))]
    return list(CLASS_NAMES_4)


def probs_dict(model, probs):
    """Ubah array probabilitas -> dict {nama_kelas: prob}."""
    names = class_names_of(model)
    return {names[i]: round(float(p), 4) for i, p in enumerate(probs)}


def classify_clip_file(model, video_path, device="cpu"):
    """Klasifikasi satu file klip video. Kembalikan dict hasil."""
    frames, n_total = extract_clip_frames(video_path)
    tensor = preprocess_frames(frames)
    label_id, label, probs = predict_clip(model, tensor, device=device)
    return {"clip": os.path.basename(video_path),
            "n_frames_total": n_total,
            "label_id": label_id, "label": label,
            "probs": probs_dict(model, probs),
            "confidence": round(float(probs[label_id]), 4)}


def classify_track_clips(model, video_path, tracks_path, device="cpu",
                         clip_len=NUM_FRAMES, stride=8, margin=1.3):
    """Alur penuh video+tracks -> klip per track -> klasifikasi.

    Mengembalikan (clip_results, track_summary). Tiap klip diklasifikasi
    satu-satu; ringkasan per track memakai voting mayoritas + rata-rata
    probabilitas dari semua klip track itu.
    """
    from clip_builder import detect_coord_scale

    frames_bgr, video_fps = read_video_frames(video_path)
    vh, vw = frames_bgr[0].shape[:2]
    tracks_fps, per_track, (max_x, max_y) = load_tracks(tracks_path)
    scale = detect_coord_scale(vw, vh, max_x, max_y)
    print(f"Video: {len(frames_bgr)} frame {vw}x{vh}; "
          f"track: {len(per_track)} ID; skala koordinat: {scale}")
    clips = build_clips(frames_bgr, per_track, clip_len=clip_len,
                        stride=stride, margin=margin, scale=scale)
    print(f"Klip terbentuk: {len(clips)}")

    clip_results = []
    for clip in clips:
        tensor = preprocess_frames(list(clip["frames"]))
        label_id, label, probs = predict_clip(model, tensor, device=device)
        clip_results.append({
            "track_id": clip["track_id"],
            "start_frame": clip["start_frame"],
            "end_frame": clip["end_frame"],
            "label_id": label_id, "label": label,
            "probs": probs_dict(model, probs),
            "confidence": round(float(probs[label_id]), 4),
        })

    by_track = defaultdict(list)
    for r in clip_results:
        by_track[r["track_id"]].append(r)
    names = class_names_of(model)
    track_summary = {}
    for tid in sorted(by_track):
        rs = by_track[tid]
        votes = Counter(r["label"] for r in rs)
        mean_probs = np.mean(
            [[r["probs"][c] for c in names] for r in rs], axis=0)
        top = int(np.argmax(mean_probs))
        track_summary[str(tid)] = {
            "n_clips": len(rs),
            "majority_label": votes.most_common(1)[0][0],
            "label_by_mean_prob": names[top],
            "mean_probs": {names[i]: round(float(p), 4)
                           for i, p in enumerate(mean_probs)},
            "label_counts": dict(votes),
        }
    return clip_results, track_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=HF_CHECKPOINT,
                    help="Id HF atau folder checkpoint hasil fine-tune.")
    ap.add_argument("--clip", default=None,
                    help="Klasifikasi satu file klip video (mode klip).")
    ap.add_argument("--video", default=None,
                    help="Video kandang (mode track; butuh --tracks).")
    ap.add_argument("--tracks", default=None,
                    help="tracks.json Stage 1 (mode track).")
    ap.add_argument("--out-dir", default="out_stage2")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if bool(args.clip) == bool(args.video):
        ap.error("Pilih tepat satu mode: --clip ATAU (--video + --tracks).")
    if args.video and not args.tracks:
        ap.error("Mode track butuh --tracks (tracks.json Stage 1).")

    pretrained = is_pretrained_checkpoint(args.checkpoint)
    if pretrained:
        print("=" * 70)
        print("PERINGATAN: checkpoint ini masih titik awal Kinetics (BELUM "
              "fine-tune ChickenAct).")
        print("Label yang dihasilkan TIDAK ADA ARTINYA — hanya smoke test "
              "plumbing (model jalan, bentuk output benar).")
        print("Pakai checkpoint hasil fine-tune setelah training selesai.")
        print("=" * 70)

    print(f"Load model: {args.checkpoint} (device={args.device}) ...")
    # Checkpoint pretrained (Kinetics): head di-init ulang 4 kelas untuk
    # smoke test. Checkpoint hasil fine-tune: dipakai apa adanya — JANGAN
    # menimpa num_labels, nanti head fine-tune rusak.
    model, _processor = load_model(
        args.checkpoint, num_labels=4 if pretrained else None,
        device=args.device)
    print(f"Model siap. Kelas model: {class_names_of(model)}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.clip:
        result = classify_clip_file(model, args.clip, device=args.device)
        print(json.dumps(result, indent=2))
        out = {"mode": "clip", "checkpoint": args.checkpoint,
               "pretrained_smoke_test_only": pretrained, "result": result}
        out_path = os.path.join(args.out_dir, "clip_prediction.json")
    else:
        clip_results, track_summary = classify_track_clips(
            model, args.video, args.tracks, device=args.device,
            stride=args.stride, margin=args.margin)
        dist = Counter(r["label"] for r in clip_results)
        print(f"\nDistribusi label {len(clip_results)} klip: {dict(dist)}")
        print(f"Track terklasifikasi: {len(track_summary)}")
        out = {"mode": "track", "checkpoint": args.checkpoint,
               "pretrained_smoke_test_only": pretrained,
               "video": os.path.basename(args.video),
               "n_clips": len(clip_results),
               "label_distribution": dict(dist),
               "clip_results": clip_results,
               "track_summary": track_summary}
        out_path = os.path.join(args.out_dir, "track_predictions.json")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Hasil tersimpan: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
