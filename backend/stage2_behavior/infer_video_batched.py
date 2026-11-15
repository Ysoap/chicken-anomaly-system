"""Inference Stage 2 di video panjang — streaming + batch (hemat memori).

Versi ``inference.py`` memuat SEMUA frame video ke RAM sekaligus
(``read_video_frames``) — untuk video 6000 frame 1280x720 itu ~16 GB,
tidak muat di VM (~867 MB tersedia). Script ini melakukan hal yang sama
secara hasil, tapi streaming:

- video dibaca frame demi frame; hanya 16 frame terakhir disimpan
  (buffer putar, ~44 MB);
- tiap jendela 16 frame (stride 8, margin 1.3 — SAMA seperti
  ``clip_builder.build_clips``) dibangun saat frame terakhir jendelanya
  tiba, lalu diklasifikasi dalam batch (default 8 klip per forward).
  Batching tidak mengubah hasil: model eval mode + LayerNorm, jadi tiap
  klip independen; preprocessing per klip identik
  (``preprocess_frames``);
- hasil ditulis bertahap ke JSONL (satu baris per klip) + state file
  berisi indeks frame terakhir yang selesai, supaya kalau putus bisa
  dilanjutkan dengan ``--resume`` tanpa mengulang dari awal.

Urutan klip di ``track_predictions.json`` akhir diurutkan
(track_id, start_frame) seperti ``inference.py``; ``track_summary``
dihitung dengan cara yang sama (voting mayoritas + rata-rata probabilitas).

    ~/workspace/.venv-rtdetr/bin/python infer_video_batched.py \\
        --video .../bigbroiler_091_720p_10fps.mp4 \\
        --tracks .../tracks_bigbroiler091.json \\
        --checkpoint backend/stage2_behavior/models \\
        --out-dir backend/stage2_behavior/results/infer_bigbroiler091
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_builder import load_tracks, square_crop  # noqa: E402
from videomae_utils import (  # noqa: E402
    CLASS_NAMES_4,
    NUM_FRAMES,
    load_model,
    preprocess_frames,
)

DEFAULT_MARGIN = 1.3
DEFAULT_STRIDE = 8


def windows_for_track(by_frame, clip_len=NUM_FRAMES, stride=DEFAULT_STRIDE):
    """Daftar start frame jendela 16 frame berurutan penuh (sama seperti
    build_clips: mulai dari frame pertama track, lompat stride, jendela
    harus hadir penuh)."""
    frame_ids = sorted(by_frame)
    if not frame_ids:
        return []
    starts = []
    for s in range(frame_ids[0], frame_ids[-1] - clip_len + 2, stride):
        if all((s + k) in by_frame for k in range(clip_len)):
            starts.append(s)
    return starts


@torch.no_grad()
def predict_batch(model, tensors, device="cpu"):
    """Klasifikasi satu batch tensor klip [B, 16, 3, 224, 224]."""
    x = torch.stack(tensors).to(device)
    logits = model(pixel_values=x).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    id2label = getattr(model.config, "id2label", None) or {}
    out = []
    for p in probs:
        lid = int(np.argmax(p))
        name = id2label.get(lid, id2label.get(str(lid),
                             CLASS_NAMES_4[lid] if lid < len(CLASS_NAMES_4)
                             else f"class_{lid}"))
        out.append((lid, name, p))
    return out


def flush_batch(model, batch, clip_results, device):
    """Klasifikasi batch yang terkumpul, tambahkan ke clip_results."""
    if not batch:
        return
    tensors = [c["tensor"] for c in batch]
    preds = predict_batch(model, tensors, device)
    names = [c["meta"] for c in batch]
    for (lid, name, p), meta in zip(preds, names):
        probs = {n: round(float(v), 4)
                 for n, v in zip(meta["names"], p)}
        clip_results.append({
            "track_id": meta["track_id"],
            "start_frame": meta["start_frame"],
            "end_frame": meta["start_frame"] + NUM_FRAMES - 1,
            "label_id": lid, "label": name,
            "probs": probs,
            "confidence": round(float(p[lid]), 4),
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Folder model fine-tune Stage 2.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Batasi frame video yang diproses (0 = semua).")
    ap.add_argument("--resume", action="store_true",
                    help="Lanjutkan dari clip_results.jsonl yang sudah ada.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "clip_results.jsonl")
    state_path = os.path.join(args.out_dir, "infer_state.json")

    # Jendela klip per track (dari tracks.json, tanpa video).
    tracks_fps, per_track, (max_x, max_y) = load_tracks(args.tracks)
    windows = {}   # track_id -> set(start_frame)
    n_windows = 0
    for tid, by_frame in per_track.items():
        ws = windows_for_track(by_frame, stride=args.stride)
        windows[tid] = set(ws)
        n_windows += len(ws)
    print(f"Track: {len(per_track)} ID; jendela klip (16 frame, stride "
          f"{args.stride}): {n_windows}")
    if n_windows == 0:
        print("Tidak ada klip yang terbentuk.")
        return 1

    done_starts = set()
    last_frame_done = -1
    clip_results = []
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                clip_results.append(r)
                done_starts.add((r["track_id"], r["start_frame"]))
        if os.path.exists(state_path):
            with open(state_path) as f:
                last_frame_done = json.load(f).get("last_frame_done", -1)
        print(f"Resume: {len(clip_results)} klip sudah ada; "
              f"frame terakhir selesai: {last_frame_done}")
        n_windows -= len(done_starts)
        print(f"Sisa jendela: {n_windows}")

    print(f"Load model: {args.checkpoint} (device={args.device}) ...")
    model, _ = load_model(args.checkpoint, num_labels=None,
                          device=args.device)
    id2label = getattr(model.config, "id2label", None) or {}
    names = [id2label.get(i, id2label.get(str(i), CLASS_NAMES_4[i]))
             for i in range(len(id2label) or len(CLASS_NAMES_4))]
    print(f"Model siap. Kelas: {names}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vw}x{vh} @ {fps:.1f} fps, {n_total} frame")
    scale_x, scale_y = 1.0, 1.0
    if max_x > vw or max_y > vh:
        s = min(vw / max_x, vh / max_y)
        scale_x = scale_y = s
        print(f"Skala koordinat bbox -> video: {s:.4f}")

    out_f = open(jsonl_path, "a")
    buf = deque(maxlen=NUM_FRAMES)  # frame BGR terakhir (buffer putar)
    batch = []
    frame_idx = -1
    n_new = 0
    import time
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        buf.append(frame)
        if frame_idx < NUM_FRAMES - 1:
            continue
        end = frame_idx
        start = end - NUM_FRAMES + 1
        # Jendela yang sudah selesai dikenali lewat done_starts (di bawah),
        # bukan lewat last_frame_done — sebuah jendela baru selesai saat
        # frame AKHIRnya tiba, jadi membandingkan start dengan
        # last_frame_done akan salah melewati jendela yang belum selesai.
        for tid in sorted(windows):
            if start not in windows[tid]:
                continue
            if (tid, start) in done_starts:
                continue
            by_frame = per_track[tid]
            # Bangun klip: crop tiap frame jendela (frame ke-k jendela
            # ada di buf[k] karena buffer berisi 16 frame terakhir).
            try:
                frames_rgb = []
                for k in range(NUM_FRAMES):
                    fi = start + k
                    x1, y1, x2, y2 = by_frame[fi]
                    bbox = (x1 * scale_x, y1 * scale_y,
                            x2 * scale_x, y2 * scale_y)
                    frames_rgb.append(
                        square_crop(buf[k], bbox, margin=args.margin))
            except ValueError as e:
                print(f"  lewati klip track {tid} f{start}: {e}")
                continue
            tensor = preprocess_frames(frames_rgb)
            batch.append({"tensor": tensor,
                          "meta": {"track_id": tid, "start_frame": start,
                                   "names": names}})
            if len(batch) >= args.batch_size:
                before = len(clip_results)
                flush_batch(model, batch, clip_results, args.device)
                for r in clip_results[before:]:
                    out_f.write(json.dumps(r) + "\n")
                out_f.flush()
                n_new += len(clip_results) - before
                batch = []
        if frame_idx % 200 == 0:
            el = time.time() - t0
            print(f"frame {frame_idx}: {n_new} klip baru "
                  f"({el / 60:.1f} mnt)", flush=True)
            with open(state_path, "w") as f:
                json.dump({"last_frame_done": frame_idx}, f)
    cap.release()
    if batch:
        before = len(clip_results)
        flush_batch(model, batch, clip_results, args.device)
        for r in clip_results[before:]:
            out_f.write(json.dumps(r) + "\n")
        out_f.flush()
        n_new += len(clip_results) - before
    out_f.close()
    with open(state_path, "w") as f:
        json.dump({"last_frame_done": frame_idx}, f)

    clip_results.sort(key=lambda r: (r["track_id"], r["start_frame"]))
    dist = Counter(r["label"] for r in clip_results)
    by_track = defaultdict(list)
    for r in clip_results:
        by_track[r["track_id"]].append(r)
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
    out = {"mode": "track", "checkpoint": args.checkpoint,
           "pretrained_smoke_test_only": False,
           "video": os.path.basename(args.video),
           "n_clips": len(clip_results),
           "label_distribution": dict(dist),
           "clip_results": clip_results,
           "track_summary": track_summary}
    out_path = os.path.join(args.out_dir, "track_predictions.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    el = time.time() - t0
    print(f"\nSelesai: {len(clip_results)} klip ({n_new} baru) dalam "
          f"{el / 60:.1f} menit")
    print(f"Distribusi label: {dict(dist)}")
    print(f"Track terklasifikasi: {len(track_summary)}")
    print(f"Hasil: {out_path} (+ {jsonl_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
