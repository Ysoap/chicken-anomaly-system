"""Render video berlabel Stage 2: kotak + ID track + label perilaku.

Membaca video asli + tracks.json Stage 1 + track_predictions.json hasil
``inference.py --video``, lalu menggambar di tiap frame:

- kotak bbox (diskalakan ke ukuran video, sama seperti clip_builder)
- "ID <n> · <LABEL> <conf>" di atas kotak

Label per frame diambil dari klip yang jendelanya mencakup frame itu
(mayoritas bila ada beberapa klip tumpang tindih); kalau tidak ada klip
yang mencakup frame itu, dipakai label mayoritas track dari
track_summary. Warna berbeda per kelas supaya mudah dibaca.

    python render_annotated_video.py --video /path/ke/video.mp4 \\
        --tracks /path/ke/tracks.json \\
        --predictions /path/ke/track_predictions.json \\
        --out /path/ke/video_berlabel.mp4
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_builder import detect_coord_scale, load_tracks  # noqa: E402

# Warna BGR per kelas (mudah dibedakan di atas kandang).
CLASS_COLORS = {
    "INGESTIVE": (0, 200, 0),    # hijau
    "ACTIVE": (0, 165, 255),     # oranye
    "RESTING": (255, 100, 0),    # biru
}
DEFAULT_COLOR = (255, 255, 255)  # putih, kalau label tak dikenal


def label_for_frame(track_id, frame_idx, clip_results, track_summary):
    """Label + confidence track di satu frame.

    Kembalikan (label, confidence, sumber) dengan sumber 'clip' (dari
    klip yang mencakup frame) atau 'track' (fallback label mayoritas).
    """
    covering = [r for r in clip_results
                if r["track_id"] == track_id
                and r["start_frame"] <= frame_idx <= r["end_frame"]]
    if covering:
        # Mayoritas label klip yang tumpang tindih; confidence = rata-rata
        # confidence klip dengan label mayoritas itu.
        from collections import Counter
        votes = Counter(r["label"] for r in covering)
        label = votes.most_common(1)[0][0]
        confs = [r["confidence"] for r in covering if r["label"] == label]
        return label, sum(confs) / len(confs), "clip"
    summary = track_summary.get(str(track_id)) or track_summary.get(track_id)
    if summary:
        label = summary["majority_label"]
        conf = summary["mean_probs"].get(label)
        return label, conf, "track"
    return None, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="Video asli.")
    ap.add_argument("--tracks", required=True, help="tracks.json Stage 1.")
    ap.add_argument("--predictions", required=True,
                    help="track_predictions.json hasil inference.py --video.")
    ap.add_argument("--out", required=True, help="MP4 hasil render.")
    ap.add_argument("--font-scale", type=float, default=0.45)
    ap.add_argument("--thickness", type=int, default=2)
    args = ap.parse_args()

    with open(args.predictions) as f:
        pred = json.load(f)
    clip_results = pred["clip_results"]
    track_summary = pred["track_summary"]
    print(f"Prediksi: {len(clip_results)} klip, "
          f"{len(track_summary)} track terklasifikasi "
          f"(checkpoint: {pred.get('checkpoint')})")
    if pred.get("pretrained_smoke_test_only"):
        print("PERINGATAN: prediksi ini dari checkpoint pretrained — "
              "label TIDAK ADA ARTINYA. Render dibatalkan.")
        return 1

    _fps, per_track, (max_x, max_y) = load_tracks(args.tracks)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = detect_coord_scale(vw, vh, max_x, max_y)
    print(f"Video {vw}x{vh} @ {fps:.1f} fps; skala bbox: "
          f"x{sx:.4f} y{sy:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (vw, vh))
    if not writer.isOpened():
        raise RuntimeError(f"Tidak bisa membuka writer: {args.out}")

    frame_idx = 0
    n_drawn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, by_frame in per_track.items():
            bbox = by_frame.get(frame_idx)
            if bbox is None:
                continue
            x1, y1, x2, y2 = (bbox[0] * sx, bbox[1] * sy,
                              bbox[2] * sx, bbox[3] * sy)
            x1, y1, x2, y2 = (int(round(x1)), int(round(y1)),
                              int(round(x2)), int(round(y2)))
            label, conf, _src = label_for_frame(
                tid, frame_idx, clip_results, track_summary)
            if label is None:
                text = f"ID {tid}"
                color = DEFAULT_COLOR
            else:
                conf_txt = f" {conf:.2f}" if conf is not None else ""
                text = f"ID {tid} {label}{conf_txt}"
                color = CLASS_COLORS.get(label, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.thickness)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, 1)
            ty = max(th + 4, y1 - 4)
            # Jaga label tetap di dalam frame (kotak di tepi kanan/bawah
            # sebelumnya terpotong).
            tx = max(0, min(x1, vw - tw - 6))
            cv2.rectangle(frame, (tx, ty - th - 4),
                          (tx + tw + 4, ty + 2), color, -1)
            cv2.putText(frame, text, (tx + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, args.font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)
            n_drawn += 1
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    print(f"Selesai: {frame_idx} frame, {n_drawn} kotak digambar -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
