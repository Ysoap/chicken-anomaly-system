"""Clip builder: tracklet Stage 1 -> klip 16 frame untuk VideoMAE.

Mengambil output tracking Stage 1 (``tracks.json`` dari
``backend/stage1_detection/inference.py --track``: bbox + Track ID per
frame) dan sebuah file video, lalu membangun klip 16 frame per track:
tiap frame di-crop di sekitar bbox ayam (crop persegi + margin, supaya
ayam tidak terpotong dan ada sedikit konteks kandang), di-resize ke
224x224 — format input VideoMAE ([16, 3, 224, 224]).

Dipakai oleh ``inference.py`` Stage 2 (alur video -> detect+track ->
klip -> klasifikasi). Training TIDAK memakai builder ini: training
membaca klip ChickenAct langsung (sudah satu ayam per klip) lewat
``train.py``.

GOTCHA KOORDINAT (ditemukan 10 Nov 2026): bbox di ``tracks.json`` bisa
berada di ruang koordinat ASLI frame (mis. 1400x900), sementara video
yang dibuka adalah re-encode yang lebih kecil (mis. 960x618 — seperti
video sample di ``tracking_tests/samples/``). Builder mendeteksi ini
otomatis: kalau bbox maksimum melebihi ukuran frame video, semua bbox
diskalakan seragam ke ukuran video. Override manual tersedia lewat
``--coord-width`` / ``--coord-height``. Selalu periksa pesan "skala
koordinat" di log sebelum percaya pada hasil crop.

Contoh pakai:

    python clip_builder.py --video /path/ke/video.mp4 \\
        --tracks /path/ke/tracks.json --out-dir out/clips

    # dari video sample Stage 1 (koordinat track 1400x900, video 960x618):
    python clip_builder.py --video ../../stage1_detection/tracking_tests/samples/fCam16.20230221_130001.mp4 \\
        --tracks ../../stage1_detection/tracking_tests/summaries/tracks_fCam16.20230221_130001.json \\
        --out-dir out/clips_fcam16
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from videomae_utils import IMAGE_SIZE, NUM_FRAMES  # noqa: E402

DEFAULT_MARGIN = 1.3   # crop persegi = max(w, h) bbox * margin
DEFAULT_STRIDE = 8     # geser jendela klip tiap 8 frame (overlap 50%)


def load_tracks(tracks_path):
    """Baca tracks.json Stage 1. Kembalikan (fps, {track_id: {frame: bbox}})."""
    with open(tracks_path) as f:
        data = json.load(f)
    fps = float(data.get("fps", 10.0))
    per_track = {}
    max_x = max_y = 0.0
    for fr in data["frames"]:
        fi = int(fr["frame"])
        for t in fr["tracks"]:
            x1, y1, x2, y2 = [float(v) for v in t["bbox"]]
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            per_track.setdefault(int(t["track_id"]), {})[fi] = (x1, y1, x2, y2)
    return fps, per_track, (max_x, max_y)


def read_video_frames(video_path, max_frames=10000):
    """Baca semua frame video (BGR, uint8). Kembalikan list + fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) > max_frames:
            cap.release()
            raise ValueError(
                f"Video terlalu panjang (>{max_frames} frame) untuk "
                f"clip_builder offline. Potong dulu videonya.")
    cap.release()
    if not frames:
        raise ValueError(f"Video kosong / tidak ada frame: {video_path}")
    return frames, fps


def detect_coord_scale(video_w, video_h, max_bbox_x, max_bbox_y,
                       coord_width=None, coord_height=None):
    """Hitung faktor skala bbox -> koordinat frame video.

    Kalau pemanggil memberi ukuran koordinat track secara eksplisit, pakai
    itu. Kalau tidak, dan bbox maksimum melebihi ukuran video, asumsikan
    bbox berada di ruang koordinat asli (lebih besar) dan skalakan seragam
    dengan rasio sisi yang paling membatasi.
    """
    if coord_width and coord_height:
        return video_w / coord_width, video_h / coord_height
    if max_bbox_x > video_w or max_bbox_y > video_h:
        s = min(video_w / max_bbox_x, video_h / max_bbox_y)
        return s, s
    return 1.0, 1.0


def square_crop(frame, bbox, margin=DEFAULT_MARGIN, size=IMAGE_SIZE):
    """Crop persegi di sekitar bbox (BGR) -> resize (RGB, size x size).

    Sisi crop = max(lebar, tinggi bbox) * margin, berpusat di tengah bbox,
    dipotong (clip) ke dalam frame. Mengembalikan array RGB uint8.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * margin
    half = side / 2.0
    sx1 = int(round(max(0, cx - half)))
    sy1 = int(round(max(0, cy - half)))
    sx2 = int(round(min(w, cx + half)))
    sy2 = int(round(min(h, cy + half)))
    if sx2 <= sx1 or sy2 <= sy1:
        raise ValueError(f"Crop kosong untuk bbox {bbox} di frame {w}x{h}")
    crop = frame[sy1:sy2, sx1:sx2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def build_clips(frames, per_track, clip_len=NUM_FRAMES, stride=DEFAULT_STRIDE,
                margin=DEFAULT_MARGIN, scale=(1.0, 1.0)):
    """Bangun klip dari frame + track.

    Mengembalikan list dict: {track_id, start_frame, end_frame,
    frames: np.ndarray [clip_len, 224, 224, 3] uint8 RGB}.
    Jendela 16 frame berurutan; track harus hadir di SEMUA frame jendela
    (kalau ada yang hilang — mis. oklusi — jendela itu dilewati).
    """
    sx, sy = scale
    clips = []
    for track_id in sorted(per_track):
        by_frame = per_track[track_id]
        frame_ids = sorted(by_frame)
        if not frame_ids:
            continue
        start_min, start_max = frame_ids[0], frame_ids[-1] - clip_len + 1
        for start in range(start_min, start_max + 1, stride):
            window = range(start, start + clip_len)
            if any(fi not in by_frame for fi in window):
                continue
            arr = np.empty((clip_len, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for k, fi in enumerate(window):
                x1, y1, x2, y2 = by_frame[fi]
                bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                arr[k] = square_crop(frames[fi], bbox, margin=margin)
            clips.append({"track_id": track_id,
                          "start_frame": start,
                          "end_frame": start + clip_len - 1,
                          "frames": arr})
    return clips


def contact_sheet(clip_frames, thumb=112):
    """Susun 16 frame klip jadi satu gambar 4x4 (untuk cek visual)."""
    import cv2 as _cv2

    thumbs = [_cv2.resize(f, (thumb, thumb),
                           interpolation=_cv2.INTER_LINEAR)
              for f in clip_frames]
    rows = []
    for r in range(4):
        rows.append(np.hstack(thumbs[r * 4:(r + 1) * 4]))
    sheet = np.vstack(rows)
    return _cv2.cvtColor(sheet, _cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="File video input.")
    ap.add_argument("--tracks", required=True,
                    help="tracks.json hasil Stage 1 inference.py --track.")
    ap.add_argument("--out-dir", default="out_clips")
    ap.add_argument("--clip-len", type=int, default=NUM_FRAMES)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--coord-width", type=float, default=None,
                    help="Lebar ruang koordinat bbox di tracks.json (opsional).")
    ap.add_argument("--coord-height", type=float, default=None)
    ap.add_argument("--max-clips", type=int, default=0,
                    help="Batasi jumlah klip yang disimpan (0 = semua).")
    ap.add_argument("--contact-sheets", type=int, default=3,
                    help="Jumlah contact sheet PNG contoh yang disimpan.")
    args = ap.parse_args()

    print(f"Video : {args.video}")
    print(f"Tracks: {args.tracks}")
    frames, video_fps = read_video_frames(args.video)
    vh, vw = frames[0].shape[:2]
    print(f"Frame video: {len(frames)} frame, {vw}x{vh}, {video_fps:.1f} fps")
    tracks_fps, per_track, (max_x, max_y) = load_tracks(args.tracks)
    print(f"Track: {len(per_track)} ID unik, bbox maks x={max_x:.0f} y={max_y:.0f}")
    scale = detect_coord_scale(vw, vh, max_x, max_y,
                               args.coord_width, args.coord_height)
    print(f"Skala koordinat bbox -> video: x{scale[0]:.4f} y{scale[1]:.4f}"
          + (" (bbox diskalakan — ruang koordinat track != ukuran video)"
             if scale != (1.0, 1.0) else " (tidak ada penskalaan)"))
    if abs(tracks_fps - video_fps) > 0.5:
        print(f"PERINGATAN: fps tracks.json ({tracks_fps}) != fps video "
              f"({video_fps}) — indeks frame mungkin tidak sejajar.")

    clips = build_clips(frames, per_track, clip_len=args.clip_len,
                        stride=args.stride, margin=args.margin, scale=scale)
    print(f"Klip terbentuk: {len(clips)} (dari {len(per_track)} track, "
          f"jendela {args.clip_len} frame, stride {args.stride})")
    if not clips:
        print("Tidak ada klip yang terbentuk. Kemungkinan: tidak ada track "
              "yang hadir penuh 16 frame berurutan (cek ID switch di "
              "tracking — lihat tracking_tests).")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    n_save = len(clips) if args.max_clips <= 0 else min(args.max_clips, len(clips))
    summary = []
    for i, clip in enumerate(clips[:n_save]):
        np.save(os.path.join(
            args.out_dir,
            f"track{clip['track_id']:03d}_f{clip['start_frame']:04d}.npy"),
            clip["frames"])
        summary.append({"track_id": clip["track_id"],
                        "start_frame": clip["start_frame"],
                        "end_frame": clip["end_frame"],
                        "file": f"track{clip['track_id']:03d}_f{clip['start_frame']:04d}.npy"})
    for i, clip in enumerate(clips[:args.contact_sheets]):
        cv2.imwrite(
            os.path.join(args.out_dir,
                         f"sheet_track{clip['track_id']:03d}_f{clip['start_frame']:04d}.png"),
            contact_sheet(clip["frames"]))
    with open(os.path.join(args.out_dir, "clips_summary.json"), "w") as f:
        json.dump({"video": os.path.basename(args.video),
                   "n_frames": len(frames),
                   "n_tracks": len(per_track),
                   "n_clips": len(clips),
                   "n_saved": n_save,
                   "clip_len": args.clip_len, "stride": args.stride,
                   "margin": args.margin, "coord_scale": scale,
                   "clips": summary}, f, indent=2)
    # Ringkasan per track (berapa klip per ID).
    from collections import Counter
    per_track_n = Counter(c["track_id"] for c in clips)
    print(f"Klip disimpan: {n_save} file .npy + "
          f"{min(args.contact_sheets, len(clips))} contact sheet di {args.out_dir}/")
    print(f"Track dengan klip terbanyak: {per_track_n.most_common(5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
