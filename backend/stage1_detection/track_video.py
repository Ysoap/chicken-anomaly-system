"""Tracking Stage 1 di video panjang: RT-DETRv2-S + ByteTrack (streaming).

Dipakai pertama kali untuk video demo Big Broiler 091 (6000 frame @10 fps,
1280x720). Berbeda dengan uji tracking multi-video (72 frame, threshold
0,3), script ini:

- membaca video frame demi frame (streaming, memori kecil — VM hanya
  punya ~867 MB tersedia),
- threshold deteksi 0,6 (ayam asli di video ini skor 0,93+, FP benda mati
  statis 0,54-0,74 — jangan over-tune diam-diam, lihat README Stage 2),
- ByteTrack ``supervision`` 0.25.1, lost_track_buffer 30 (sama seperti
  baseline fCam16), frame_rate 10,
- menyimpan checkpoint JSON tiap N frame (atomic rename) supaya kalau
  proses putus di tengah, hasil sampai frame terakhir tidak hilang dan
  bisa dilanjutkan dengan ``--resume-from``.

Detektor sama persis seperti ``prediksi_contoh.py`` (bobot EMA best.pth,
resize 640x640 + /255, ``orig_size`` = [w, h] frame asli).

    ~/workspace/.venv-rtdetr/bin/python track_video.py \\
        --video ~/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_720p_10fps.mp4 \\
        --out backend/stage2_behavior/results/infer_bigbroiler091/tracks_bigbroiler091.json \\
        --max-frames 600 --thresh 0.6

Format output sama seperti tracks_fCam16.json: {"fps", "score_threshold",
"frames": [{"frame", "name", "tracks": [{"track_id", "bbox", "score"}]}]}.
bbox dalam koordinat piksel video asli.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

REPO = os.path.expanduser(
    "~/workspace/pbl-ayam-obs-rtsp/RT-DETR/rtdetrv2_pytorch")
CONFIG = os.path.join(REPO, "configs", "rtdetrv2",
                      "rtdetrv2_r18vd_120e_coco.yml")
BEST = os.path.expanduser("~/workspace/pbl-ayam-obs-rtsp/best_final.pth")

sys.path.insert(0, REPO)


def load_model():
    from src.core import YAMLConfig

    cfg = YAMLConfig(CONFIG, num_classes=1, remap_mscoco_category=True)
    model = cfg.model
    post = cfg.postprocessor
    ckpt = torch.load(BEST, map_location="cpu", weights_only=False)
    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
        state = ckpt["ema"]["module"]
        print("Pakai bobot EMA (sama seperti evaluasi AP 0,9261).")
    else:
        state = ckpt["model"]
        print("Pakai bobot model (tanpa EMA).")
    model.load_state_dict(state)
    model.eval()
    return model, post


def predict(model, post, frame_bgr, thresh):
    """Deteksi satu frame BGR. Kembalikan (boxes [N,4] piksel asli, scores)."""
    import torch as _t

    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = _t.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
    with _t.no_grad():
        outputs = model(tensor.unsqueeze(0))
    results = post(outputs, _t.tensor([[w, h]]))  # orig_size = [w, h]
    r = results[0]
    keep = r["scores"] >= thresh
    return r["boxes"][keep].tolist(), r["scores"][keep].tolist()


def save_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def save_tracker_state(path, tracker):
    """Simpan state internal ByteTrack (pickle, atomic) supaya resume
    melanjutkan tracker YANG SAMA — tanpa ini resume memulai tracker baru
    dan semua ID berubah di sambungan. Dipanggil SEBELUM save_atomic
    JSON pada checkpoint yang sama, supaya kalau proses mati di tengah,
    pickle tidak lebih baru dari JSON (cek konsistensi saat resume:
    tracker.frame_id harus == jumlah frame di JSON)."""
    import pickle

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(tracker, f)
    os.replace(tmp, path)


def load_tracker_state(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.6,
                    help="Threshold skor deteksi masuk tracker.")
    ap.add_argument("--track-buffer", type=int, default=30,
                    help="ByteTrack lost_track_buffer (frame).")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Batasi jumlah frame yang diproses (0 = semua).")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="Lewati frame sebelum ini (untuk resume manual).")
    ap.add_argument("--checkpoint-every", type=int, default=600,
                    help="Simpan checkpoint tiap N frame.")
    ap.add_argument("--resume-from", default=None,
                    help="tracks.json checkpoint untuk dilanjutkan.")
    args = ap.parse_args()

    import supervision as sv

    frames_out = []
    tracker = None
    tracker_path = args.out + ".tracker.pkl"
    if args.resume_from:
        with open(args.resume_from) as f:
            prev = json.load(f)
        frames_out = prev["frames"]
        print(f"Resume: {len(frames_out)} frame sudah ada "
              f"(terakhir frame {frames_out[-1]['frame']}).")
        if os.path.exists(tracker_path):
            tracker = load_tracker_state(tracker_path)
            if tracker.frame_id != len(frames_out):
                print(f"PERINGATAN: state tracker tidak konsisten "
                      f"(frame_id={tracker.frame_id} != "
                      f"{len(frames_out)} frame JSON) — tracker dibuang, "
                      f"mulai baru (ID bisa berubah di sambungan).")
                tracker = None
            else:
                print(f"State tracker dipulihkan (frame_id="
                      f"{tracker.frame_id}) — ID berlanjut tanpa putus.")
        else:
            print("CATATAN: tidak ada state tracker — tracker dimulai baru "
                  "dari frame berikut (ID bisa berubah di sambungan).")
        # Tracker (kalau dipulihkan) melanjutkan persis dari frame
        # berikutnya; frame yang sudah ada dilewati tanpa update.
        args.start_frame = frames_out[-1]["frame"] + 1

    print("Memuat model RT-DETRv2-S ...")
    model, post = load_model()
    if tracker is None:
        tracker = sv.ByteTrack(
            track_activation_threshold=args.thresh,
            lost_track_buffer=args.track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=10)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vw}x{vh} @ {fps:.1f} fps, {n_total} frame; "
          f"threshold {args.thresh}, track_buffer {args.track_buffer}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    frame_idx = -1
    n_processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < args.start_frame:
            continue
        if args.max_frames and n_processed >= args.max_frames:
            break
        boxes, scores = predict(model, post, frame, args.thresh)
        if boxes:
            det = sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                confidence=np.array(scores, dtype=np.float32),
                class_id=np.zeros(len(boxes), dtype=int))
        else:
            det = sv.Detections.empty()
        tracked = tracker.update_with_detections(det)
        tracks = []
        for xyxy, tid, sc in zip(tracked.xyxy, tracked.tracker_id,
                                 tracked.confidence):
            tracks.append({
                "track_id": int(tid),
                "bbox": [round(float(v), 2) for v in xyxy],
                "score": round(float(sc), 4),
            })
        frames_out.append({
            "frame": frame_idx,
            "name": f"frame_{frame_idx:04d}",
            "tracks": tracks,
        })
        n_processed += 1
        if n_processed % 100 == 0:
            el = time.time() - t0
            print(f"frame {frame_idx} ({n_processed} diproses): "
                  f"{len(tracks)} track, {el / n_processed:.2f} dtk/frame, "
                  f"elapsed {el / 60:.1f} mnt", flush=True)
        if n_processed % args.checkpoint_every == 0:
            save_tracker_state(tracker_path, tracker)
            save_atomic(args.out, {"fps": fps, "score_threshold": args.thresh,
                                   "frames": frames_out})
            print(f"  checkpoint tersimpan: {n_processed} frame -> {args.out}",
                  flush=True)

    cap.release()
    save_tracker_state(tracker_path, tracker)
    save_atomic(args.out, {"fps": fps, "score_threshold": args.thresh,
                           "frames": frames_out})
    el = time.time() - t0
    ids = {t["track_id"] for fr in frames_out for t in fr["tracks"]}
    print(f"Selesai: {n_processed} frame dalam {el / 60:.1f} menit "
          f"({el / max(n_processed, 1):.2f} dtk/frame); "
          f"{len(ids)} ID unik -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
