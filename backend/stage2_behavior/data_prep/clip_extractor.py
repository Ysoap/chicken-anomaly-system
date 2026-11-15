"""Ekstrak 16 frame dari satu file video (klip ChickenAct / klip uji).

Fungsi di sini adalah versi "satu klip" dari ekstraksi frame:
``extract_frames.py`` (di folder ini) melakukan hal yang sama secara
massal + multiprocessing untuk seluruh dataset training, sedangkan
``clip_builder.py`` (satu level di atas) membangun klip dari tracklet
hasil tracking Stage 1 — keduanya memakai sampling merata yang sama
(``videomae_utils.sample_frames_uniform``).

Dipakai oleh ``inference.py`` mode ``--clip``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from videomae_utils import NUM_FRAMES, sample_frames_uniform  # noqa: E402


def extract_clip_frames(video_path, n_frames=NUM_FRAMES):
    """Baca video, kembalikan (frames RGB, n_total).

    ``frames``: list sepanjang ``n_frames``, tiap frame array HWC uint8
    RGB, diambil merata dari seluruh durasi klip. Kalau klip lebih
    pendek dari ``n_frames``, frame terakhir diulang (padding).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Video kosong: {video_path}")
    idx = sample_frames_uniform(len(frames), n_frames)
    return [frames[i] for i in idx], len(frames)
