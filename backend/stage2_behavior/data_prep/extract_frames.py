"""Ekstrak 16 frame per klip ChickenAct -> JPEG (untuk training cepat).

Membaca ``manifest.json`` hasil ``prepare_chickenact.py`` (klip mp4 +
label), lalu untuk tiap klip mengambil 16 frame merata dan menyimpannya
sebagai JPEG di ``--out-dir/<label4>/<clip_id>/frame_00.jpg`` ...
``frame_15.jpg``. Output: ``frames_manifest.json`` (dipakai ``train.py``).

Kenapa tidak decode mp4 tiap epoch? Decode 15 ribu klip tiap epoch
lambat; ekstrak sekali (±15-30 menit dengan multiprocessing), setelah
itu tiap epoch hanya membaca JPEG — jauh lebih cepat dan deterministik
(frame yang dipakai selalu sama).

Contoh pakai (di Colab, setelah prepare):

    python extract_frames.py --manifest prepared/manifest.json \\
        --data-root /content/chickenact_work/ChickenAct \\
        --out-dir /content/chickenact_work/frames --workers 4

Klip yang frame-nya kurang dari 16 akan di-padding dengan frame terakhir
(``videomae_utils.sample_frames_uniform``) dan dicatat di
``frames_manifest.json`` (field ``padded: true``).
"""

import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from videomae_utils import NUM_FRAMES, sample_frames_uniform  # noqa: E402


def extract_one(entry, data_root, out_dir):
    """Ekstrak 1 klip. Kembalikan dict manifest frame atau None bila gagal."""
    import cv2

    src = os.path.join(data_root, entry["path"])
    clip_id = os.path.splitext(os.path.basename(entry["path"]))[0]
    dest_dir = os.path.join(out_dir, entry["label4"],
                            f"{entry['label15_raw']}_{clip_id}")
    frame_files = [os.path.join(os.path.relpath(dest_dir, out_dir),
                                f"frame_{i:02d}.jpg")
                   for i in range(NUM_FRAMES)]
    # Lewati kalau sudah lengkap (resume-safe).
    if all(os.path.exists(os.path.join(out_dir, f)) for f in frame_files):
        return {"clip": entry["path"], "label15": entry["label15"],
                "label15_raw": entry["label15_raw"],
                "label4": entry["label4"], "label4_id": entry["label4_id"],
                "split": entry["split"], "frame_files": frame_files,
                "padded": False, "n_frames_total": -1}

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return {"clip": entry["path"], "error": "tidak bisa dibuka"}
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return {"clip": entry["path"], "error": "0 frame"}
    n_total = len(frames)
    idx = sample_frames_uniform(n_total, NUM_FRAMES)
    os.makedirs(dest_dir, exist_ok=True)
    for k, i in enumerate(idx):
        cv2.imwrite(os.path.join(dest_dir, f"frame_{k:02d}.jpg"),
                    frames[i], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"clip": entry["path"], "label15": entry["label15"],
            "label15_raw": entry["label15_raw"],
            "label4": entry["label4"], "label4_id": entry["label4_id"],
            "split": entry["split"], "frame_files": frame_files,
            "padded": n_total < NUM_FRAMES, "n_frames_total": n_total}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="manifest.json hasil prepare_chickenact.py.")
    ap.add_argument("--data-root", required=True,
                    help="Folder root klip mp4 (path relatif di manifest).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="Batasi jumlah klip (0 = semua). Untuk smoke test.")
    args = ap.parse_args()

    with open(args.manifest) as f:
        entries = json.load(f)
    if args.limit:
        entries = entries[:args.limit]
    print(f"Klip diproses: {len(entries)} (workers={args.workers})")

    os.makedirs(args.out_dir, exist_ok=True)
    fn = partial(extract_one, data_root=args.data_root, out_dir=args.out_dir)
    results = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(fn, entries, chunksize=20)):
                results.append(r)
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{len(entries)} klip ...")
    else:
        for i, e in enumerate(entries):
            results.append(fn(e))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(entries)} klip ...")

    failed = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    padded = sum(1 for r in ok if r.get("padded"))
    print(f"Selesai: {len(ok)} berhasil ({padded} ter-padding karena "
          f"<{NUM_FRAMES} frame), {len(failed)} gagal.")
    if failed:
        print("Contoh gagal:", failed[:5])
    with open(os.path.join(args.out_dir, "frames_manifest.json"), "w") as f:
        json.dump(ok, f, indent=2)
    with open(os.path.join(args.out_dir, "extract_failures.json"), "w") as f:
        json.dump(failed, f, indent=2)
    print(f"Manifest frame tersimpan: {args.out_dir}/frames_manifest.json")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
