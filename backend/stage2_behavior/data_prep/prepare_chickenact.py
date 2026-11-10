"""Data prep ChickenAct: mapping 15 -> kelas sistem + split + laporan.

Membaca ChickenAct.zip (atau folder hasil ekstraknya), memetakan tiap klip
ke kelas sistem (lihat ``label_mapping.py`` — 3 kelas yang ada datanya:
INGESTIVE/ACTIVE/RESTING; AGONISTIC dicadangkan tanpa data; pooping
dikecualikan), membuat split train/val/test stratified per label-15
(proporsi mengikuti split resmi: train 74% / val 12% / test 14%), dan
menulis manifest + laporan distribusi.

Struktur zip yang diasumsikan (diverifikasi saat jalan): satu folder per
kelas berisi file video .mp4, mis. ``<root>/<kelas>/<klip>.mp4``. Kalau ada
satu folder pembungkus bersama (mis. semua file di bawah ``ChickenAct/``),
folder itu dilewati otomatis. Kalau ada folder kelas yang namanya tidak
dikenal pemetaan, script BERHENTI dengan pesan jelas — jangan ditebak.

Contoh pakai (di Colab, setelah unzip — lihat notebook):

    python prepare_chickenact.py --extract-dir /content/chickenact_work/ChickenAct \\
        --out-dir /content/chickenact_work/prepared

    python prepare_chickenact.py --zip /content/ChickenAct.zip \\
        --extract-dir /content/chickenact_work/ChickenAct --out-dir out/

Output (di --out-dir):
    manifest.json            — semua klip: path relatif, label15, label4,
                               label4_id, split (train/val)
    class_distribution.json  — jumlah klip per label15 & label4, per split
    clip_check.json          — hasil verifikasi contoh klip (frame/fps/resolusi)

Jalan di CPU biasa; tidak butuh GPU.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_mapping import (  # noqa: E402
    CLASS_NAMES_4,
    map_label15,
    map_label15_to_id,
    normalize_class_name,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def md5_of_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_clips(root):
    """Kembalikan {label15_asli: [path_absolut ...]} dari folder dataset.

    Kelas = komponen folder pertama di bawah root (setelah melewati satu
    folder pembungkus bersama bila ada). Hanya folder yang berisi video.
    """
    root = os.path.abspath(root)
    all_videos = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                all_videos.append(os.path.join(dirpath, fn))
    if not all_videos:
        raise FileNotFoundError(f"Tidak ada file video di {root}")
    all_videos.sort()

    # Folder pembungkus bersama: komponen path yang sama untuk semua video.
    rels = [os.path.relpath(p, root) for p in all_videos]
    first_parts = [r.split(os.sep)[0] for r in rels]
    if len(set(first_parts)) == 1 and len(rels[0].split(os.sep)) > 2:
        # Semua video di bawah satu folder yang sama DAN masih ada
        # subfolder di bawahnya -> folder itu pembungkus, geser satu level.
        root = os.path.join(root, first_parts[0])
        rels = [os.path.relpath(p, root) for p in all_videos]

    by_class = defaultdict(list)
    for p, rel in zip(all_videos, rels):
        parts = rel.split(os.sep)
        if len(parts) < 2:
            raise ValueError(f"Video tanpa folder kelas: {rel}")
        by_class[parts[0]].append(p)
    return {k: sorted(v) for k, v in sorted(by_class.items())}


def verify_clip(path):
    """Periksa satu klip dengan OpenCV: jumlah frame, fps, resolusi."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "error": "tidak bisa dibuka cv2"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"ok": n > 0, "n_frames": n, "fps": round(fps, 2),
            "width": w, "height": h,
            "duration_s": round(n / fps, 2) if fps else None}


def build_manifest(by_class, root, val_frac=0.12, test_frac=0.14, seed=42,
                   verify_n=2):
    """Bangun manifest + distribusi. Raise KeyError kalau ada kelas tak dikenal.

    Split stratified per label-15 (seed tetap -> hasil reproduktif).
    Default val 12% / test 14% mengikuti proporsi split resmi ChickenAct
    (train 11.278 / val 1.829 / test 2.143 dari 15.250 klip) — tapi split
    resmi itu sendiri tidak bisa direproduksi dari zip karena tidak ada
    manifest split di dalamnya, jadi ini split acak stratified kami
    sendiri dengan proporsi yang sama. Test = held-out, hanya untuk
    evaluasi akhir (seperti test set Stage 1).
    Kelas yang dikecualikan (map_label15 -> None, mis. pooping) TIDAK
    masuk manifest; jumlahnya dicatat di distribution["excluded"].
    """
    rng = random.Random(seed)
    manifest = []
    dist15 = Counter()
    dist4 = Counter()
    dist15_split = defaultdict(Counter)
    dist4_split = defaultdict(Counter)
    excluded = Counter()
    clip_check = {}

    for label15_raw, paths in by_class.items():
        label4 = map_label15(label15_raw)  # KeyError kalau tak dikenal
        if label4 is None:
            excluded[label15_raw] += len(paths)
            clip_check[label15_raw] = {
                "n_clips": len(paths), "label4": None,
                "excluded_reason": "peristiwa fisiologis sesaat, "
                                   "bukan state perilaku (lihat label_mapping)",
                "checked": [],
            }
            continue
        label4_id = CLASS_NAMES_4.index(label4)
        dist15[label15_raw] += len(paths)
        dist4[label4] += len(paths)

        # Verifikasi contoh klip per kelas (baca header video saja).
        sample = paths[:verify_n] if verify_n > 0 else []
        clip_check[label15_raw] = {
            "n_clips": len(paths),
            "label4": label4,
            "checked": [dict(path=os.path.basename(p), **verify_clip(p))
                        for p in sample],
        }

        # Split stratified per label-15 (seed tetap -> hasil reproduktif).
        idx = list(range(len(paths)))
        rng.shuffle(idx)
        n_val = max(1, round(len(paths) * val_frac)) if len(paths) > 1 else 0
        n_test = max(1, round(len(paths) * test_frac)) if len(paths) > 1 else 0
        val_idx = set(idx[:n_val])
        test_idx = set(idx[n_val:n_val + n_test])
        for i, p in enumerate(paths):
            split = ("val" if i in val_idx else
                     "test" if i in test_idx else "train")
            dist15_split[split][label15_raw] += 1
            dist4_split[split][label4] += 1
            manifest.append({
                "path": os.path.relpath(p, root),
                "label15": normalize_class_name(label15_raw),
                "label15_raw": label15_raw,
                "label4": label4,
                "label4_id": label4_id,
                "split": split,
            })

    manifest.sort(key=lambda r: (r["split"], r["label4"], r["path"]))
    distribution = {
        "total_clips": len(manifest),
        "excluded": dict(excluded),
        "n_classes_15": len(dist15),
        "n_classes_4": len(dist4),
        "classes_4_present": sorted(dist4),
        "per_label15": dict(sorted(dist15.items())),
        "per_label4": {k: dist4[k] for k in CLASS_NAMES_4 if dist4[k]},
        "per_label15_train": dict(sorted(dist15_split["train"].items())),
        "per_label15_val": dict(sorted(dist15_split["val"].items())),
        "per_label15_test": dict(sorted(dist15_split["test"].items())),
        "per_label4_train": {k: dist4_split["train"][k] for k in CLASS_NAMES_4},
        "per_label4_val": {k: dist4_split["val"][k] for k in CLASS_NAMES_4},
        "per_label4_test": {k: dist4_split["test"][k] for k in CLASS_NAMES_4},
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
        "note": "Split acak stratified kami sendiri (seed tetap). Split "
                "resmi ChickenAct (train 11278/val 1829/test 2143) tidak "
                "bisa direproduksi dari zip karena tidak ada manifest "
                "split di dalamnya.",
    }
    return manifest, distribution, clip_check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=None,
                    help="Path ChickenAct.zip (akan diekstrak bila --extract-dir kosong).")
    ap.add_argument("--extract-dir", default=None,
                    help="Folder hasil ekstrak / folder dataset yang sudah ada.")
    ap.add_argument("--out-dir", default="prepared",
                    help="Folder output manifest + laporan.")
    ap.add_argument("--val-frac", type=float, default=0.12,
                    help="Fraksi val (default 0.12, mengikuti split resmi).")
    ap.add_argument("--test-frac", type=float, default=0.14,
                    help="Fraksi test held-out (default 0.14).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify-n", type=int, default=2,
                    help="Jumlah klip per kelas yang diperiksa header videonya (0 = lewati).")
    ap.add_argument("--expected-md5", default=None,
                    help="Md5 zip yang diharapkan (dicek sebelum ekstrak).")
    args = ap.parse_args()

    if args.zip and args.expected_md5:
        got = md5_of_file(args.zip)
        print(f"md5 zip: {got}")
        assert got == args.expected_md5, (
            f"md5 tidak cocok: {got} != {args.expected_md5}")
        print("md5 cocok.")

    if args.extract_dir and os.path.isdir(args.extract_dir) and any(
            os.scandir(args.extract_dir)):
        root = args.extract_dir
        print(f"Pakai folder dataset yang sudah ada: {root}")
    elif args.zip:
        root = args.extract_dir or os.path.splitext(args.zip)[0]
        print(f"Ekstrak {args.zip} -> {root} ...")
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(root)
        print("Ekstrak selesai.")
    else:
        ap.error("Butuh --zip atau --extract-dir yang berisi dataset.")

    print("Memindai klip ...")
    by_class = find_clips(root)
    print(f"Ditemukan {len(by_class)} folder kelas:")
    for k, v in by_class.items():
        print(f"  {k}: {len(v)} klip")

    manifest, distribution, clip_check = build_manifest(
        by_class, root, val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed, verify_n=args.verify_n)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, "class_distribution.json"), "w") as f:
        json.dump(distribution, f, indent=2)
    with open(os.path.join(args.out_dir, "clip_check.json"), "w") as f:
        json.dump(clip_check, f, indent=2)

    print(f"\nTotal klip: {distribution['total_clips']}")
    if distribution["excluded"]:
        print(f"Dikecualikan dari training: {distribution['excluded']} "
              f"(lihat label_mapping.EXCLUDED_CLASSES)")
    print("Distribusi 15 kelas -> 4 kelas:")
    for l4 in CLASS_NAMES_4:
        if l4 not in distribution["per_label4"]:
            print(f"  {l4}: 0 klip — TIDAK ADA DATA LATIH "
                  f"(kelas dicadangkan, lihat label_mapping)")
            continue
        members = {k: v for k, v in distribution["per_label15"].items()
                   if map_label15(k) == l4}
        print(f"  {l4}: {distribution['per_label4'][l4]} klip "
              f"({', '.join(f'{k}={v}' for k, v in members.items())})")
    n_train = sum(1 for r in manifest if r["split"] == "train")
    n_val = sum(1 for r in manifest if r["split"] == "val")
    n_test = sum(1 for r in manifest if r["split"] == "test")
    print(f"Split: train {n_train} / val {n_val} / test {n_test} "
          f"(val_frac={args.val_frac}, test_frac={args.test_frac}, "
          f"seed={args.seed})")
    print(f"Manifest + laporan tersimpan di {args.out_dir}/")

    bad = [c for checks in clip_check.values() for c in checks["checked"]
           if not c.get("ok")]
    if bad:
        print(f"PERINGATAN: {len(bad)} klip contoh gagal dibaca: {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
