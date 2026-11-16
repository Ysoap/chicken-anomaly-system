"""Bangun klip latih fine-tune Big Broiler dari kotak GT (bukan prediksi).

Dua pass per video (hemat RAM — frame tidak pernah disimpan di memori
lebih dari 16 frame):
- Pass 1: baca video sekuensial, hitung label tiap jendela 16 frame
  @10fps, stride 16 (tanpa overlap — tidak ada kebocoran train/val).
- Pass 2: baca ulang video, tulis JPEG hanya untuk klip yang
  dipertahankan setelah split + balancing.

Crop persegi mengikuti kotak GT tiap frame (margin 1.3, sama seperti
clip_builder.py — supaya distribusi latih sama dengan klip inference),
resize 224x224, JPEG kualitas 90 (sama seperti extract_frames.py).

Label klip = suara mayoritas aksi GT dalam 16 frame (pemetaan
compare_gt.ACTION_MAP + 5 perluasan untuk aksi yang hanya ada di
080/075 — lihat ACTION_MAP_EXT; 091 tidak berubah). Aturan STRICT:
ke-16 frame harus bersuara (tanpa Out_Of_Sight / tanpa anotasi);
seri suara mayoritas -> klip dibuang.

Split (seed 42):
- 091 paruh-1 (start < 3000) + 080 semua + 067/057/060/064 semua
  -> pool train/val; val = 10% stratified per kelas; sisanya train.
- 091 paruh-2 (start >= 3000) -> split "test" (uji utama, GT rapat).
- 075 semua -> split "test075" (uji generalisasi video tak terlihat;
  manifest terpisah frames_manifest_test075.json, split "test").
  075 dan test utama TIDAK BERUBAH saat perluasan 16 Nov 2026.
- Balancing TRAIN saja: RESTING di-downsample ke jumlah INGESTIVE
  train (ACTIVE dibiarkan apa adanya — sudah dikenali baik, recall
  95% di test-set ChickenAct). Val/test TIDAK di-downsample (metrik
  jujur). Klip RESTING train yang dibuang TIDAK ditulis ke disk dan
  TIDAK masuk manifest (dicatat di dataset_info.json).

Output (out-dir):
- INGESTIVE|ACTIVE|RESTING/<video>_<track>_<start>/frame_XX.jpg
- frames_manifest.json (train/val/test; dipakai train.py)
- frames_manifest_test075.json (075 saja, split "test"; dipakai
  eval_test_set.py)
- dataset_info.json (hitungan per video/kelas/split, pemetaan, aturan,
  md5 GT + video)
- contact_sheets/ (contoh visual per kelas, untuk cek crop)

Contoh pakai:
    python data_prep/build_bigbroiler_clips.py \\
        --out-dir ~/workspace/pbl-ayam-obs-rtsp/bigbroiler_finetune/frames
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_gt import ACTION_MAP  # noqa: E402
from clip_builder import square_crop  # noqa: E402
from videomae_utils import CLASS_NAMES_4, IMAGE_SIZE, NUM_FRAMES  # noqa: E402

# Perluasan pemetaan HANYA untuk aksi yang tidak ada di CSV 091 (angka
# 091 tidak berubah satu pun). Alasan per aksi mengikuti logika taksonomi
# yang sama (data_prep/label_mapping.py ChickenAct):
# - Sleeping -> RESTING (tidur = istirahat)
# - Flapping -> ACTIVE (cf. wing_flapping -> ACTIVE)
# - Head_Shaking -> RESTING (cf. body_shaking -> RESTING, comfort)
# - Short_Leg_Move -> ACTIVE (langkah kecil = lokomosi, cf. walking)
# - Swallowing -> INGESTIVE (menelan = rangkaian makan/minum)
# Tanpa suara (bukan state perilaku / tidak diketahui):
# Sneezing, Defecating (fisiologis sesaat, cf. pooping dikecualikan),
# Video_Problem, Other.
ACTION_MAP_EXT = dict(ACTION_MAP)
ACTION_MAP_EXT.update({
    "Sleeping": "RESTING",
    "Flapping": "ACTIVE",
    "Head_Shaking": "RESTING",
    "Short_Leg_Move": "ACTIVE",
    "Swallowing": "INGESTIVE",
})

CLASSES = ["INGESTIVE", "ACTIVE", "RESTING"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES_4) if n in CLASSES}

VIDEOS = [
    # (id, video 720p 10fps, GT CSV, n_frames_10fps)
    ("091", "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_720p_10fps.mp4",
     "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_3_32_1920_008_0_10.csv", 6000),
    ("080", "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_080_720p_10fps.mp4",
     "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_080_2_40_2688_024_0_10.csv", 6008),
    ("075", "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_075_720p_10fps.mp4",
     "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_075_2_38_1920_026_0_10.csv", 6066),
    # Ditambah 16 Nov 2026 (perluasan INGESTIVE): 4 video setup-2 sudut
    # miring dengan klip INGESTIVE terbanyak + GT rapat (100% coverage).
    # Ranking dari bb_rank.py (aturan jendela sama persis): 067 ING 784,
    # 057 ING 611, 060 ING 579, 064 ING 501. Semua masuk pool train/val.
    ("067", "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_067_720p_10fps.mp4",
     "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_067_2_39_2688_061_0_10.csv", 6008),
    ("057", "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_057_720p_10fps.mp4",
     "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_057_2_29_2688_041_0_10.csv", 6005),
    ("060", "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_060_720p_10fps.mp4",
     "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_060_2_29_2688_029_0_10.csv", 6008),
    ("064", "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_064_720p_10fps.mp4",
     "/home/hatch/workspace/pbl-ayam-obs-rtsp/topview_test/oblique_bigbroiler_064_2_34_2688_035_0_10.csv", 6001),
]

STRIDE = 16
CLIP_LEN = NUM_FRAMES  # 16
MARGIN = 1.3
SEED = 42
VAL_FRAC = 0.10
TEST_SPLIT_FRAME_091 = 3000  # paruh-2 091 = test utama


def md5_of_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_gt(path):
    """{track: {frame_gt: (box_norm, action)}}."""
    per_track = defaultdict(dict)
    for row in csv.DictReader(open(path)):
        per_track[int(row["track"])][int(row["frame"])] = (
            tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")),
            row["actions_ground"])
    return per_track


def window_label(by_fr, start):
    """Label mayoritas jendela [start, start+16) atau None (tak layak).

    Mengembalikan (label4, label15_raw, votes, boxes) atau None bila:
    ada frame tanpa suara valid (STRICT), atau seri suara mayoritas.
    boxes = {frame10: box_norm} untuk crop."""
    votes = Counter()
    vote_raw = Counter()
    boxes = {}
    for f in range(start, start + CLIP_LEN):
        ent = by_fr.get(int(round(f * 2.5)))
        if ent is None:
            return None
        box, act = ent
        cls = ACTION_MAP_EXT.get(act)
        if cls is None:
            return None
        votes[cls] += 1
        vote_raw[act] += 1
        boxes[f] = box
    if sum(votes.values()) < CLIP_LEN:
        return None
    top = votes.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0], vote_raw.most_common(1)[0][0], dict(votes), boxes


def scan_video(video_id, video_path, gt_path, n_frames):
    """Pass 1: hitung kandidat klip (label saja, tanpa simpan frame)."""
    per_track = load_gt(gt_path)
    print(f"[{video_id}] GT: {len(per_track)} track, video {n_frames} frame")
    candidates = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {video_path}")
    f = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        if (f + 1) % STRIDE == 0:
            start = f - CLIP_LEN + 1
            for tid in sorted(per_track):
                res = window_label(per_track[tid], start)
                if res is None:
                    continue
                label4, label15_raw, votes, boxes = res
                candidates.append({
                    "clip_id": f"bb{video_id}_track{tid}_f{start:04d}",
                    "video": video_id, "track": tid,
                    "start_frame": start, "label4": label4,
                    "label15": label15_raw, "label15_raw": label15_raw,
                    "votes": votes, "boxes": boxes,
                })
        f += 1
        if f >= n_frames:
            break
    cap.release()
    print(f"[{video_id}] kandidat klip: {len(candidates)} "
          f"({dict(Counter(c['label4'] for c in candidates))})")
    return candidates


def write_kept_frames(video_id, video_path, n_frames, keep_by_start, out_dir):
    """Pass 2: baca ulang video, tulis JPEG hanya untuk klip keep.

    keep_by_start: {start_frame: {track: kandidat}}.
    Mengembalikan {clip_id: frame_files}."""
    written = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {video_path}")
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = {}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf[f] = frame
        for old in [k for k in buf if k < f - CLIP_LEN + 1]:
            del buf[old]
        if (f + 1) % STRIDE == 0:
            start = f - CLIP_LEN + 1
            for tid, c in keep_by_start.get(start, {}).items():
                dest_dir = os.path.join(out_dir, c["label4"], c["clip_id"])
                os.makedirs(dest_dir, exist_ok=True)
                frame_files = []
                for k, ff in enumerate(range(start, start + CLIP_LEN)):
                    x1, y1, x2, y2 = c["boxes"][ff]
                    box_px = (x1 * vw, y1 * vh, x2 * vw, y2 * vh)
                    crop = square_crop(buf[ff], box_px, margin=MARGIN,
                                       size=IMAGE_SIZE)
                    rel = os.path.join(c["label4"], c["clip_id"],
                                       f"frame_{k:02d}.jpg")
                    cv2.imwrite(os.path.join(out_dir, rel),
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_files.append(rel)
                written[c["clip_id"]] = frame_files
        f += 1
        if f >= n_frames:
            break
    cap.release()
    print(f"[{video_id}] frame ditulis: {len(written)} klip")
    return written


def contact_sheet(frames_rgb, thumb=112):
    import numpy as np
    thumbs = [cv2.resize(f, (thumb, thumb), interpolation=cv2.INTER_LINEAR)
              for f in frames_rgb]
    rows = [np.hstack(thumbs[r * 4:(r + 1) * 4]) for r in range(4)]
    return cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Pass 1: kandidat klip ---
    all_candidates = []
    gt_info = {}
    for video_id, video_path, gt_path, n_frames in VIDEOS:
        cands = scan_video(video_id, video_path, gt_path, n_frames)
        all_candidates.extend(cands)
        gt_info[video_id] = {
            "video": os.path.basename(video_path),
            "gt_csv": os.path.basename(gt_path),
            "gt_md5": md5_of_file(gt_path),
            "video_md5": md5_of_file(video_path),
            "n_frames_10fps": n_frames,
            "n_candidates": len(cands),
            "per_class": dict(Counter(c["label4"] for c in cands)),
        }

    # --- Split ---
    pool, test_main, test075 = [], [], []
    for c in all_candidates:
        if c["video"] == "075":
            c["split"] = "test075"
            test075.append(c)
        elif c["video"] == "091" and c["start_frame"] >= TEST_SPLIT_FRAME_091:
            c["split"] = "test"
            test_main.append(c)
        else:
            pool.append(c)  # 091 paruh-1 + 080 semua
    by_class = defaultdict(list)
    for c in pool:
        by_class[c["label4"]].append(c)
    val_ids = set()
    for cls, items in by_class.items():
        items_sorted = sorted(items, key=lambda c: c["clip_id"])
        rng.shuffle(items_sorted)
        n_val = max(1, round(len(items_sorted) * VAL_FRAC))
        for c in items_sorted[:n_val]:
            c["split"] = "val"
            val_ids.add(c["clip_id"])
    train = [c for c in pool if c["clip_id"] not in val_ids]
    for c in train:
        c["split"] = "train"

    # --- Balancing TRAIN: RESTING -> jumlah INGESTIVE train ---
    train_by_class = defaultdict(list)
    for c in train:
        train_by_class[c["label4"]].append(c)
    n_ing_train = len(train_by_class["INGESTIVE"])
    rest_train = sorted(train_by_class["RESTING"], key=lambda c: c["clip_id"])
    rng.shuffle(rest_train)
    keep_rest_ids = {c["clip_id"] for c in rest_train[:n_ing_train]}
    drop_rest = rest_train[n_ing_train:]
    kept = [c for c in all_candidates
            if c["split"] != "train" or c["label4"] != "RESTING"
            or c["clip_id"] in keep_rest_ids]
    keep_ids = {c["clip_id"] for c in kept}
    dropped = [c for c in all_candidates if c["clip_id"] not in keep_ids]

    print("\nSplit (sebelum balancing train):")
    for name, grp in (("pool train", train),
                      ("val", [c for c in pool if c["split"] == "val"]),
                      ("test 091 paruh-2", test_main),
                      ("test075", test075)):
        print(f"  {name:20s} total {len(grp):5d} "
              f"({dict(Counter(c['label4'] for c in grp))})")
    print(f"Balancing train: RESTING {len(train_by_class['RESTING'])} -> "
          f"{n_ing_train} (dibuang {len(drop_rest)} klip, tidak ditulis)")
    print(f"Train akhir: {sum(1 for c in kept if c['split'] == 'train')} klip "
          f"({dict(Counter(c['label4'] for c in kept if c['split'] == 'train'))})")

    # --- Pass 2: tulis frame hanya untuk klip yang dipertahankan ---
    keep_by_video_start = defaultdict(lambda: defaultdict(dict))
    for c in kept:
        keep_by_video_start[c["video"]][c["start_frame"]][c["track"]] = c
    written = {}
    for video_id, video_path, _gt, n_frames in VIDEOS:
        written.update(write_kept_frames(
            video_id, video_path, n_frames,
            keep_by_video_start.get(video_id, {}), out_dir))

    manifest = []
    for c in sorted(kept, key=lambda x: (x["split"], x["clip_id"])):
        manifest.append({
            "clip": c["clip_id"], "video": c["video"], "track": c["track"],
            "start_frame": c["start_frame"], "label15": c["label15"],
            "label15_raw": c["label15_raw"], "label4": c["label4"],
            "label4_id": CLASS_TO_ID[c["label4"]], "split": c["split"],
            "frame_files": written[c["clip_id"]], "padded": False,
            "n_frames_total": CLIP_LEN, "gt_votes": c["votes"],
        })
    main_manifest = [m for m in manifest if m["split"] in ("train", "val", "test")]
    with open(os.path.join(out_dir, "frames_manifest.json"), "w") as f:
        json.dump(main_manifest, f, indent=2)
    m075 = []
    for m in manifest:
        if m["split"] == "test075":
            m2 = dict(m)
            m2["split"] = "test"
            m075.append(m2)
    with open(os.path.join(out_dir, "frames_manifest_test075.json"), "w") as f:
        json.dump(m075, f, indent=2)

    # Contact sheets (2 per kelas, baca ulang JPEG yang ditulis).
    sheet_dir = os.path.join(out_dir, "contact_sheets")
    os.makedirs(sheet_dir, exist_ok=True)
    seen_sheet = Counter()
    for m in sorted(manifest, key=lambda x: (x["label4"], x["video"], x["clip"])):
        if seen_sheet[m["label4"]] >= 2:
            continue
        seen_sheet[m["label4"]] += 1
        imgs = []
        for rel in m["frame_files"]:
            img = cv2.imread(os.path.join(out_dir, rel))
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(sheet_dir, f"{m['label4']}_{m['clip']}.png"),
                    contact_sheet(imgs))
        print(f"  sheet: {m['label4']}_{m['clip']}.png "
              f"(video {m['video']}, votes {m['gt_votes']})")

    info = {
        "dataset": "BigBroiler-Finetune (091 + 080 + 075 + 067 + 057 + 060 + 064)",
        "seed": SEED, "stride": STRIDE, "clip_len": CLIP_LEN, "margin": MARGIN,
        "frame_mapping": "frame_gt = round(frame_10fps * 2.5)",
        "label_rule": "suara mayoritas aksi GT dalam 16 frame; STRICT: ke-16 frame harus bersuara; seri dibuang",
        "mapping_base": "compare_gt.ACTION_MAP persis (Explor_Forag -> ACTIVE)",
        "mapping_extension": {
            "Sleeping": "RESTING", "Flapping": "ACTIVE",
            "Head_Shaking": "RESTING", "Short_Leg_Move": "ACTIVE",
            "Swallowing": "INGESTIVE",
            "tanpa_suara": ["Sneezing", "Defecating", "Video_Problem", "Other"],
            "alasan": "hanya untuk aksi yang tidak ada di CSV 091; mengikuti logika label_mapping.py ChickenAct; angka 091 tidak berubah",
        },
        "split_rules": {
            "train_val_pool": "091 paruh-1 (start<3000) + 080 + 067 + 057 + 060 + 064 semua",
            "val": "10% stratified per kelas dari pool (seed 42)",
            "test": "091 paruh-2 (start>=3000) — TIDAK BERUBAH dari v1",
            "test075": "075 seluruhnya (video tak terlihat); manifest terpisah — TIDAK BERUBAH dari v1",
            "perluasan_v2": "16 Nov 2026: tambah 067/057/060/064 (ranking bb_rank.py: ING 784/611/579/501, GT coverage 100%) khusus untuk menaikkan INGESTIVE train; video baru masuk pool train/val saja",
        },
        "imbalance": {
            "cara": "downsample RESTING train ke jumlah INGESTIVE train (seed 42)",
            "alasan": "paling gampang + andal; train.py tidak dukung class-weighted loss tanpa ubah kode; ACTIVE dibiarkan (recall 95% di test-set ChickenAct, sudah dikenali baik); val/test tidak di-downsample",
            "train_sebelum": {k: len(v) for k, v in train_by_class.items()},
            "train_sesudah": dict(Counter(c["label4"] for c in kept if c["split"] == "train")),
            "dropped_resting_train_clips": len(drop_rest),
        },
        "counts": {
            "per_video_per_class": {v: gt_info[v]["per_class"] for v in gt_info},
            "per_split_per_class": {
                s: dict(Counter(m["label4"] for m in manifest if m["split"] == s))
                for s in ("train", "val", "test", "test075")},
            "total_manifest": len(main_manifest),
            "total_test075_manifest": len(m075),
        },
        "videos": gt_info,
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSelesai. Manifest utama: {len(main_manifest)} klip, "
          f"test075: {len(m075)} klip.")
    print(f"Info: {os.path.join(out_dir, 'dataset_info.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
