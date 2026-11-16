"""Bandingkan prediksi perilaku pipeline (Big Broiler 091) vs ground truth.

Metode
------
1. Pemetaan aksi GT (23 perilaku -> 3 kelas kita), diverifikasi langsung
   dari CSV (94.973 baris):
     INGESTIVE: Eat (24.467) + Drink (5.750) = 30.217 (31,8%)
     ACTIVE:    Locomotion (3.249) + Explor_Forag (3.183)
                + Stretching (576) + Feather_Pecking (771) = 7.779 (8,2%)
     RESTING:   Inactive (43.321) + Resting (3.877) + Preening (4.020)
                + Panting (687) + Lying (2.792, aksi kosong '') = 54.697 (57,6%)
     ABAIKAN:   Out_Of_Sight (2.280; 2,4%) — kotaknya posisi basi.
   (Angka ACTIVE/RESTING di sini 8,2%/57,6%; laporan lama menulis
   8,5%/57,3% — selisih kecil, angka di sini yang dihitung ulang
   dari CSV.)
2. Penyelarasan frame DIVERIFIKASI dulu (bukan ditebak): GT 25 fps
   (15.000 frame) vs pipeline 10 fps (6.000 frame). Pemetaan nominal
   frame_gt = round(frame_10fps * 2,5). Verifikasi: (a) jarak pusat
   kotak prediksi vs kotak GT pasangannya, HANYA pada frame saat ayam
   GT sedang bergerak (aksi Locomotion atau pusat GT bergeser
   >15 px/frame) — ayam yang diam tidak bisa membedakan offset;
   (b) IoU kotak sebagai bukti pendukung. Pemetaan diterima bila
   offset 0 berada di lembah datar (±3 frame GT = ±120 mdtk, di
   bawah noise gerak) SEMENTARA offset jauh (±25 frame = ±1 dtk,
   ±250 frame = ±10 dtk) jelas jauh lebih buruk. Pada video ini:
   jarak pusat median saat bergerak 8,9 px di offset 0 vs 34-36 px
   di ±25 dan 81-129 px di ±250 — terverifikasi.
3. Pencocokan lewat KOTAK (IoU), bukan ID track (ID prediksi 17,
   ID GT 8 — beda). Tiap klip 16 frame @10 fps: untuk tiap frame,
   kotak prediksi track dicocokkan ke kotak GT dengan IoU tertinggi;
   bila IoU >= 0,3 dan aksi GT-nya bukan Out_Of_Sight, frame itu
   menyumbang satu suara kelas GT. Label GT klip = suara mayoritas
   (seri: label frame tengah klip; jumlah seri dilaporkan). Klip
   tanpa satu pun suara GT = tidak tercocokkan (dilaporkan terpisah).
4. Metrik dihitung HANYA pada klip yang tercocokkan: akurasi
   keseluruhan, recall/precision per kelas, confusion matrix 3x3
   (baris = GT, kolom = prediksi), coverage, dan sorotan
   INGESTIVE<->RESTING.

Jalankan:
    python compare_gt.py \
        --tracks results/infer_bigbroiler091/merged_tracks_bigbroiler091.filtered.json \
        --predictions results/infer_bigbroiler091/track_predictions_filtered.json \
        --gt ~/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_3_32_1920_008_0_10.csv \
        --out results/infer_bigbroiler091/gt_comparison.json
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict

CLASSES = ["INGESTIVE", "ACTIVE", "RESTING"]

ACTION_MAP = {
    "Eat": "INGESTIVE", "Drink": "INGESTIVE",
    "Locomotion": "ACTIVE", "Explor_Forag": "ACTIVE",
    "Stretching": "ACTIVE", "Feather_Pecking": "ACTIVE",
    "Inactive": "RESTING", "Resting": "RESTING", "Preening": "RESTING",
    "Panting": "RESTING", "": "RESTING",  # aksi kosong = postur Lying
    # "Out_Of_Sight" -> None (diabaikan)
}

# Sensitivitas pemetaan: "Explor_Forag" (pecah-mencari di litter) di
# taksonomi ChickenAct lebih dekat ke INGESTIVE (litter_pecking /
# litter_scratching masuk INGESTIVE) ketimbang ACTIVE. Pemetaan UTAMA
# tetap yang di atas (sesuai brief); --map-explor-forag ingestive hanya
# untuk uji sensitivitas, hasilnya dilaporkan terpisah, bukan pengganti.
EXPLOR_FORAG_ALT = "INGESTIVE"

IOU_THRESH = 0.3


def iou(a, b):
    """IoU dua bbox (x1, y1, x2, y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw, ih = x2 - x1, y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_gt(path, video_w=1280.0, video_h=720.0, explor_forag="ACTIVE"):
    """GT: {frame_gt: [(x1,y1,x2,y2 px, kelas3 atau None)]}.

    Mengembalikan juga {frame_gt: {gtid: (cx, cy, aksi_mentah)}} untuk
    verifikasi penyelarasan (jarak pusat + aksi gerak)."""
    action_map = dict(ACTION_MAP)
    action_map["Explor_Forag"] = explor_forag
    gt = defaultdict(list)
    centers = defaultdict(dict)
    n_rows = 0
    for row in csv.DictReader(open(path)):
        n_rows += 1
        fr = int(row["frame"])
        x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        box = (x1 * video_w, y1 * video_h, x2 * video_w, y2 * video_h)
        cls = action_map.get(row["actions_ground"])
        gt[fr].append((box, cls))
        centers[fr][int(row["track"])] = (
            (x1 + x2) / 2 * video_w, (y1 + y2) / 2 * video_h,
            row["actions_ground"])
    return gt, centers, n_rows


def load_pred_tracks(path):
    """Tracks terfilter: {tid: {frame10: (x1,y1,x2,y2)}}."""
    with open(path) as f:
        data = json.load(f)
    per_track = defaultdict(dict)
    for fr in data["frames"]:
        for t in fr.get("tracks", []):
            per_track[int(t["track_id"])][int(fr["frame"])] = tuple(
                float(v) for v in t["bbox"])
    return per_track


def pair_tracks(per_track, gt_centers, video_fps=10.0, gt_fps=25.0,
                dist_thresh=60.0, min_votes=50):
    """Pasangkan tiap ID prediksi ke track GT via voting pusat terdekat
    (offset 0). Kembalikan {tid: gtid} untuk pasangan dengan suara >= 50."""
    ratio = gt_fps / video_fps
    pairs = {}
    for tid, by_frame in per_track.items():
        votes = Counter()
        for f, b in by_frame.items():
            of = int(round(f * ratio))
            if of not in gt_centers:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best, bd = None, 1e18
            for gid, (gx, gy, _a) in gt_centers[of].items():
                dd = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                if dd < bd:
                    bd, best = dd, gid
            if bd < dist_thresh:
                votes[best] += 1
        if votes:
            g, v = votes.most_common(1)[0]
            if v >= min_votes:
                pairs[tid] = g
    return pairs


def verify_alignment(per_track, gt, gt_centers, video_fps=10.0, gt_fps=25.0):
    """Verifikasi penyelarasan frame dengan dua bukti.

    (a) Jarak pusat (utama): untuk pasangan track prediksi-GT, jarak
        pusat median pada frame saat ayam GT sedang bergerak (aksi
        Locomotion atau pusat GT bergeser >15 px/frame) — ayam yang
        diam tidak membedakan offset, ayam yang bergerak membedakan
        tajam (offset ±25 frame = ±1 dtk sudah 4x lebih jauh).
    (b) IoU kotak (pendukung): median IoU prediksi vs kotak GT mana
        pun pada frame sampel.
    Mengembalikan (offset_dipakai, tabel_jarak, tabel_iou, terverifikasi).
    Offset dipakai selalu 0 (pemetaan nominal); ±3 frame GT (±120 mdtk)
    berada di bawah noise gerak sehingga tidak dibedakan."""
    ratio = gt_fps / video_fps
    offsets = [-250, -25, -3, -2, -1, 0, 1, 2, 3, 25, 250]
    pairs = pair_tracks(per_track, gt_centers, video_fps, gt_fps)
    # Kecepatan pusat GT per frame (untuk memilih frame bergerak).
    speed = {}
    for gid in {g for fr in gt_centers.values() for g in fr}:
        prev = None
        for fr in sorted(gt_centers):
            if gid in gt_centers[fr]:
                c = gt_centers[fr][gid][:2]
                if prev is not None:
                    speed[(gid, fr)] = ((c[0] - prev[0]) ** 2
                                        + (c[1] - prev[1]) ** 2) ** 0.5
                prev = c
    dist_table = {}
    for d in offsets:
        dists = []
        for tid, gid in pairs.items():
            for f, b in per_track[tid].items():
                of = int(round(f * ratio)) + d
                if of not in gt_centers or gid not in gt_centers[of]:
                    continue
                gx, gy, act = gt_centers[of][gid]
                if not (act == "Locomotion"
                        or speed.get((gid, of), 0) > 15):
                    continue
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                dists.append(((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5)
        if dists:
            dist_table[d] = {
                "n_pairs": len(dists),
                "median_dist_px": round(statistics.median(dists), 1),
                "mean_dist_px": round(sum(dists) / len(dists), 1),
                "frac_lt_10px": round(
                    sum(1 for v in dists if v < 10) / len(dists), 3),
            }
    # (b) IoU pendukung.
    sample_frames = list(range(0, 6000, 250)) + [5999]
    iou_table = {}
    for d in offsets:
        ious = []
        for f in sample_frames:
            gt_fr = int(round(f * ratio)) + d
            boxes_gt = gt.get(gt_fr, [])
            if not boxes_gt:
                continue
            for by_frame in per_track.values():
                if f not in by_frame:
                    continue
                pb = by_frame[f]
                best = max((iou(pb, gb) for gb, _ in boxes_gt), default=0.0)
                ious.append(best)
        if ious:
            iou_table[d] = {
                "n_boxes": len(ious),
                "median_iou": round(statistics.median(ious), 3),
                "mean_iou": round(sum(ious) / len(ious), 3),
                "frac_iou_ge_0.5": round(
                    sum(1 for v in ious if v >= 0.5) / len(ious), 3),
            }
    # Kriteria terima: offset 0 dekat (<=15 px) dan offset jauh jelas
    # lebih buruk (>=2,5x di ±25 frame, >=4x di ±250 frame).
    m0 = dist_table.get(0, {}).get("median_dist_px", 1e9)
    m25 = max(dist_table.get(-25, {}).get("median_dist_px", 0),
              dist_table.get(25, {}).get("median_dist_px", 0))
    m250 = max(dist_table.get(-250, {}).get("median_dist_px", 0),
               dist_table.get(250, {}).get("median_dist_px", 0))
    verified = m0 <= 15 and m25 >= 2.5 * m0 and m250 >= 4 * m0
    return 0, dist_table, iou_table, verified, pairs


def match_clips(clips, per_track, gt, frame_offset, video_fps=10.0,
                gt_fps=25.0, iou_thresh=IOU_THRESH):
    """Cocokkan tiap klip ke GT lewat IoU kotak. Kembalikan daftar hasil
    per klip: {track_id, start_frame, end_frame, pred_label, confidence,
    gt_label (atau None bila tak tercocokkan), n_votes, votes, tie}."""
    ratio = gt_fps / video_fps
    results = []
    for c in clips:
        tid = int(c["track_id"])
        votes = Counter()
        n_voted_frames = 0
        for f in range(int(c["start_frame"]), int(c["end_frame"]) + 1):
            pb = per_track.get(tid, {}).get(f)
            if pb is None:
                continue
            gt_fr = int(round(f * ratio)) + frame_offset
            boxes_gt = gt.get(gt_fr, [])
            if not boxes_gt:
                continue
            best_iou, best_cls = 0.0, None
            for gb, cls in boxes_gt:
                v = iou(pb, gb)
                if v > best_iou:
                    best_iou, best_cls = v, cls
            if best_iou >= iou_thresh and best_cls is not None:
                votes[best_cls] += 1
                n_voted_frames += 1
        if not votes:
            results.append({**c, "gt_label": None, "n_votes": 0,
                            "votes": {}, "tie": False})
            continue
        top = votes.most_common()
        tie = len(top) > 1 and top[0][1] == top[1][1]
        if tie:
            # Seri: pakai label frame tengah klip.
            mid = (int(c["start_frame"]) + int(c["end_frame"])) // 2
            pb = per_track.get(tid, {}).get(mid)
            mid_label = None
            if pb is not None:
                gt_fr = int(round(mid * ratio)) + frame_offset
                best_iou, best_cls = 0.0, None
                for gb, cls in gt.get(gt_fr, []):
                    v = iou(pb, gb)
                    if v > best_iou:
                        best_iou, best_cls = v, cls
                if best_iou >= iou_thresh:
                    mid_label = best_cls
            gt_label = mid_label if mid_label in [t[0] for t in top[:2]] \
                else top[0][0]
        else:
            gt_label = top[0][0]
        results.append({**c, "gt_label": gt_label,
                        "n_votes": n_voted_frames,
                        "votes": dict(votes), "tie": tie})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--video-fps", type=float, default=10.0)
    ap.add_argument("--gt-fps", type=float, default=25.0)
    ap.add_argument("--video-w", type=float, default=1280.0)
    ap.add_argument("--video-h", type=float, default=720.0)
    ap.add_argument("--iou-thresh", type=float, default=IOU_THRESH)
    ap.add_argument("--map-explor-forag", default="ACTIVE",
                    choices=["ACTIVE", "INGESTIVE"],
                    help="Pemetaan Explor_Forag (sensitivitas saja; "
                         "default ACTIVE sesuai brief).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt, gt_centers, n_rows = load_gt(args.gt, args.video_w, args.video_h,
                                     args.map_explor_forag)
    print(f"GT: {n_rows} baris, {len(gt)} frame beranotasi.")
    per_track = load_pred_tracks(args.tracks)
    print(f"Track prediksi (terfilter): {len(per_track)} track.")

    # --- Langkah 1: verifikasi penyelarasan frame ---
    best_offset, dist_table, iou_table, verified, pairs = verify_alignment(
        per_track, gt, gt_centers, args.video_fps, args.gt_fps)
    print("Verifikasi penyelarasan frame (offset dalam frame GT, 40 mdtk):")
    print("  (a) Jarak pusat median, frame ayam GT sedang bergerak:")
    for d in sorted(dist_table):
        t = dist_table[d]
        print(f"    offset {d:+5d}: n={t['n_pairs']:5d} "
              f"median={t['median_dist_px']:6.1f}px "
              f"mean={t['mean_dist_px']:6.1f}px <10px: {t['frac_lt_10px']:.2f}")
    print("  (b) IoU kotak (pendukung):")
    for d in sorted(iou_table):
        t = iou_table[d]
        print(f"    offset {d:+5d}: n={t['n_boxes']:4d} "
              f"median IoU={t['median_iou']:.3f} IoU>=0.5: {t['frac_iou_ge_0.5']:.2f}")
    print(f"  Pasangan track prediksi->GT: "
          f"{ {str(k): v for k, v in sorted(pairs.items())} }")
    if not verified:
        print("PERINGATAN: pemetaan frame TIDAK terverifikasi dengan yakin — "
              "perbandingan di bawah TIDAK SAH. Berhenti.")
        out = {"frame_alignment_verified": False,
               "frame_alignment_dist_table": dist_table,
               "frame_alignment_iou_table": iou_table,
               "frame_offset_used": None}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        return 1
    print("  Penyelarasan frame TERVERIFIKASI: offset 0, jarak pusat median "
          f"saat bergerak {dist_table[0]['median_dist_px']:.1f}px vs "
          f"{max(dist_table[-25]['median_dist_px'], dist_table[25]['median_dist_px']):.1f}px "
          f"(±1 dtk) dan "
          f"{max(dist_table[-250]['median_dist_px'], dist_table[250]['median_dist_px']):.1f}px "
          f"(±10 dtk); lembah datar ±3 frame (±120 mdtk) di bawah noise gerak.")

    # --- Langkah 2: cocokkan klip ---
    with open(args.predictions) as f:
        pred = json.load(f)
    clips = pred["clip_results"]
    matched = match_clips(clips, per_track, gt, best_offset,
                          args.video_fps, args.gt_fps, args.iou_thresh)
    n_total = len(matched)
    ok = [m for m in matched if m["gt_label"] is not None]
    n_matched = len(ok)
    coverage = n_matched / n_total if n_total else 0.0
    n_ties = sum(1 for m in ok if m["tie"])
    print(f"Klip: {n_total} total, {n_matched} tercocokkan GT "
          f"(coverage {coverage:.1%}), {n_total - n_matched} tak tercocokkan, "
          f"{n_ties} seri suara mayoritas.")

    # --- Langkah 3: metrik ---
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for m in ok:
        cm[m["gt_label"]][m["label"]] += 1
    n_correct = sum(cm[c][c] for c in CLASSES)
    acc = n_correct / n_matched if n_matched else 0.0
    per_class = {}
    for c in CLASSES:
        tp = cm[c][c]
        fn = sum(cm[c][p] for p in CLASSES if p != c)
        fp = sum(cm[g][c] for g in CLASSES if g != c)
        per_class[c] = {
            "n_gt": tp + fn,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4)
                  if (2 * tp + fp + fn) else None,
        }
    ing_as_rest = (cm["INGESTIVE"]["RESTING"] / per_class["INGESTIVE"]["n_gt"]
                   if per_class["INGESTIVE"]["n_gt"] else None)
    rest_as_ing = (cm["RESTING"]["INGESTIVE"] / per_class["RESTING"]["n_gt"]
                   if per_class["RESTING"]["n_gt"] else None)
    gt_dist_matched = Counter(m["gt_label"] for m in ok)
    pred_dist_matched = Counter(m["label"] for m in ok)
    unmatched_by_track = Counter(
        str(m["track_id"]) for m in matched if m["gt_label"] is None)
    unmatched_by_time = Counter(
        ("awal <200 dtk" if m["start_frame"] < 2000
         else "tengah 200-400 dtk" if m["start_frame"] < 4000
         else "akhir >400 dtk")
        for m in matched if m["gt_label"] is None)
    conf_bins = defaultdict(lambda: [0, 0])  # [benar, total]
    for m in ok:
        b = ("<0.9" if m["confidence"] < 0.9 else
             "0.9-0.99" if m["confidence"] < 0.99 else ">=0.99")
        conf_bins[b][1] += 1
        if m["label"] == m["gt_label"]:
            conf_bins[b][0] += 1

    result = {
        "frame_alignment_verified": True,
        "frame_alignment_dist_table_moving_only": dist_table,
        "frame_alignment_iou_table": iou_table,
        "track_pairs_pred_to_gt": {str(k): v for k, v in sorted(pairs.items())},
        "frame_offset_used": best_offset,
        "frame_mapping": f"frame_gt = round(frame_10fps * {args.gt_fps}/{args.video_fps}) + {best_offset}",
        "iou_thresh": args.iou_thresh,
        "mapping_gt_actions": {k: sorted(v) for k, v in {
            "INGESTIVE": {"Eat", "Drink"} | (
                {"Explor_Forag"} if args.map_explor_forag == "INGESTIVE" else set()),
            "ACTIVE": {"Locomotion", "Explor_Forag", "Stretching", "Feather_Pecking"} - (
                {"Explor_Forag"} if args.map_explor_forag == "INGESTIVE" else set()),
            "RESTING": {"Inactive", "Resting", "Preening", "Panting", "Lying (aksi kosong)"},
        }.items()},
        "explor_forag_mapped_to": args.map_explor_forag,
        "n_clips_total": n_total,
        "n_clips_matched": n_matched,
        "n_clips_unmatched": n_total - n_matched,
        "coverage": round(coverage, 4),
        "n_ties_majority_vote": n_ties,
        "accuracy_matched": round(acc, 4),
        "n_correct": n_correct,
        "confusion_matrix_rows_gt_cols_pred": cm,
        "per_class": per_class,
        "ingestive_predicted_as_resting": round(ing_as_rest, 4)
            if ing_as_rest is not None else None,
        "resting_predicted_as_ingestive": round(rest_as_ing, 4)
            if rest_as_ing is not None else None,
        "gt_label_distribution_matched": dict(gt_dist_matched),
        "pred_label_distribution_matched": dict(pred_dist_matched),
        "unmatched_by_track": dict(unmatched_by_track),
        "unmatched_by_time": dict(unmatched_by_time),
        "accuracy_by_pred_confidence": {
            b: {"correct": v[0], "total": v[1],
                "acc": round(v[0] / v[1], 4) if v[1] else None}
            for b, v in sorted(conf_bins.items())},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print(f"\nAkurasi (klip tercocokkan, n={n_matched}): {acc:.2%}")
    print("Confusion matrix (baris=GT, kolom=prediksi):")
    print("            " + "  ".join(f"{c:>10}" for c in CLASSES))
    for g in CLASSES:
        print(f"  {g:>9} " + "  ".join(f"{cm[g][p]:>10}" for p in CLASSES))
    for c in CLASSES:
        pc = per_class[c]
        print(f"  {c}: recall {pc['recall']:.2%} precision {pc['precision']:.2%} "
              f"(n GT {pc['n_gt']})")
    print(f"  INGESTIVE GT diprediksi RESTING: {ing_as_rest:.2%}")
    print(f"  RESTING GT diprediksi INGESTIVE: {rest_as_ing:.2%}")
    print(f"Hasil: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
