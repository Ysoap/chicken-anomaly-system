"""Analisis hasil pipeline penuh Big Broiler 091 + bandingkan ke GT.

Membaca tracks.json (Stage 1) + track_predictions.json (Stage 2) dan
opsional CSV ground truth Big Broiler, lalu menulis summary.json:

- jumlah ID unik, median span track, box-frame total & putih
  (definisi kotak putih = box milik track TANPA klip, sama seperti
  laporan fCam16),
- klip per label + track terklasifikasi,
- track yang diduga FP (statis: rentang gerak pusat <= 3 px sepanjang
  track DAN tidak cocok dengan GT — benda mati seperti kotak listrik /
  bohlam drinker; ayam asli yang diam pun kotaknya bergoyang beberapa
  px karena noise deteksi),
- perbandingan ke GT (kalau CSV diberikan): berapa track GT tercocok,
  berapa ID kita per track GT (ID switch), jarak median kotak kita
  ke kotak GT, dan distribusi aksi GT (dipetakan kasar ke 3 kelas
  kita — BUKAN akurasi: taksonomi GT beda).

    python analyze_bigbroiler091.py \\
        --tracks results/infer_bigbroiler091/tracks_bigbroiler091.json \\
        --predictions results/infer_bigbroiler091/track_predictions.json \\
        --gt ~/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_3_32_1920_008_0_10.csv \\
        --video-fps 10 --gt-fps 25 --video-w 1280 --video-h 720 \\
        --out results/infer_bigbroiler091/summary.json
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict


def load_tracks(path):
    with open(path) as f:
        data = json.load(f)
    per_track = defaultdict(dict)  # tid -> {frame: (x1,y1,x2,y2,score)}
    n_boxes = 0
    for fr in data["frames"]:
        for t in fr["tracks"]:
            per_track[int(t["track_id"])][int(fr["frame"])] = (
                tuple(float(v) for v in t["bbox"]) + (float(t["score"]),))
            n_boxes += 1
    return data.get("fps", 10.0), per_track, n_boxes


def track_stats(by_frame):
    fs = sorted(by_frame)
    cxs = [ (b[0] + b[2]) / 2 for b in by_frame.values() ]
    cys = [ (b[1] + b[3]) / 2 for b in by_frame.values() ]
    scores = [b[4] for b in by_frame.values()]
    return {
        "n_frames": len(fs),
        "span_first": fs[0], "span_last": fs[-1],
        "span_len": fs[-1] - fs[0] + 1,
        "center_range_x": round(max(cxs) - min(cxs), 1),
        "center_range_y": round(max(cys) - min(cys), 1),
        "mean_score": round(sum(scores) / len(scores), 3),
        "min_score": round(min(scores), 3),
    }


def load_gt(path, video_w, video_h):
    """GT Big Broiler: frame (fps GT), track, x1..y2 ternormalisasi.
    Kembalikan {frame_gt: {gtid: (cx, cy, w, h)}} piksel video."""
    gt = defaultdict(dict)
    actions = Counter()
    with open(path) as f:
        for row in csv.DictReader(f):
            fr = int(row["frame"])
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            gt[fr][int(row["track"])] = (
                (x1 + x2) / 2 * video_w, (y1 + y2) / 2 * video_h,
                (x2 - x1) * video_w, (y2 - y1) * video_h)
            actions[row["actions_ground"]] += 1
    return gt, actions


def match_gt(per_track, gt, video_fps, gt_fps, dist_thresh=60.0):
    """Untuk tiap ID kita, hitung GT track yang paling sering jadi tetangga
    terdekat (jarak pusat < ambang). Kembalikan {tid: {gtid: n_frame}}."""
    ratio = gt_fps / video_fps
    matches = {}
    for tid, by_frame in per_track.items():
        votes = Counter()
        for f, b in by_frame.items():
            of = int(round(f * ratio))
            if of not in gt:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best, bd = None, 1e18
            for gid, (gx, gy, _w, _h) in gt[of].items():
                dd = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                if dd < bd:
                    bd, best = dd, gid
            if bd < dist_thresh:
                votes[best] += 1
        matches[tid] = dict(votes)
    return matches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--video-fps", type=float, default=10.0)
    ap.add_argument("--gt-fps", type=float, default=25.0)
    ap.add_argument("--video-w", type=float, default=1280)
    ap.add_argument("--video-h", type=float, default=720)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fps, per_track, n_boxes = load_tracks(args.tracks)
    with open(args.predictions) as f:
        pred = json.load(f)
    clip_results = pred["clip_results"]
    track_summary = pred["track_summary"]

    stats = {tid: track_stats(by) for tid, by in per_track.items()}
    spans = sorted(s["span_len"] for s in stats.values())
    n_clips_per_track = Counter(r["track_id"] for r in clip_results)
    classified = set(track_summary)
    white_boxes = sum(s["n_frames"] for tid, s in stats.items()
                      if tid not in n_clips_per_track)
    summary = {
        "video_fps": fps,
        "n_frames": max((max(by) for by in per_track.values()), default=0) + 1,
        "n_unique_ids": len(per_track),
        "n_boxes_total": n_boxes,
        "median_track_span_frames": statistics.median(spans) if spans else 0,
        "track_span_min": min(spans) if spans else 0,
        "track_span_max": max(spans) if spans else 0,
        "n_clips": len(clip_results),
        "label_distribution_clips": dict(Counter(r["label"] for r in clip_results)),
        "n_tracks_classified": len(classified),
        "n_tracks_unclassified": len(per_track) - len(classified),
        "white_box_frames": white_boxes,
        "tracks": {str(tid): {**stats[tid],
                              "n_clips": n_clips_per_track.get(tid, 0),
                              "majority_label": (track_summary[str(tid)]["majority_label"]
                                                 if str(tid) in track_summary else None)}
                   for tid in sorted(stats)},
    }

    # Track diduga FP: statis (gerak pusat <= 3 px di kedua sumbu) dan
    # panjang >= 30 frame. Ayam asli yang diam pun kotaknya bergeser
    # beberapa px karena noise deteksi — kotak yang pixel-identik
    # selama ratusan frame adalah benda mati.
    fp_suspects = [tid for tid, s in stats.items()
                   if s["center_range_x"] <= 3 and s["center_range_y"] <= 3
                   and s["n_frames"] >= 30]
    summary["fp_suspects_static"] = sorted(fp_suspects)
    summary["fp_suspect_boxes"] = sum(stats[t]["n_frames"] for t in fp_suspects)
    summary["fp_suspect_clips"] = sum(n_clips_per_track.get(t, 0)
                                      for t in fp_suspects)

    if args.gt:
        gt, gt_actions = load_gt(args.gt, args.video_w, args.video_h)
        matches = match_gt(per_track, gt, args.video_fps, args.gt_fps)
        gt_ids = sorted({g for fr in gt.values() for g in fr})
        matched_gt = sorted({g for votes in matches.values() for g in votes})
        # ID kita per track GT (indikasi ID switch bila > 1).
        per_gt = defaultdict(list)
        for tid, votes in matches.items():
            if votes:
                best = max(votes, key=votes.get)
                per_gt[best].append((tid, votes[best]))
        summary["ground_truth"] = {
            "n_gt_tracks": len(gt_ids),
            "gt_tracks_matched": matched_gt,
            "n_gt_tracks_matched": len(matched_gt),
            "our_ids_per_gt_track": {str(g): [t for t, _ in v]
                                     for g, v in sorted(per_gt.items())},
            "our_ids_with_no_gt_match": sorted(
                tid for tid, votes in matches.items() if not votes),
            "matches": {str(tid): votes for tid, votes in sorted(matches.items())},
            "gt_action_distribution": dict(gt_actions),
        }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"ID unik: {summary['n_unique_ids']}; "
          f"median span: {summary['median_track_span_frames']} frame; "
          f"klip: {summary['n_clips']} {summary['label_distribution_clips']}; "
          f"kotak putih: {summary['white_box_frames']}; "
          f"FP statis: {summary['fp_suspects_static']} "
          f"({summary['fp_suspect_boxes']} box, "
          f"{summary['fp_suspect_clips']} klip)")
    if args.gt:
        g = summary["ground_truth"]
        print(f"GT: {g['n_gt_tracks_matched']}/{g['n_gt_tracks']} track tercocok; "
              f"ID tanpa GT: {g['our_ids_with_no_gt_match']}")
    print(f"Ringkasan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
