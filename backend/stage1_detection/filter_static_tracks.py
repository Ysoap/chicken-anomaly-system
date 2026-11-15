"""Filter track statis (benda mati) dari hasil tracking Stage 1.

Alasan: detektor RT-DETR kadang mengunci benda mati (kotak listrik,
bohlam drinker) sebagai "ayam" dengan skor tinggi (0,97) — threshold skor
tidak bisa membedakannya. Tapi benda mati TIDAK PERNAH BERGERAK, sementara
ayam asli selalu berpindah walau sedang istirahat.

Aturan: untuk tiap track, hitung rentang pusat bbox (max-min x dan y)
sepanjang hidup track. Kalau perpindahan maksimum < ambang (default 10 px
pada video 720p), track dianggap statis dan dibuang.

Bukti di Big Broiler 091 (6000 frame, 720p):
- 14 ID FP statis punya perpindahan maks 0,4-3,8 px (kotak listrik ID 2:
  1,0 x 0,4 px selama 6000 frame penuh; 749 klip, semua RESTING).
- Ayam asli dengan klip paling diam (ID 1032, istirahat 76 dtk) punya
  perpindahan 31,2 px — jarak 8x dari FP terbesar.
- Ambang 10 px berada di tengah celah (3,8 ... 31,2): membuang 24 track
  (1.045 klip, 14,6%) dan tidak membuang satu pun ayam asli.
- Fragmen pendek (<16 frame) yang ikut terbuang memang tidak bisa
  menghasilkan klip 16 frame, jadi tidak ada yang hilang buat Stage 2.

Dipakai dua cara:
1. Otomatis di akhir track_video.py (default aktif, --no-filter-static
   untuk mematikan): menulis <out>.filtered.json + laporan di stdout.
2. Manual pada hasil lama:
       python filter_static_tracks.py --tracks tracks.json \
           --out tracks.filtered.json
       python filter_static_tracks.py --tracks tracks.json \
           --out tracks.filtered.json \
           --predictions track_predictions.json \
           --predictions-out track_predictions_filtered.json
"""

import argparse
import json
import os
from collections import Counter

DEFAULT_MIN_DISPLACEMENT_PX = 10.0


def track_displacements(frames):
    """Rentang pusat bbox per track_id: {id: {n_frames, range_x, range_y}}.

    frames: list {"frame", "name", "tracks": [{"track_id", "bbox", ...}]}.
    """
    xs = {}
    ys = {}
    counts = Counter()
    for fr in frames:
        for t in fr.get("tracks", []):
            tid = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            xs.setdefault(tid, []).append((x1 + x2) / 2.0)
            ys.setdefault(tid, []).append((y1 + y2) / 2.0)
            counts[tid] += 1
    out = {}
    for tid in counts:
        rx = max(xs[tid]) - min(xs[tid])
        ry = max(ys[tid]) - min(ys[tid])
        out[tid] = {
            "n_frames": counts[tid],
            "range_x": round(rx, 2),
            "range_y": round(ry, 2),
            "max_disp": round(max(rx, ry), 2),
        }
    return out


def filter_static_tracks(data, min_displacement_px=DEFAULT_MIN_DISPLACEMENT_PX):
    """Buang track statis dari data tracks.json.

    Mengembalikan (filtered_data, report). filtered_data formatnya sama
    persis dengan input; frame yang isinya habis terbuang tetap ada dengan
    tracks kosong. Data input tidak diubah.
    """
    disp = track_displacements(data["frames"])
    dropped = {tid for tid, d in disp.items()
               if d["max_disp"] < min_displacement_px}
    kept = {tid for tid in disp if tid not in dropped}
    filtered_frames = []
    for fr in data["frames"]:
        kept_tracks = [t for t in fr.get("tracks", [])
                       if t["track_id"] not in dropped]
        filtered_frames.append({**fr, "tracks": kept_tracks})
    filtered = {**data, "frames": filtered_frames}
    n_boxes_before = sum(len(fr.get("tracks", [])) for fr in data["frames"])
    n_boxes_after = sum(len(fr.get("tracks", [])) for fr in filtered_frames)
    report = {
        "min_displacement_px": min_displacement_px,
        "n_tracks_before": len(disp),
        "n_tracks_kept": len(kept),
        "n_tracks_dropped": len(dropped),
        "dropped_track_ids": sorted(dropped),
        "kept_track_ids": sorted(kept),
        "n_boxes_before": n_boxes_before,
        "n_boxes_after": n_boxes_after,
        "n_boxes_dropped": n_boxes_before - n_boxes_after,
        "displacements": {str(k): v for k, v in sorted(disp.items())},
    }
    return filtered, report


def filter_predictions(predictions, dropped_ids):
    """Buang klip milik track statis dari track_predictions.json.

    Menghitung ulang n_clips + label_distribution + track_summary.
    Field lain (mode, checkpoint, video) dipertahankan apa adanya.
    """
    dropped = {str(tid) for tid in dropped_ids}
    kept_clips = [c for c in predictions.get("clip_results", [])
                  if str(c.get("track_id")) not in dropped]
    dist = Counter(c["label"] for c in kept_clips)
    summary = {tid: s for tid, s in predictions.get("track_summary", {}).items()
               if str(tid) not in dropped}
    out = {**predictions,
           "n_clips": len(kept_clips),
           "label_distribution": dict(dist),
           "clip_results": kept_clips,
           "track_summary": summary,
           "static_filter": {
               "dropped_track_ids": sorted(dropped),
               "n_clips_before": predictions.get("n_clips"),
               "n_clips_dropped": (predictions.get("n_clips", 0)
                                  - len(kept_clips)),
           }}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True,
                    help="tracks.json hasil tracking (format track_video.py).")
    ap.add_argument("--out", required=True,
                    help="tracks.json hasil filter.")
    ap.add_argument("--min-displacement", type=float,
                    default=DEFAULT_MIN_DISPLACEMENT_PX,
                    help="Ambang perpindahan pusat bbox (px). Default 10.")
    ap.add_argument("--predictions", default=None,
                    help="track_predictions.json untuk ikut difilter.")
    ap.add_argument("--predictions-out", default=None,
                    help="track_predictions.json hasil filter.")
    args = ap.parse_args()

    with open(args.tracks) as f:
        data = json.load(f)
    filtered, report = filter_static_tracks(data, args.min_displacement)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(filtered, f)
    report_path = args.out + ".report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"Track: {report['n_tracks_before']} -> {report['n_tracks_kept']} "
          f"(buang {report['n_tracks_dropped']}: {report['dropped_track_ids']})")
    print(f"Box: {report['n_boxes_before']} -> {report['n_boxes_after']} "
          f"(buang {report['n_boxes_dropped']})")
    print(f"Hasil filter: {args.out}\nLaporan: {report_path}")

    if args.predictions:
        if not args.predictions_out:
            raise SystemExit("--predictions-out wajib diisi bila "
                             "--predictions dipakai.")
        with open(args.predictions) as f:
            preds = json.load(f)
        filtered_preds = filter_predictions(preds, report["dropped_track_ids"])
        with open(args.predictions_out, "w") as f:
            json.dump(filtered_preds, f)
        before = preds.get("label_distribution", {})
        after = filtered_preds["label_distribution"]
        print(f"Klip: {preds.get('n_clips')} -> {filtered_preds['n_clips']} "
              f"(buang {filtered_preds['static_filter']['n_clips_dropped']})")
        print(f"Distribusi sebelum: {before}")
        print(f"Distribusi sesudah: {after}")
        print(f"Hasil prediksi terfilter: {args.predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
