"""Hitung metrik perbandingan GT per paruh video 091 (paruh-1 vs paruh-2).

Dipakai untuk uji video 091 dengan model fine-tune: klip 091 paruh-1
(frame 0-2999) dipakai sebagai TRAINING fine-tune, jadi akurasi di sana
terkontaminasi. Yang jujur dibandingkan dengan baseline adalah paruh-2
saja (frame >= 3000, tidak pernah dilatih).

Memakai fungsi yang SAMA dari compare_gt.py (load_gt, load_pred_tracks,
match_clips) dengan offset frame 0 yang sudah terverifikasi di baseline
(gt_comparison.json: frame_alignment_verified=True). Klip yang
mengangkangi batas (start_frame 2992, frame 2992-3007) dikeluarkan dari
kedua paruh dan dilaporkan terpisah.

Pemakaian:
    python analyze_gt_halves.py --tracks <tracks.json> \
        --predictions <track_predictions.json> \
        --gt ~/workspace/pbl-ayam-obs-rtsp/topview_test/bigbroiler_091_3_32_1920_008_0_10.csv \
        --out <hasil_halves.json>
"""

import argparse
import json
from collections import Counter, defaultdict

from compare_gt import CLASSES, load_gt, load_pred_tracks, match_clips

SPLIT_FRAME = 3000  # paruh-1: frame < 3000; paruh-2: frame >= 3000


def metrics_for(matched):
    """Metrik (rumus sama seperti compare_gt.main) untuk satu subset klip."""
    ok = [m for m in matched if m["gt_label"] is not None]
    n_total = len(matched)
    n_matched = len(ok)
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for m in ok:
        cm[m["gt_label"]][m["label"]] += 1
    n_correct = sum(cm[c][c] for c in CLASSES)
    acc = n_correct / n_matched if n_matched else None
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
    return {
        "n_clips_total": n_total,
        "n_clips_matched": n_matched,
        "n_clips_unmatched": n_total - n_matched,
        "coverage": round(n_matched / n_total, 4) if n_total else None,
        "accuracy_matched": round(acc, 4) if acc is not None else None,
        "n_correct": n_correct,
        "confusion_matrix_rows_gt_cols_pred": cm,
        "per_class": per_class,
        "ingestive_predicted_as_resting": (
            round(cm["INGESTIVE"]["RESTING"] / per_class["INGESTIVE"]["n_gt"], 4)
            if per_class["INGESTIVE"]["n_gt"] else None),
        "resting_predicted_as_ingestive": (
            round(cm["RESTING"]["INGESTIVE"] / per_class["RESTING"]["n_gt"], 4)
            if per_class["RESTING"]["n_gt"] else None),
        "gt_label_distribution_matched": dict(Counter(m["gt_label"] for m in ok)),
        "pred_label_distribution_matched": dict(Counter(m["label"] for m in ok)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt, _, n_rows = load_gt(args.gt)
    print(f"GT: {n_rows} baris.")
    per_track = load_pred_tracks(args.tracks)
    print(f"Track prediksi: {len(per_track)} track.")
    with open(args.predictions) as f:
        pred = json.load(f)
    clips = pred["clip_results"]
    print(f"Klip prediksi: {len(clips)}")

    # Offset 0 sudah terverifikasi di baseline 091 (lihat gt_comparison.json).
    matched = match_clips(clips, per_track, gt, 0)
    print(f"Tercocokkan GT: {sum(1 for m in matched if m['gt_label'] is not None)}"
          f" / {len(matched)}")

    subsets = {
        "all": matched,
        "first_half": [m for m in matched if int(m["end_frame"]) < SPLIT_FRAME],
        "second_half": [m for m in matched if int(m["start_frame"]) >= SPLIT_FRAME],
        "straddling": [m for m in matched
                       if int(m["start_frame"]) < SPLIT_FRAME
                       and int(m["end_frame"]) >= SPLIT_FRAME],
    }
    result = {"split_frame": SPLIT_FRAME,
              "frame_alignment_offset_used": 0,
              "note": "first_half = klip yang seluruhnya di frame < 3000 "
                      "(dipakai sebagai training fine-tune = terkontaminasi); "
                      "second_half = klip yang seluruhnya di frame >= 3000 "
                      "(tidak pernah dilatih = angka jujur); straddling = "
                      "klip yang mengangkangi batas (start 2992) dan "
                      "dikeluarkan dari kedua paruh."}
    for name, sub in subsets.items():
        result[name] = metrics_for(sub)
        r = result[name]
        print(f"\n[{name}] n={r['n_clips_total']} matched={r['n_clips_matched']} "
              f"acc={r['accuracy_matched']}")
        for c in CLASSES:
            pc = r["per_class"][c]
            print(f"  {c}: recall {pc['recall']} precision {pc['precision']} "
                  f"(n GT {pc['n_gt']})")
        print(f"  INGESTIVE->RESTING: {r['ingestive_predicted_as_resting']}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nHasil: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
