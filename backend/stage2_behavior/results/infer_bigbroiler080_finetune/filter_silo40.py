"""Buang klip silo drinker ID 40 dari track_predictions.json (model fine-tune).

Prosedur sama seperti baseline 080: inference jalan di tracks terfilter 52 ID
(termasuk silo), lalu klip ID 40 dibuang pada tahap filter prediksi memakai
filter_static_tracks.filter_predictions. Hasil: track_predictions_filtered.json
(6.936 klip, 26 track) + blok static_filter seperti baseline.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "stage1_detection"))
from filter_static_tracks import filter_predictions  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
preds = json.load(open(os.path.join(OUT, "track_predictions.json")))
out = filter_predictions(preds, [40])
out["static_filter"]["note"] = (
    "ID 40 = silo drinker putih (3.998 frame, skor rata-rata 0,92), "
    "terverifikasi visual frame 3000/5700; lolos filter otomatis 10 px karena "
    "oklusi membuat pusat kotaknya bergoyang 28,8 x 51,3 px. 36 ID lain sudah "
    "dibuang filter otomatis di akhir tracking (88 -> 52 ID) sebelum inference. "
    "Prosedur sama seperti baseline 080 (commit 11fc5a9); model di sini model "
    "fine-tune v2. Lihat fp_removal_080.json."
)
json.dump(out, open(os.path.join(OUT, "track_predictions_filtered.json"), "w"))
print("sebelum:", preds["n_clips"], "klip -> sesudah:", out["n_clips"], "klip")
print("distribusi:", out["label_distribution"])
print("track di summary:", len(out["track_summary"]), "(ID 40 ada:",
      "40" in out["track_summary"], ")")
assert "40" not in out["track_summary"]
assert all(c["track_id"] != 40 for c in out["clip_results"])
assert out["n_clips"] == 6936, f"jumlah klip {out['n_clips']} != 6936 harapan baseline"
print("OK: silo terbuang, 6.936 klip tersisa (sama seperti baseline).")
