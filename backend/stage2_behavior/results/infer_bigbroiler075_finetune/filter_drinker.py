"""Buang klip drinker (ID 33 + ID 112) dari track_predictions.json
(model fine-tune v2).

Prosedur sama seperti filter_silo40.py di 080: inference jalan di tracks
terfilter 62 ID (termasuk drinker), lalu klip drinker dibuang pada tahap
filter prediksi memakai filter_static_tracks.filter_predictions.
Hasil: track_predictions_filtered.json (6.851 klip, 60 track).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "stage1_detection"))
from filter_static_tracks import filter_predictions  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
preds = json.load(open(os.path.join(OUT, "track_predictions.json")))
out = filter_predictions(preds, [33, 112])
out["static_filter"]["note"] = (
    "ID 33 (frame 685-4081, 3.395 box, skor rata-rata 0,91) + ID 112 "
    "(frame 4201-4630, 428 box, skor 0,92) = SATU drinker lonceng putih "
    "yang sama, terverifikasi visual frame 2000/4400 (kotak median "
    "130x216 px). Lolos filter otomatis 10 px karena oklusi membuat "
    "pusat kotaknya bergoyang (ID 33: 11,0 x 11,0 px; ID 112: 8,7 x "
    "32,6 px) — mekanisme sama seperti silo drinker ID 40 di 080. "
    "38 ID lain sudah dibuang filter otomatis di akhir tracking "
    "(100 -> 62 ID) sebelum inference. Lihat fp_removal_075.json."
)
json.dump(out, open(os.path.join(OUT, "track_predictions_filtered.json"), "w"))
print("sebelum:", preds["n_clips"], "klip -> sesudah:", out["n_clips"], "klip")
print("distribusi:", out["label_distribution"])
print("track di summary:", len(out["track_summary"]), "(33 ada:",
      "33" in out["track_summary"], ", 112 ada:", "112" in out["track_summary"], ")")
assert "33" not in out["track_summary"] and "112" not in out["track_summary"]
assert all(c["track_id"] not in (33, 112) for c in out["clip_results"])
assert out["n_clips"] == 6851, f"jumlah klip {out['n_clips']} != 6851 harapan"
print("OK: drinker terbuang, 6.851 klip tersisa.")
