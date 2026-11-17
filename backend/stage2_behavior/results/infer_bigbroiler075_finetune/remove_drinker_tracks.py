"""Buat tracks_bigbroiler075.filtered_nofp.json: tracks terfilter TANPA
drinker (ID 33 + ID 112) — dipakai render + banding GT.

ID 33 (frame 685-4081, 3.395 box) dan ID 112 (frame 4201-4630, 428 box)
adalah SATU drinker lonceng putih yang sama (terverifikasi visual frame
2000/4400; kotak median 130x216 px, skor 0,91-0,92). Keduanya lolos filter
statis otomatis 10 px karena pusat kotaknya bergoyang 11,0 px (ID 33) /
32,6 px (ID 112) saat ayam lewat di depannya — mekanisme sama seperti
silo drinker ID 40 di 080 (28,8 x 51,3 px). Lihat fp_removal_075.json.
"""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
FP_IDS = {33, 112}

data = json.load(open(os.path.join(OUT, "tracks_bigbroiler075.filtered.json")))
n_before = sum(len(fr["tracks"]) for fr in data["frames"])
for fr in data["frames"]:
    fr["tracks"] = [t for t in fr["tracks"] if t["track_id"] not in FP_IDS]
n_after = sum(len(fr["tracks"]) for fr in data["frames"])
json.dump(data, open(os.path.join(
    OUT, "tracks_bigbroiler075.filtered_nofp.json"), "w"))
print(f"box {n_before} -> {n_after} (buang {n_before - n_after} box drinker)")
ids = {t["track_id"] for fr in data["frames"] for t in fr["tracks"]}
print(f"track tersisa: {len(ids)} (33 ada: {33 in ids}, 112 ada: {112 in ids})")
assert 33 not in ids and 112 not in ids
assert n_before - n_after == 3823, f"buang {n_before - n_after} != 3823"
print("OK: drinker terbuang dari tracks (nofp).")
