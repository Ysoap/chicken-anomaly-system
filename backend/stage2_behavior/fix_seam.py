"""Perbaiki tracks.json Big Broiler 091 yang tercemar sambungan resume.

INSIDEN (15 Nov 2026): tracking 6000 frame mula-mula jalan sebagai satu
proses, lalu VM restart dan prosesnya mati (SIGTERM) di frame 3600 —
checkpoint JSON 0-3599 selamat, tapi state ByteTrack (pickle) belum
ada karena script versi lama belum menyimpannya. Resume memakai tracker
BARU dari frame 3600, dan tracker baru itu memberi ID dengan urutan
BERBEDA dari tracker lama: mis. ayam yang tadinya ID 1 menjadi ID 5,
ID 6 menjadi ID 1, dst. (hanya ID 2/3/4 yang kebetulan sama — kotak
listrik + 2 ayam). Akibatnya di tracks.json mentah, ID 1/5/6/7/8/9/11
masing-masing berisi DUA ayam berbeda (satu sebelum frame 3600, satu
sesudahnya) — statistik per ID dari file mentah itu KORUP.

Perbaikan: semua track di frame >= SEAM diberi ID baru (+OFFSET) supaya
satu ID hanya merujuk satu objek. Setelah itu
``merge_track_fragments.py`` (stitching) bisa menyambung kembali
pasangan seam (gap 0 frame, jarak pusat ~0) seperti fragmen biasa.

    python fix_seam.py --tracks tracks_bigbroiler091.json \\
        --seam 3600 --offset 1000 \\
        --out tracks_bigbroiler091_fixed.json

Verifikasi di akhir (gagal keras bila dilanggar): tidak ada ID yang
muncul di kedua sisi seam, dan jumlah box-frame total tidak berubah.
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--seam", type=int, required=True,
                    help="Frame pertama dari run resume (di sini 3600).")
    ap.add_argument("--offset", type=int, default=1000,
                    help="Ditambahkan ke ID di frame >= seam.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.tracks) as f:
        data = json.load(f)
    n_before = n_after = 0
    ids_before, ids_after = set(), set()
    for fr in data["frames"]:
        fi = int(fr["frame"])
        for t in fr["tracks"]:
            if fi < args.seam:
                ids_before.add(int(t["track_id"]))
                n_before += 1
            else:
                ids_after.add(int(t["track_id"]))
                t["track_id"] = int(t["track_id"]) + args.offset
                n_after += 1
    # Verifikasi: ID hasil renumber tidak boleh sama dengan ID lama.
    renumbered = {i + args.offset for i in ids_after}
    clash = renumbered & ids_before
    assert not clash, f"ID hasil renumber bentrok dengan ID lama: {sorted(clash)}"
    total_after = sum(len(fr["tracks"]) for fr in data["frames"])
    assert total_after == n_before + n_after, "jumlah box berubah!"
    data["seam_fix"] = {
        "seam_frame": args.seam, "offset": args.offset,
        "note": ("Resume 15 Nov 2026 setelah VM restart mematikan tracking "
                 "di frame 3600; tracker baru memberi ID dengan urutan "
                 "berbeda. Frame >= seam di-renumber +offset supaya satu "
                 "ID = satu objek. Sambung kembali dengan "
                 "merge_track_fragments.py."),
    }
    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"Box sebelum seam: {n_before}; sesudah: {n_after}; total {total_after}")
    print(f"ID sebelum seam: {sorted(ids_before)}")
    print(f"ID sesudah seam (setelah +{args.offset}): {sorted(renumbered)}")
    print(f"Tersimpan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
