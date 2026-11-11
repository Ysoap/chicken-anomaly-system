"""Gabung fragmen track Stage 1 (track stitching) untuk inference Stage 2.

ByteTrack kadang memutus satu ayam menjadi beberapa ID (oklusi, ayam
keluar-masuk frame): satu ayam muncul sebagai beberapa "run" pendek
(rangkaian frame berurutan dengan satu ID) yang tidak cukup panjang
untuk membentuk klip 16 frame, sehingga di video berlabel muncul sebagai
kotak putih tanpa label. Script ini menyambung run-run yang kemungkinan
besar ayam yang sama.

PENTING — komponen dimulai dari ID utuh, bukan dari run:
setiap komponen awalnya berisi SEMUA run dari satu ID asli. Edge
(run A -> run B) menggabungkan SELURUH komponen kedua ID tersebut.
Alasannya (diuji 11 Nov 2026 di fCam16): varian run-level — di mana
run-run satu ID boleh terpecah ke komponen berbeda — justru MENAMBAH
kotak putih (364 -> 512 box-frame putih; dengan fase perbaikan naive
malah 610), karena run ekor/kepala dari ID yang tadinya terklasifikasi
terpisah menjadi track putih dan kehilangan label fallback-nya. Dengan
komponen ID-utuh, merge TIDAK PERNAH mengurangi box-frame berlabel
(terbukti di bawah: tiap edge yang diterima punya gain >= 0).

Kriteria sambung (KONSERVATIF — lebih baik tidak tersambung daripada
tersambung ke ayam yang salah):
  * run B mulai setelah run A berakhir, gap 0..MAX_GAP frame;
  * jarak pusat bbox terakhir A ke bbox pertama B <= MAX_DIST_REF x
    ukuran bbox (ukuran = rata-rata sisi terpanjang kedua bbox);
  * rasio luas bbox dalam [MIN_AREA_RATIO, MAX_AREA_RATIO];
  * uji rasio (ratio test): jarak kandidat terbaik harus < RATIO_TEST x
    jarak kandidat terbaik kedua untuk run sumber yang sama — kalau dua
    kandidat hampir sama dekatnya, sambungan ditolak (ambigu).
  * Dua komponen yang disambung tidak boleh punya frame yang tumpang
    tindih (kalau tidak, satu frame akan punya dua bbox untuk ID yang
    sama — data korup). Satu run maksimal satu pendahulu + satu
    penerus; rantai (A -> B -> C) didukung.

Jaminan: karena komponen tidak pernah terpecah dan merge hanya
menambah frame ke komponen terklasifikasi (atau membentuk komponen
terklasifikasi baru), jumlah box-frame berlabel TIDAK PERNAH turun.
Script memverifikasi ini di akhir dan gagal keras bila dilanggar.

ID track gabungan = ID run paling awal dalam rantai. File tracks asli
TIDAK diubah.

Pemakaian:
    python merge_track_fragments.py --tracks tracks_fCam16.json \\
        --out merged_tracks_fCam16.json --report merge_report.json

Keluaran: tracks.json hasil gabungan (format sama dengan input) +
laporan JSON (pasangan yang digabung beserta gap/jaraknya, yang
ditolak, dan ringkasan sebelum/sesudah).
"""

import argparse
import json
import math
import os
import sys

DEFAULT_MAX_GAP = 4          # frame; 0,4 dtk @10 fps
DEFAULT_MAX_DIST_REF = 0.6   # jarak pusat <= 0,6 x ukuran bbox
DEFAULT_MIN_AREA_RATIO = 0.6
DEFAULT_MAX_AREA_RATIO = 1.67
DEFAULT_RATIO_TEST = 0.75    # terbaik < 0,75 x terbaik kedua
CLIP_LEN = 16
STRIDE = 8


def center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_size(bbox):
    return ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2.0


def longest_run(frames_set):
    """Panjang run konsekutif terpanjang dalam satu set frame."""
    fs = sorted(frames_set)
    if not fs:
        return 0
    best = cur = 1
    for a, b in zip(fs, fs[1:]):
        cur = cur + 1 if b == a + 1 else 1
        best = max(best, cur)
    return best


def count_windows(frames_set, clip_len=CLIP_LEN, stride=STRIDE):
    fs = sorted(frames_set)
    if not fs:
        return 0
    n = 0
    for start in range(fs[0], fs[-1] - clip_len + 2, stride):
        if all((start + k) in frames_set for k in range(clip_len)):
            n += 1
    return n


def decompose_runs(per_track):
    """Pecah tiap ID menjadi run-run konsekutif.

    Kembalikan list run: {orig_id, start, end, frames: {frame: bbox},
    scores: {frame: score}}.
    """
    runs = []
    for tid in sorted(per_track):
        by_frame = per_track[tid]
        frames = sorted(by_frame)
        chunk = [frames[0]]
        prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                chunk.append(f)
            else:
                runs.append(_make_run(tid, chunk, by_frame))
                chunk = [f]
            prev = f
        runs.append(_make_run(tid, chunk, by_frame))
    runs.sort(key=lambda r: (r["start"], r["orig_id"]))
    return runs


def _make_run(tid, frames, by_frame):
    return {
        "orig_id": tid,
        "start": frames[0],
        "end": frames[-1],
        "frames": {f: by_frame[f][0] for f in frames},
        "scores": {f: by_frame[f][1] for f in frames},
    }


def load_tracks_raw(tracks_path):
    """Baca tracks.json; kembalikan (data mentah, per_track {id: {frame: (bbox, score)}})."""
    with open(tracks_path) as f:
        data = json.load(f)
    per_track = {}
    for fr in data["frames"]:
        fi = int(fr["frame"])
        for t in fr["tracks"]:
            bbox = tuple(float(v) for v in t["bbox"])
            per_track.setdefault(int(t["track_id"]), {})[fi] = (
                bbox, float(t.get("score", 1.0)))
    return data, per_track


def candidate_edges(runs, max_gap, max_dist_ref, min_ar, max_ar):
    """Semua pasangan run (A -> B) yang lolos ambang jarak/ukuran.

    Kembalikan list edge dict, diurutkan berdasarkan (dist_ref, gap).
    """
    edges = []
    for i, a in enumerate(runs):
        ca = center(a["frames"][a["end"]])
        sa = bbox_size(a["frames"][a["end"]])
        area_a = ((a["frames"][a["end"]][2] - a["frames"][a["end"]][0])
                  * (a["frames"][a["end"]][3] - a["frames"][a["end"]][1]))
        for j, b in enumerate(runs):
            if a["orig_id"] == b["orig_id"] or i == j:
                continue
            gap = b["start"] - a["end"] - 1
            if gap < 0 or gap > max_gap:
                continue
            cb = center(b["frames"][b["start"]])
            sb = bbox_size(b["frames"][b["start"]])
            dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            ref = max(sa, sb)
            dist_ref = dist / ref if ref > 0 else float("inf")
            area_b = ((b["frames"][b["start"]][2] - b["frames"][b["start"]][0])
                      * (b["frames"][b["start"]][3] - b["frames"][b["start"]][1]))
            ar = area_b / area_a if area_a > 0 else float("inf")
            if dist_ref > max_dist_ref or not (min_ar <= ar <= max_ar):
                continue
            edges.append({
                "from_run": i, "to_run": j,
                "from_id": a["orig_id"], "to_id": b["orig_id"],
                "from_span": [a["start"], a["end"]],
                "to_span": [b["start"], b["end"]],
                "gap": gap, "dist_px": round(dist, 1),
                "dist_ref": round(dist_ref, 3),
                "area_ratio": round(ar, 2),
            })
    edges.sort(key=lambda e: (e["dist_ref"], e["gap"]))
    return edges


def apply_ratio_test(edges, ratio_test):
    """Tolak semua edge keluar dari satu run sumber bila dua kandidat
    terbaiknya hampir sama dekat (ambigu). Kembalikan (lolos, ditolak)."""
    by_source = {}
    for e in edges:
        by_source.setdefault(e["from_run"], []).append(e)
    kept, rejected = [], []
    for src, es in by_source.items():
        es.sort(key=lambda e: e["dist_ref"])
        if len(es) > 1 and es[0]["dist_ref"] >= ratio_test * es[1]["dist_ref"]:
            for e in es:
                e["reject_reason"] = (
                    f"ambigu: kandidat terbaik dist_ref={es[0]['dist_ref']} "
                    f"vs kedua {es[1]['dist_ref']} (rasio "
                    f"{es[0]['dist_ref'] / es[1]['dist_ref']:.2f} >= {ratio_test})")
                rejected.append(e)
        else:
            kept.extend(es)
    kept.sort(key=lambda e: (e["dist_ref"], e["gap"]))
    return kept, rejected


class Components:
    """Union-find atas run. AWAL: tiap komponen = satu ID utuh (semua
    run dari ID itu sudah digabung) — komponen tidak pernah terpecah,
    hanya bergabung, sehingga merge tidak pernah mengurangi frame
    berlabel. Frame tiap komponen dilacak untuk cek tumpang tindih."""

    def __init__(self, runs):
        self.parent = list(range(len(runs)))
        self.frames = {i: set(runs[i]["frames"]) for i in range(len(runs))}
        by_id = {}
        for i, r in enumerate(runs):
            by_id.setdefault(r["orig_id"], []).append(i)
        for members in by_id.values():
            first = members[0]
            for i in members[1:]:
                self.parent[i] = first
                self.frames[first] |= self.frames[i]
                del self.frames[i]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def can_merge(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False, "sudah satu komponen"
        if self.frames[ra] & self.frames[rb]:
            overlap = sorted(self.frames[ra] & self.frames[rb])
            return False, ("frame tumpang tindih dengan komponen target "
                           f"(frame {overlap[:5]}"
                           f"{'...' if len(overlap) > 5 else ''})")
        return True, ""

    def merge(self, a, b):
        ra, rb = self.find(a), self.find(b)
        self.parent[rb] = ra
        self.frames[ra] |= self.frames[rb]
        del self.frames[rb]

    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def greedy_merge(runs, edges):
    """Terima edge secara greedy (jarak terkecil dulu).

    Satu run maksimal satu pendahulu + satu penerus. Kembalikan
    (accepted, skipped, components).
    """
    comp = Components(runs)
    has_out, has_in = set(), set()
    accepted, skipped = [], []
    for e in edges:
        if e["from_run"] in has_out:
            e["reject_reason"] = "run sumber sudah punya penerus"
            skipped.append(e)
            continue
        if e["to_run"] in has_in:
            e["reject_reason"] = "run target sudah punya pendahulu"
            skipped.append(e)
            continue
        ok, reason = comp.can_merge(e["from_run"], e["to_run"])
        if not ok:
            e["reject_reason"] = reason
            skipped.append(e)
            continue
        comp.merge(e["from_run"], e["to_run"])
        has_out.add(e["from_run"])
        has_in.add(e["to_run"])
        accepted.append(e)
    return accepted, skipped, comp


def build_merged_tracks(raw_data, runs, accepted):
    """Bangun tracks.json hasil gabungan (format sama dengan input).

    ID gabungan = ID run paling awal dalam rantai. Dijamin: tiap
    (frame, track_id) muncul maksimal satu kali.
    Kembalikan (tracks gabungan, run_to_merged, renumbered).
    """
    comp = Components(runs)
    for e in accepted:
        comp.merge(e["from_run"], e["to_run"])
    groups = comp.groups()

    run_to_merged = {}
    renumbered = []
    max_id = max(r["orig_id"] for r in runs)
    next_fresh = max_id + 1
    used_ids = set()
    ordered = sorted(
        groups.values(),
        key=lambda ms: (min(runs[i]["start"] for i in ms),
                        min(runs[i]["orig_id"] for i in ms)))
    for members in ordered:
        earliest = min(members, key=lambda i: (runs[i]["start"], runs[i]["orig_id"]))
        preferred = runs[earliest]["orig_id"]
        if preferred in used_ids:
            # Tidak seharusnya terjadi (komponen berisi ID utuh), tapi
            # dijaga supaya output tetap valid.
            merged_id = next_fresh
            next_fresh += 1
            renumbered.append({
                "orig_id": preferred,
                "new_id": merged_id,
                "runs": [[runs[i]["start"], runs[i]["end"]] for i in members],
                "reason": "ID bentrok antar komponen (pengaman)",
            })
        else:
            merged_id = preferred
        used_ids.add(merged_id)
        for i in members:
            run_to_merged[i] = merged_id

    frame_owner = {}
    for i, r in enumerate(runs):
        for f in r["frames"]:
            frame_owner[(r["orig_id"], f)] = run_to_merged[i]

    out_frames = []
    for fr in raw_data["frames"]:
        fi = int(fr["frame"])
        tracks = []
        for t in fr["tracks"]:
            mid = frame_owner.get((int(t["track_id"]), fi), int(t["track_id"]))
            tracks.append({"track_id": mid, "bbox": t["bbox"],
                           "score": t.get("score", 1.0)})
        tracks.sort(key=lambda t: t["track_id"])
        seen = [t["track_id"] for t in tracks]
        assert len(seen) == len(set(seen)), \
            f"ID ganda di frame {fi}: {seen}"
        out_frames.append({"frame": fi, "name": fr.get("name", f"frame_{fi:04d}"),
                           "tracks": tracks})
    out = {"fps": raw_data.get("fps", 10.0),
           "score_threshold": raw_data.get("score_threshold"),
           "frames": out_frames}
    out = {k: v for k, v in out.items() if v is not None}
    return out, run_to_merged, renumbered


def labeled_frames(per_track):
    """Jumlah box-frame berlabel = frame milik track yang punya >=1
    jendela klip (render memakai label mayoritas track untuk frame
    yang tidak tercakup klip). Kembalikan (berlabel, putih)."""
    lab = white = 0
    for fs in per_track.values():
        if longest_run(set(fs)) >= CLIP_LEN:
            lab += len(fs)
        else:
            white += len(fs)
    return lab, white


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True, help="tracks.json Stage 1.")
    ap.add_argument("--out", required=True, help="tracks.json hasil gabungan.")
    ap.add_argument("--report", required=True, help="Laporan JSON.")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    ap.add_argument("--max-dist-ref", type=float, default=DEFAULT_MAX_DIST_REF)
    ap.add_argument("--min-area-ratio", type=float, default=DEFAULT_MIN_AREA_RATIO)
    ap.add_argument("--max-area-ratio", type=float, default=DEFAULT_MAX_AREA_RATIO)
    ap.add_argument("--ratio-test", type=float, default=DEFAULT_RATIO_TEST)
    args = ap.parse_args()

    raw_data, per_track = load_tracks_raw(args.tracks)
    n_ids_before = len(per_track)
    runs = decompose_runs(per_track)
    n_runs = len(runs)
    n_runs_ge16 = sum(1 for r in runs if r["end"] - r["start"] + 1 >= 16)
    classified_before = {tid for tid, byf in per_track.items()
                         if longest_run(set(byf)) >= CLIP_LEN}
    lab_before, white_before = labeled_frames(
        {tid: set(byf) for tid, byf in per_track.items()})

    edges = candidate_edges(runs, args.max_gap, args.max_dist_ref,
                            args.min_area_ratio, args.max_area_ratio)
    kept, rejected_amb = apply_ratio_test(edges, args.ratio_test)
    accepted, skipped, _ = greedy_merge(runs, kept)

    merged, run_to_merged, renumbered = build_merged_tracks(
        raw_data, runs, accepted)
    per_after = {}
    for fr in merged["frames"]:
        for t in fr["tracks"]:
            per_after.setdefault(int(t["track_id"]), set()).add(int(fr["frame"]))
    n_ids_after = len(per_after)
    n_ids_ge16_after = sum(1 for fs in per_after.values()
                           if longest_run(fs) >= CLIP_LEN)
    lab_after, white_after = labeled_frames(per_after)

    if lab_after < lab_before:
        print(f"GAGAL: box-frame berlabel turun {lab_before} -> {lab_after} "
              f"(seharusnya tidak pernah turun). Periksa ambang/edge.",
              file=sys.stderr)
        return 1

    def total_windows(per):
        return sum(count_windows(set(fs)) for fs in per.values())
    clips_before = total_windows(
        {tid: set(byf) for tid, byf in per_track.items()})
    clips_after = total_windows(per_after)

    def edge_dict(e):
        return {"from_id": e["from_id"], "to_id": e["to_id"],
                "from_span": e["from_span"], "to_span": e["to_span"],
                "gap": e["gap"], "dist_px": e["dist_px"],
                "dist_ref": e["dist_ref"], "area_ratio": e["area_ratio"],
                "reason": e.get("reject_reason", "")}

    report = {
        "input": os.path.basename(args.tracks),
        "params": {"max_gap": args.max_gap,
                   "max_dist_ref": args.max_dist_ref,
                   "min_area_ratio": args.min_area_ratio,
                   "max_area_ratio": args.max_area_ratio,
                   "ratio_test": args.ratio_test},
        "summary": {
            "n_ids_before": n_ids_before,
            "n_ids_after": n_ids_after,
            "n_runs": n_runs,
            "n_runs_ge16": n_runs_ge16,
            "n_ids_with_clip_window_before": len(classified_before),
            "n_ids_with_clip_window_after": n_ids_ge16_after,
            "n_edges_candidate": len(edges),
            "n_edges_accepted": len(accepted),
            "n_edges_rejected_ambiguous": len(rejected_amb),
            "n_edges_skipped_conflict": len(skipped),
            "n_ids_renumbered": len(renumbered),
            "clip_windows_before": clips_before,
            "clip_windows_after": clips_after,
            "labeled_box_frames_before": lab_before,
            "labeled_box_frames_after": lab_after,
            "white_box_frames_before": white_before,
            "white_box_frames_after": white_after,
        },
        "accepted_merges": [edge_dict(e) for e in accepted],
        "rejected_ambiguous": [edge_dict(e) for e in rejected_amb],
        "skipped_edges": [edge_dict(e) for e in skipped],
        "renumbered_runs": renumbered,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    print(f"ID: {s['n_ids_before']} -> {s['n_ids_after']} "
          f"(ID dg jendela klip: {s['n_ids_with_clip_window_before']} -> "
          f"{s['n_ids_with_clip_window_after']})")
    print(f"Run: {s['n_runs']} (run >=16 frame: {s['n_runs_ge16']})")
    print(f"Edge kandidat: {s['n_edges_candidate']}, diterima: "
          f"{s['n_edges_accepted']}, ditolak ambigu: "
          f"{s['n_edges_rejected_ambiguous']}, dilewati (konflik): "
          f"{s['n_edges_skipped_conflict']}")
    print(f"Jendela klip 16 frame (stride 8): {s['clip_windows_before']} -> "
          f"{s['clip_windows_after']}")
    print(f"Box-frame berlabel: {s['labeled_box_frames_before']} -> "
          f"{s['labeled_box_frames_after']} "
          f"(putih: {s['white_box_frames_before']} -> "
          f"{s['white_box_frames_after']})")
    for e in accepted:
        print(f"  gabung ID {e['from_id']} {e['from_span']} -> ID {e['to_id']} "
              f"{e['to_span']}: gap={e['gap']} dist={e['dist_px']}px "
              f"dist_ref={e['dist_ref']} ar={e['area_ratio']}")
    print(f"Keluar: {args.out}\nLaporan: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
