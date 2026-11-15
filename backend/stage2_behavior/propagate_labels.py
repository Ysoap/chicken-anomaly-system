"""Label propagation (label warisan) di atas stitching method 2.

Membaca ``merge_report.json`` + ``track_predictions.json`` hasil merge
(``infer_fcam16_merged/``) + ``track_predictions.json`` ASLI
(``infer_fcam16/``), lalu menentukan fragmen mana yang tadinya putih
(tanpa klip/label) dan kini mendapat label lewat sambungan merge.

Untuk tiap pasangan merge yang diterima:
- sisi yang TIDAK berlabel di prediksi ASLI mewarisi label mayoritas
  track gabungannya (dari prediksi hasil merge);
- sisi yang SUDAH berlabel di prediksi asli dilewati — tidak ada yang
  baru diwariskan, labelnya sudah ada sebelum merge.

Hasil: ``track_predictions_inherited.json`` — file DELTA (bukan
pengganti ``track_predictions.json``): daftar label warisan + merge
yang dilewati + hitungan + peringatan display-only. Label prediksi
model tidak diubah sama sekali.

Dua jaminan yang dicek keras oleh script ini (gagal keras bila
dilanggar):
1. tiap frame warisan memang PUTIH di render asli (tidak ada label);
2. jumlah box-frame warisan sama persis dengan selisih box-frame
   berlabel di ``merge_report.json`` (sesudah merge - sebelum merge).

    python propagate_labels.py --merge-report \\
        results/infer_fcam16_merged/merge_report.json \\
        --predictions-merged results/infer_fcam16_merged/track_predictions.json \\
        --predictions-original results/infer_fcam16/track_predictions.json \\
        --tracks-original /path/ke/tracks_fCam16.json \\
        --tracks-merged results/infer_fcam16_merged/merged_tracks_fCam16.json \\
        --out results/infer_fcam16_merged/track_predictions_inherited.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_builder import load_tracks  # noqa: E402
from render_annotated_video import label_for_frame  # noqa: E402


def merged_id_for_merge(merge, orig_per_track, merged_per_track):
    """ID track gabungan yang menampung fragmen merge ini.

    Dicari lewat kecocokan bbox: bbox fragmen 'from' di frame awalnya
    harus sama persis dengan bbox track gabungan di frame yang sama
    (script merge menyalin bbox apa adanya). Dicek silang dengan
    fragmen 'to' juga.
    """
    from_id, to_id = merge["from_id"], merge["to_id"]
    f0 = merge["from_span"][0]
    ref_bbox = orig_per_track[from_id][f0]
    cands = [tid for tid, by_frame in merged_per_track.items()
             if by_frame.get(f0) is not None
             and list(by_frame[f0]) == list(ref_bbox)]
    if len(cands) != 1:
        raise RuntimeError(
            f"merge {from_id}->{to_id}: {len(cands)} kandidat ID gabungan "
            f"untuk frame {f0} (diharapkan tepat 1): {cands}")
    mid = cands[0]
    t0 = merge["to_span"][0]
    ref_to = orig_per_track[to_id][t0]
    if list(merged_per_track[mid].get(t0, [])) != list(ref_to):
        raise RuntimeError(
            f"merge {from_id}->{to_id}: ID gabungan {mid} tidak memuat "
            f"bbox fragmen 'to' di frame {t0} — cek silang gagal")
    return mid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge-report", required=True)
    ap.add_argument("--predictions-merged", required=True)
    ap.add_argument("--predictions-original", required=True)
    ap.add_argument("--tracks-original", required=True)
    ap.add_argument("--tracks-merged", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.merge_report) as f:
        report = json.load(f)
    with open(args.predictions_merged) as f:
        pred_merged = json.load(f)
    with open(args.predictions_original) as f:
        pred_orig = json.load(f)

    _, orig_per_track, _ = load_tracks(args.tracks_original)
    _, merged_per_track, _ = load_tracks(args.tracks_merged)

    merged_summary = pred_merged["track_summary"]
    orig_summary = pred_orig["track_summary"]
    merged_clips = pred_merged["clip_results"]
    orig_clips = pred_orig["clip_results"]

    inherited = []
    skipped = []
    for m in report["accepted_merges"]:
        from_id, to_id = m["from_id"], m["to_id"]
        from_labeled = str(from_id) in orig_summary
        to_labeled = str(to_id) in orig_summary
        mid = merged_id_for_merge(m, orig_per_track, merged_per_track)
        if from_labeled and to_labeled:
            skipped.append({
                "from_id": from_id, "to_id": to_id,
                "merged_track_id": mid,
                "reason": "kedua fragmen sudah berlabel sebelum merge; "
                          "tidak ada yang baru diwariskan",
            })
            continue
        if not from_labeled and not to_labeled:
            skipped.append({
                "from_id": from_id, "to_id": to_id,
                "merged_track_id": mid,
                "reason": "kedua fragmen tidak berlabel sebelum merge; "
                          "tidak ada sumber label untuk diwariskan",
            })
            continue
        # Tepat satu sisi berlabel: sisi yang tidak berlabel mewarisi.
        if not from_labeled:
            unlabeled_id, labeled_id = from_id, to_id
        else:
            unlabeled_id, labeled_id = to_id, from_id
        label = merged_summary[str(mid)]["majority_label"]
        frames = sorted(orig_per_track[unlabeled_id].keys())
        # Jaminan 1: tiap frame warisan memang putih di render asli.
        for fr in frames:
            lab, _, _ = label_for_frame(
                unlabeled_id, fr, orig_clips, orig_summary)
            if lab is not None:
                raise RuntimeError(
                    f"fragmen {unlabeled_id} frame {fr} sudah berlabel "
                    f"'{lab}' sebelum merge — bukan warisan")
        # Jaminan 2 (per frame): render hasil merge memang memberi
        # label yang sama lewat fallback track gabungan.
        for fr in frames:
            lab, _, src = label_for_frame(
                mid, fr, merged_clips, merged_summary)
            if lab != label:
                raise RuntimeError(
                    f"track gabungan {mid} frame {fr}: label fallback "
                    f"'{lab}' (sumber {src}) != label warisan '{label}'")
        inherited.append({
            "merged_track_id": mid,
            "fragment_id": unlabeled_id,
            "frames": frames,
            "label": label,
            "inherited": True,
            "inherited_from": labeled_id,
            "source": f"label mayoritas track gabungan {mid} "
                      f"({merged_summary[str(mid)]['label_counts']})",
            "note": f"fragmen {unlabeled_id} ({len(frames)} frame) tidak "
                    f"punya klip sebelum merge; label diwarisi dari "
                    f"fragmen {labeled_id} yang disambung dengannya",
        })

    n_frames = sum(len(e["frames"]) for e in inherited)
    summ = report["summary"]
    delta = summ["labeled_box_frames_after"] - summ["labeled_box_frames_before"]
    # Jaminan 3: jumlah box-frame warisan = selisih merge_report.
    if n_frames != delta:
        raise RuntimeError(
            f"jumlah box-frame warisan {n_frames} != selisih "
            f"merge_report {delta} "
            f"({summ['labeled_box_frames_after']} - "
            f"{summ['labeled_box_frames_before']})")

    out = {
        "mode": "track_inherited",
        "display_only": True,
        "warning": "Label warisan HANYA untuk tampilan video demo — "
                   "BUKAN prediksi model. Label ini copy dari label "
                   "mayoritas track gabungannya dan bisa BASI (perilaku "
                   "ayam bisa berubah selama di luar frame) atau SALAH "
                   "bila sambungan merge-nya keliru. Jangan dipakai "
                   "mentah-mentah di Stage 3 (aturan durasi); bila "
                   "dipakai, tandai dan bobot rendah.",
        "method": "Label propagation di atas stitching method 2 "
                  "(merge_track_fragments.py): fragmen tanpa klip "
                  "mewarisi label mayoritas track gabungan yang "
                  "disambung dengannya. Hanya merge yang diterima "
                  "yang dipakai; tidak ada label baru yang dikarang.",
        "based_on": {
            "predictions_merged": args.predictions_merged,
            "merge_report": args.merge_report,
            "predictions_original": args.predictions_original,
            "video": "fCam16_sample.mp4",
        },
        "inherited_labels": inherited,
        "skipped_merges": skipped,
        "counts": {
            "n_merges_accepted": len(report["accepted_merges"]),
            "n_fragments_inherited": len(inherited),
            "n_box_frames_inherited": n_frames,
            "n_merges_skipped": len(skipped),
            "white_box_frames_before_merge":
                summ["white_box_frames_before"],
            "white_box_frames_after_merge":
                summ["white_box_frames_after"],
            "white_box_frames_after_inheritance":
                summ["white_box_frames_after"] - n_frames,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Fragmen diwarisi: {len(inherited)} "
          f"({', '.join(str(e['fragment_id']) for e in inherited)})")
    print(f"Box-frame warisan: {n_frames} "
          f"(cocok dengan selisih merge_report: {delta})")
    print(f"Kotak putih tersisa: "
          f"{summ['white_box_frames_after']} -> "
          f"{summ['white_box_frames_after'] - n_frames}")
    print(f"Tersimpan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
