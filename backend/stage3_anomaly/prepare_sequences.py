"""Prepare Stage 3 training sequences from real Stage 1 and Stage 2 outputs.

Data sources (downloaded from the repo, kept next to this script in data/):

* ``data/merged_tracks.json``: merged_tracks_fCam16.json from branch
  stage2/behavior-setup, backend/stage2_behavior/results/infer_fcam16_merged/.
  Per-frame tracks in the same layout as the Stage 1 output: a dict with
  ``fps`` and ``frames``, each frame holding ``tracks`` entries with
  ``track_id`` and ``bbox`` in xyxy pixel coordinates.
* ``data/track_predictions.json``: track_predictions.json from the same
  folder. Stage 2 VideoMAE predictions as 120 clips, each with
  ``track_id``, ``start_frame``, ``end_frame``, ``label`` and
  ``confidence``. Labels are RESTING, ACTIVE and INGESTIVE.
* ``data/tracks.json``: raw Stage 1 output from branch
  stage1/detection-setup, backend/stage1_detection/results/tracks.json.
  Inspected for comparison only (see TRAINING_NOTES.md): the Stage 2 clips
  were computed on the merged track set, which a coverage check confirmed
  (120 of 120 clips match merged tracks over more than 90 percent of their
  frames, versus 24 of 120 for the raw Stage 1 tracks), so the merged
  tracks are the correct pairing for the labels.

Behavior label mapping
----------------------
Clips are 16 frames long with a stride of 8, so clips overlap. For every
(track, frame) pair, the label is taken from the covering clip with the
highest confidence; ties are broken by the clip whose center is closest
to the frame. Frames not covered by any clip of their track get
``behavior = None`` in the record; the feature extractor forward-fills
labels inside a continuous track segment, and behavior features stay 0
until the first label appears.

Frame size
----------
The Stage 2 README states the bounding boxes live in a 1400x900
coordinate space (scaled to 960x618 for the annotated preview video).
The observed maxima (x2 = 1401, y2 = 901 over all boxes) agree with
that, so ``frame_size`` is (1400, 900) here, not the extractor default.

Outputs (written next to this script):
    sequences_all.npy, train_sequences.npy, val_sequences.npy,
    prep_meta.json

The train and validation split repeats exactly what train.py does with
its defaults (numpy default_rng(42) permutation, val fraction 0.15), so
the saved split arrays are the same split train.py uses internally when
it is run on sequences_all.npy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STAGE3_DIR = HERE.parent / "backend" / "stage3_anomaly"
sys.path.insert(0, str(STAGE3_DIR))

from feature_extractor import extract_sequences  # noqa: E402

WINDOW = 32
STRIDE = 16
FRAME_SIZE = (1400, 900)
SEED = 42
VAL_FRAC = 0.15


def build_label_lookup(preds: Dict[str, Any]) -> Dict[Any, List[Dict[str, Any]]]:
    """Group Stage 2 clips by track id."""
    by_track: Dict[Any, List[Dict[str, Any]]] = {}
    for clip in preds["clip_results"]:
        by_track.setdefault(clip["track_id"], []).append(clip)
    return by_track


def label_for(clips: List[Dict[str, Any]], frame: int) -> Optional[str]:
    """Highest-confidence clip covering ``frame``; nearest center breaks ties."""
    covering = [c for c in clips if c["start_frame"] <= frame <= c["end_frame"]]
    if not covering:
        return None
    best = max(
        covering,
        key=lambda c: (c["confidence"], -abs((c["start_frame"] + c["end_frame"]) / 2.0 - frame)),
    )
    return best["label"]


def build_records(
    tracks_payload: Dict[str, Any],
    clips_by_track: Dict[Any, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Convert xyxy track frames into extractor records with behavior labels."""
    records: List[Dict[str, Any]] = []
    labeled = 0
    for frame_entry in tracks_payload["frames"]:
        frame_idx = int(frame_entry["frame"])
        for track in frame_entry["tracks"]:
            x1, y1, x2, y2 = (float(v) for v in track["bbox"])
            behavior = label_for(clips_by_track.get(track["track_id"], []), frame_idx)
            if behavior is not None:
                labeled += 1
            records.append(
                {
                    "frame": frame_idx,
                    "track_id": track["track_id"],
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "behavior": behavior,
                }
            )
    print("records: %d built, %d with a behavior label" % (len(records), labeled))
    return records


def split_like_train_py(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the train.py default split (seed 42, val fraction 0.15)."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(sequences.shape[0])
    num_val = max(1, int(round(sequences.shape[0] * VAL_FRAC)))
    num_val = min(num_val, sequences.shape[0] - 1)
    val = sequences[order[:num_val]]
    train = sequences[order[num_val:]]
    return train, val


def main() -> None:
    merged = json.loads((DATA / "merged_tracks.json").read_text())
    raw = json.loads((DATA / "tracks.json").read_text())
    preds = json.loads((DATA / "track_predictions.json").read_text())

    merged_track_ids = {
        t["track_id"] for f in merged["frames"] for t in f["tracks"]
    }
    raw_track_ids = {t["track_id"] for f in raw["frames"] for t in f["tracks"]}
    frame_indices = [f["frame"] for f in merged["frames"]]
    print(
        "merged tracks: %d frames (%d to %d), %d distinct tracks, fps %s"
        % (len(merged["frames"]), min(frame_indices), max(frame_indices),
           len(merged_track_ids), merged.get("fps"))
    )
    print(
        "raw stage 1 tracks (comparison only): %d frames, %d distinct tracks"
        % (len(raw["frames"]), len(raw_track_ids))
    )
    print(
        "stage 2 predictions: %d clips over %d tracks, labels %s"
        % (
            preds["n_clips"],
            len({c["track_id"] for c in preds["clip_results"]}),
            preds["label_distribution"],
        )
    )

    clips_by_track = build_label_lookup(preds)
    records = build_records(merged, clips_by_track)

    sequences, meta = extract_sequences(
        records, window=WINDOW, stride=STRIDE, frame_size=FRAME_SIZE
    )
    print(
        "sequences: shape %s, behavior_available=%s"
        % (sequences.shape, meta["behavior_available"])
    )

    train, val = split_like_train_py(sequences)
    np.save(HERE / "sequences_all.npy", sequences)
    np.save(HERE / "train_sequences.npy", train)
    np.save(HERE / "val_sequences.npy", val)
    print("split: %d train, %d validation" % (len(train), len(val)))

    window_tracks = sorted({w["track_id"] for w in meta["windows"]}, key=str)
    prep_meta = {
        "window": WINDOW,
        "stride": STRIDE,
        "frame_size": list(FRAME_SIZE),
        "fps": merged.get("fps"),
        "source_frames": len(merged["frames"]),
        "frame_index_range": [int(min(frame_indices)), int(max(frame_indices))],
        "source_distinct_tracks_merged": len(merged_track_ids),
        "source_distinct_tracks_raw_stage1": len(raw_track_ids),
        "num_records": len(records),
        "num_sequences": int(sequences.shape[0]),
        "sequence_shape": list(sequences.shape),
        "tracks_contributing_windows": [str(t) for t in window_tracks],
        "num_tracks_contributing_windows": len(window_tracks),
        "num_train": int(len(train)),
        "num_val": int(len(val)),
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "behavior_available": bool(meta["behavior_available"]),
        "feature_names": meta["feature_names"],
    }
    (HERE / "prep_meta.json").write_text(json.dumps(prep_meta, indent=2))
    print("wrote sequences_all.npy, train_sequences.npy, val_sequences.npy, prep_meta.json")


if __name__ == "__main__":
    main()
