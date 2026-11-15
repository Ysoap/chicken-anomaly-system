"""Stage 3 inference: score tracking data or feature sequences for anomalies.

Two ways to use this module:

1. As a library (this is what the integration pipeline should do)::

       from backend.stage3_anomaly.inference import Stage3Scorer
       scorer = Stage3Scorer("backend/stage3_anomaly/models/stage3_lstm_ae.pt")
       results = scorer.score_frames(tracking_records)   # Stage 1 + 2 output
       results = scorer.score_sequences(feature_array)   # precomputed features

   ``score_frames`` runs the feature extractor and then scores every
   window. At 30 FPS with the default window of 32 frames and stride 16,
   one scoring pass over the newest window per track is cheap enough to
   run about once per second per track on CPU.

2. As a CLI::

       python backend/stage3_anomaly/inference.py \
           --checkpoint backend/stage3_anomaly/models/stage3_lstm_ae.pt \
           --input tracking.json

   Input may be a .npy file of feature sequences, a .json file holding a
   list of tracking records (or an object with a "records", "frames" or
   "sequences" key), or a .csv file. A CSV with a ``track_id`` column is
   read as tracking records; any other CSV is read as a numeric feature
   stream and cut into windows. Output is JSON with one entry per window:
   its score, the stored threshold, and an ``is_anomaly`` flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from feature_extractor import extract_sequences  # noqa: E402
from model import LSTMAutoencoder  # noqa: E402


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> Tuple[LSTMAutoencoder, Dict[str, Any]]:
    """Load a checkpoint written by train.py and rebuild the model."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LSTMAutoencoder(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


class Stage3Scorer:
    """Loads a Stage 3 checkpoint once and scores windows against it."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, device=device)
        self.threshold = float(self.checkpoint["threshold"])
        self.window = int(self.checkpoint.get("window", 32))
        self.norm_mean = np.asarray(self.checkpoint["norm_mean"], dtype=np.float32)
        self.norm_std = np.asarray(self.checkpoint["norm_std"], dtype=np.float32)

    def _normalise(self, sequences: np.ndarray) -> np.ndarray:
        return (sequences - self.norm_mean) / self.norm_std

    def score_sequences(self, sequences: np.ndarray, batch_size: int = 256) -> List[Dict[str, Any]]:
        """Score precomputed feature sequences, shape (num, time, features)."""
        sequences = np.asarray(sequences, dtype=np.float32)
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (num, time, features)")
        if sequences.shape[2] != self.model.input_dim:
            raise ValueError(
                "expected %d features, got %d" % (self.model.input_dim, sequences.shape[2])
            )
        if sequences.shape[0] == 0:
            return []
        normalised = self._normalise(sequences)
        scores: List[np.ndarray] = []
        for i in range(0, normalised.shape[0], batch_size):
            chunk = torch.as_tensor(normalised[i : i + batch_size], device=self.device)
            scores.append(self.model.score(chunk).cpu().numpy())
        flat = np.concatenate(scores)
        return [
            {
                "index": int(i),
                "score": float(flat[i]),
                "is_anomaly": bool(flat[i] > self.threshold),
            }
            for i in range(flat.shape[0])
        ]

    def score_frames(
        self,
        frames: Sequence[Any],
        window: Optional[int] = None,
        stride: Optional[int] = None,
        frame_size: Tuple[float, float] = (640, 480),
    ) -> List[Dict[str, Any]]:
        """Score raw tracking records from Stage 1 (plus Stage 2 labels).

        Returns one entry per extracted window, including the track id and
        the frame range the window covers.
        """
        window = window or self.window
        stride = stride or max(1, window // 2)
        sequences, meta = extract_sequences(
            frames, window=window, stride=stride, frame_size=frame_size
        )
        scored = self.score_sequences(sequences)
        results: List[Dict[str, Any]] = []
        for entry, window_meta in zip(scored, meta["windows"]):
            merged = dict(entry)
            merged["track_id"] = window_meta["track_id"]
            merged["start_frame"] = window_meta["start_frame"]
            merged["end_frame"] = window_meta["end_frame"]
            results.append(merged)
        return results


def _read_tracking_csv(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            behavior = row.get("behavior") or None
            records.append(
                {
                    "frame": int(float(row["frame"])),
                    "track_id": row["track_id"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                    "behavior": behavior,
                }
            )
    return records


def _load_input(path: Path, window: int) -> Tuple[str, Any]:
    """Return ("sequences", array) or ("frames", records) for a CLI input."""
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            return "sequences", arr
        if arr.ndim == 2:
            n = arr.shape[0] // window
            return "sequences", arr[: n * window].reshape(n, window, arr.shape[1])
        raise ValueError("%s: expected a 2-D or 3-D array" % path)

    if path.suffix == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "sequences" in payload:
                return "sequences", np.asarray(payload["sequences"], dtype=np.float32)
            for key in ("records", "frames", "tracking"):
                if key in payload:
                    return "frames", payload[key]
            raise ValueError("%s: JSON object needs a records, frames or sequences key" % path)
        if isinstance(payload, list):
            return "frames", payload
        raise ValueError("%s: unsupported JSON payload" % path)

    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            header = handle.readline()
        if "track_id" in header:
            return "frames", _read_tracking_csv(path)
        from train import load_sequences  # local import: numeric feature CSV

        return "sequences", load_sequences([str(path)], window=window)

    raise ValueError("%s: unsupported input type, use .npy, .json or .csv" % path)


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    scorer = Stage3Scorer(args.checkpoint, device=args.device)
    kind, payload = _load_input(Path(args.input), window=scorer.window)
    if kind == "sequences":
        results = scorer.score_sequences(payload)
    else:
        results = scorer.score_frames(
            payload,
            window=args.window,
            stride=args.stride,
            frame_size=(args.frame_width, args.frame_height),
        )
    output = {
        "checkpoint": scorer.checkpoint_path,
        "threshold": scorer.threshold,
        "num_windows": len(results),
        "num_anomalies": sum(1 for r in results if r["is_anomaly"]),
        "results": results,
    }
    text = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print("wrote %s (%d windows, %d anomalies)" % (args.output, output["num_windows"], output["num_anomalies"]))
    else:
        print(text)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3 anomaly scoring.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".npy, .json or .csv input")
    parser.add_argument("--output", default=None, help="write JSON here instead of stdout")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--frame-width", type=float, default=640)
    parser.add_argument("--frame-height", type=float, default=480)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: List[str] | None = None) -> None:
    run_cli(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
