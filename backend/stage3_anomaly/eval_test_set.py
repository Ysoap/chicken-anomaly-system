"""Evaluate Stage 3 on a labeled test set.

Usage example:
    python backend/stage3_anomaly/eval_test_set.py \
        --checkpoint backend/stage3_anomaly/models/stage3_lstm_ae.pt \
        --data data/stage3/test_set.csv

Accepted data formats
---------------------
* CSV, long format: one row per timestep with columns ``seq_id`` and
  ``label`` (0 normal, 1 anomaly), an optional ``t`` column, and one
  column per feature. Rows are grouped by ``seq_id``.
* CSV, wide format: one row per sequence with a ``label`` column and
  window times feature_dim numeric feature columns (the flattened
  sequence, timestep-major).
* NPY: an array of shape (num_sequences, window, feature_dim) plus a
  ``--labels`` file, either a .npy array of 0/1 labels or a .csv with a
  ``label`` column.

The report compares the checkpoint threshold against the pipeline
targets from the project README: anomaly recall above 90 percent and a
false positive rate below 5 percent.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import Stage3Scorer  # noqa: E402

TARGET_RECALL = 0.90
TARGET_FPR = 0.05


def _load_csv_sequences(path: Path, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "label" not in fieldnames:
        raise ValueError("%s: a label column is required" % path)

    if "seq_id" in fieldnames:
        feature_cols = [c for c in fieldnames if c not in ("seq_id", "t", "label")]
        groups: dict[str, list[dict]] = {}
        order: List[str] = []
        for row in rows:
            key = row["seq_id"]
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        for key in order:
            group = groups[key]
            if "t" in fieldnames:
                group = sorted(group, key=lambda r: float(r["t"]))
            sequences.append(
                np.array([[float(r[c]) for c in feature_cols] for r in group], dtype=np.float32)
            )
            labels.append(int(float(group[0]["label"])))
        min_len = min(s.shape[0] for s in sequences)
        seq_array = np.stack([s[:min_len] for s in sequences])
        return seq_array, np.asarray(labels, dtype=np.int64)

    feature_cols = [c for c in fieldnames if c != "label"]
    if len(feature_cols) % input_dim != 0:
        raise ValueError(
            "%s: %d feature columns is not a multiple of the model input_dim %d"
            % (path, len(feature_cols), input_dim)
        )
    steps = len(feature_cols) // input_dim
    sequences = []
    labels = []
    for row in rows:
        values = np.array([float(row[c]) for c in feature_cols], dtype=np.float32)
        sequences.append(values.reshape(steps, input_dim))
        labels.append(int(float(row["label"])))
    return np.stack(sequences), np.asarray(labels, dtype=np.int64)


def _load_labels(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.int64).ravel()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "label" in reader.fieldnames:
            return np.array([int(float(r["label"])) for r in reader], dtype=np.int64)
    values = np.genfromtxt(path, delimiter=",")
    return np.atleast_1d(values).astype(np.int64).ravel()


def load_test_set(args: argparse.Namespace, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(args.data)
    if path.suffix == ".npy":
        sequences = np.load(path).astype(np.float32)
        if sequences.ndim != 3:
            raise ValueError("%s: expected shape (num, window, features)" % path)
        if not args.labels:
            raise ValueError("--labels is required when --data is a .npy file")
        labels = _load_labels(Path(args.labels))
    elif path.suffix == ".csv":
        sequences, labels = _load_csv_sequences(path, input_dim)
    else:
        raise ValueError("%s: unsupported file type, use .npy or .csv" % path)
    if sequences.shape[0] != labels.shape[0]:
        raise ValueError(
            "sequence count %d does not match label count %d"
            % (sequences.shape[0], labels.shape[0])
        )
    return sequences, labels


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = (scores > threshold).astype(np.int64)
    tp = int(np.sum((predicted == 1) & (labels == 1)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    tn = int(np.sum((predicted == 0) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "fpr": fpr, "accuracy": accuracy,
    }


def _fmt(value: float) -> str:
    return "n/a" if value != value else "%.3f" % value  # NaN check


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 on a labeled test set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="labeled .csv or .npy test set")
    parser.add_argument("--labels", default=None, help="labels file for .npy data")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    scorer = Stage3Scorer(args.checkpoint, device=args.device)
    sequences, labels = load_test_set(args, scorer.model.input_dim)
    scored = scorer.score_sequences(sequences)
    scores = np.array([entry["score"] for entry in scored], dtype=np.float64)
    metrics = compute_metrics(scores, labels, scorer.threshold)

    print("test set: %d sequences (%d normal, %d anomaly)" % (
        len(labels), int(np.sum(labels == 0)), int(np.sum(labels == 1))))
    print("threshold: %.6f" % scorer.threshold)
    print("")
    print("%-22s %-8s %-10s %s" % ("metric", "value", "target", "status"))
    print("%-22s %-8s %-10s %s" % ("-" * 22, "-" * 8, "-" * 10, "-" * 6))
    recall_ok = metrics["recall"] == metrics["recall"] and metrics["recall"] > TARGET_RECALL
    fpr_ok = metrics["fpr"] == metrics["fpr"] and metrics["fpr"] < TARGET_FPR
    print("%-22s %-8s %-10s %s" % ("recall", _fmt(metrics["recall"]), "> 0.90", "PASS" if recall_ok else "FAIL"))
    print("%-22s %-8s %-10s %s" % ("false_positive_rate", _fmt(metrics["fpr"]), "< 0.05", "PASS" if fpr_ok else "FAIL"))
    print("%-22s %-8s %-10s %s" % ("accuracy", _fmt(metrics["accuracy"]), "-", "-"))
    print("")
    print("confusion: tp=%d fp=%d tn=%d fn=%d" % (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]))


if __name__ == "__main__":
    main()
