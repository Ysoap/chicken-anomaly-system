"""Train the Stage 3 LSTM autoencoder on normal behavior sequences only.

Usage example:
    python backend/stage3_anomaly/train.py \
        --data data/stage3/normal_sequences.npy \
        --epochs 50 --hidden-dim 64 --latent-dim 16

Input data
----------
Each ``--data`` file is either:

* a ``.npy`` array of shape (num_sequences, window, feature_dim), or a
  single sequence of shape (window, feature_dim), or a long stream of shape
  (timesteps, feature_dim) which is cut into non-overlapping windows of
  length ``--window``;
* a ``.csv`` file holding a pure numeric feature matrix, one timestep per
  row and one feature per column, optionally with a header row. The rows
  are cut into non-overlapping windows of length ``--window``.

Sequences of different lengths are truncated to the shortest length so
they can be batched. Features are standardised with the mean and standard
deviation of the training split; both statistics are stored in the
checkpoint and reused by inference, so training and scoring always see
the same scaling.

The anomaly threshold is the ``--threshold-percentile`` percentile
(default 99) of the reconstruction errors on the held-out validation
split. The checkpoint saved to ``--out`` contains the model weights, the
model configuration, the threshold, and the normalisation statistics.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from feature_extractor import FEATURE_DIM, FEATURE_NAMES  # noqa: E402
from model import LSTMAutoencoder, count_parameters  # noqa: E402

CHECKPOINT_NAME = "stage3_lstm_ae.pt"


def _read_numeric_csv(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for line_no, row in enumerate(reader):
            if not row:
                continue
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                if line_no == 0:
                    continue  # header row
                raise ValueError("%s: non-numeric value on line %d" % (path, line_no + 1))
    if not rows:
        raise ValueError("%s: no numeric rows found" % path)
    return np.asarray(rows, dtype=np.float32)


def _chunk_stream(stream: np.ndarray, window: int) -> np.ndarray:
    """Cut a (timesteps, features) stream into non-overlapping windows."""
    n_windows = stream.shape[0] // window
    if n_windows == 0:
        return np.zeros((0, window, stream.shape[1]), dtype=np.float32)
    trimmed = stream[: n_windows * window]
    return trimmed.reshape(n_windows, window, stream.shape[1])


def load_sequences(paths: List[str], window: int) -> np.ndarray:
    """Load and unify feature sequences from .npy and .csv files."""
    blocks: List[np.ndarray] = []
    for raw in paths:
        path = Path(raw)
        if path.suffix == ".npy":
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 3:
                blocks.append(arr)
            elif arr.ndim == 2:
                blocks.append(_chunk_stream(arr, window))
            else:
                raise ValueError("%s: expected a 2-D or 3-D array, got %d-D" % (path, arr.ndim))
        elif path.suffix == ".csv":
            blocks.append(_chunk_stream(_read_numeric_csv(path), window))
        else:
            raise ValueError("%s: unsupported file type, use .npy or .csv" % path)

    blocks = [b for b in blocks if b.shape[0] > 0]
    if not blocks:
        raise ValueError("no usable sequences found in the given files")

    feature_dim = blocks[0].shape[2]
    for block in blocks:
        if block.shape[2] != feature_dim:
            raise ValueError("all files must share the same feature dimension")
    min_len = min(b.shape[1] for b in blocks)
    blocks = [b[:, :min_len, :] for b in blocks]
    return np.concatenate(blocks, axis=0).astype(np.float32)


def _batched_scores(model: LSTMAutoencoder, data: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    scores: List[np.ndarray] = []
    for i in range(0, data.shape[0], batch):
        chunk = torch.as_tensor(data[i : i + batch], dtype=torch.float32, device=device)
        scores.append(model.score(chunk).cpu().numpy())
    if not scores:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(scores).astype(np.float64)


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sequences = load_sequences(args.data, window=args.window)
    num_seq, seq_len, feature_dim = sequences.shape
    if feature_dim != FEATURE_DIM:
        print(
            "warning: data has %d features, the Stage 3 extractor produces %d"
            % (feature_dim, FEATURE_DIM)
        )
    print(
        "loaded %d sequences of length %d with %d features"
        % (num_seq, seq_len, feature_dim)
    )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(num_seq)
    num_val = 0
    if num_seq >= 2:
        num_val = max(1, int(round(num_seq * args.val_frac)))
        num_val = min(num_val, num_seq - 1)
    val_idx = order[:num_val]
    train_idx = order[num_val:]
    train_data = sequences[train_idx]
    val_data = sequences[val_idx] if num_val else sequences[:0]
    print("split: %d train, %d validation" % (len(train_idx), len(val_idx)))

    norm_mean = train_data.mean(axis=(0, 1))
    norm_std = train_data.std(axis=(0, 1))
    norm_std = np.where(norm_std < 1e-6, 1.0, norm_std)
    sequences_n = (sequences - norm_mean) / norm_std
    train_n = sequences_n[train_idx]
    val_n = sequences_n[val_idx] if num_val else sequences_n[:0]

    model = LSTMAutoencoder(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
    ).to(device)
    print("model parameters: %d" % count_parameters(model))

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    train_tensor = torch.as_tensor(train_n, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(args.batch_size, len(train_tensor)),
        shuffle=True,
    )
    val_tensor = torch.as_tensor(val_n, dtype=torch.float32, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item()) * batch.size(0)
            seen += batch.size(0)
        train_loss = total_loss / max(seen, 1)

        val_loss = float("nan")
        if num_val:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_tensor), val_tensor).item())
        print(
            "epoch %d/%d train_loss=%.6f val_loss=%s"
            % (epoch, args.epochs, train_loss, ("%.6f" % val_loss) if num_val else "n/a")
        )

    score_source = val_n if num_val else train_n
    val_errors = _batched_scores(model, score_source, device)
    threshold = float(np.percentile(val_errors, args.threshold_percentile))
    print(
        "threshold: %.6f (percentile %.1f of %s errors, mean %.6f)"
        % (
            threshold,
            args.threshold_percentile,
            "validation" if num_val else "training",
            float(val_errors.mean()) if val_errors.size else float("nan"),
        )
    )

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": model.config_dict(),
        "window": int(seq_len),
        "threshold": threshold,
        "threshold_percentile": float(args.threshold_percentile),
        "norm_mean": norm_mean.astype(np.float64).tolist(),
        "norm_std": norm_std.astype(np.float64).tolist(),
        "feature_names": list(FEATURE_NAMES) if feature_dim == FEATURE_DIM else [],
    }
    torch.save(checkpoint, checkpoint_path)
    print("checkpoint saved to %s" % checkpoint_path)
    return checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Stage 3 LSTM autoencoder.")
    parser.add_argument("--data", nargs="+", required=True, help=".npy or .csv files with normal sequences")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--window", type=int, default=32, help="window used to chunk stream files")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--out", default=None, help="output directory (default: backend/stage3_anomaly/models)")
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
