"""Tests for the Stage 3 feature extractor and model.

Run from the repository root:
    pytest tests/test_stage3_features.py

The feature extractor tests need only NumPy. The model test is skipped
automatically when PyTorch is not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE3_DIR = REPO_ROOT / "backend" / "stage3_anomaly"
for _p in (str(REPO_ROOT), str(STAGE3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_extractor import FEATURE_DIM, FEATURE_NAMES, extract_sequences  # noqa: E402

FRAME_SIZE = (640, 480)


def synthetic_records(num_frames=40):
    """Two tracks: one walking across the frame, one sitting still."""
    records = []
    for f in range(num_frames):
        records.append(
            {
                "frame": f,
                "track_id": 1,
                "x": 50 + 3 * f,
                "y": 100 + f,
                "w": 40,
                "h": 50,
                "behavior": "walking" if (f // 10) % 2 == 0 else "pecking",
            }
        )
        records.append(
            {
                "frame": f,
                "track_id": 2,
                "x": 400,
                "y": 300,
                "w": 60,
                "h": 35,
                "behavior": "resting",
            }
        )
    return records


def test_extract_shapes_and_meta():
    sequences, meta = extract_sequences(
        synthetic_records(), window=10, stride=5, frame_size=FRAME_SIZE
    )
    # 40 frames, window 10, stride 5 gives 7 windows per track, 2 tracks.
    assert sequences.shape == (14, 10, FEATURE_DIM)
    assert sequences.dtype == np.float32
    assert np.isfinite(sequences).all()
    assert meta["num_sequences"] == 14
    assert len(meta["windows"]) == 14
    assert meta["feature_dim"] == FEATURE_DIM
    assert list(meta["feature_names"]) == list(FEATURE_NAMES)
    assert meta["behavior_available"] is True
    first = meta["windows"][0]
    assert first["track_id"] == 1
    assert first["start_frame"] == 0
    assert first["end_frame"] == 9


def test_short_track_is_skipped_unless_padded():
    records = synthetic_records()
    for f in range(5):  # a third track that only lives for 5 frames
        records.append(
            {"frame": f, "track_id": 3, "x": 10, "y": 10, "w": 30, "h": 30, "behavior": None}
        )
    sequences, _ = extract_sequences(records, window=10, stride=5, frame_size=FRAME_SIZE)
    assert sequences.shape[0] == 14

    padded, meta = extract_sequences(
        records, window=10, stride=5, frame_size=FRAME_SIZE, pad_short=True
    )
    assert padded.shape[0] == 15
    assert any(w["padded"] for w in meta["windows"])


def test_empty_input_returns_empty_array():
    sequences, meta = extract_sequences([], window=10, stride=5, frame_size=FRAME_SIZE)
    assert sequences.shape == (0, 10, FEATURE_DIM)
    assert meta["num_sequences"] == 0


def test_feature_dim_must_match():
    with pytest.raises(ValueError):
        extract_sequences(synthetic_records(), window=10, feature_dim=8)


def test_stationary_track_scores_high_stillness():
    records = [r for r in synthetic_records() if r["track_id"] == 2]
    sequences, _ = extract_sequences(records, window=10, stride=10, frame_size=FRAME_SIZE)
    stillness = sequences[:, :, FEATURE_NAMES.index("stillness")]
    speed = sequences[:, :, FEATURE_NAMES.index("centroid_speed")]
    assert stillness.mean() > 0.9
    assert speed.max() == 0.0


def test_moving_track_has_speed_and_behavior_change():
    records = [r for r in synthetic_records() if r["track_id"] == 1]
    sequences, _ = extract_sequences(records, window=30, stride=10, frame_size=FRAME_SIZE)
    speed = sequences[:, :, FEATURE_NAMES.index("centroid_speed")]
    change = sequences[:, :, FEATURE_NAMES.index("behavior_change_rate")]
    assert speed.mean() > 0.0
    assert change.max() > 0.0


def test_model_forward_and_score():
    torch = pytest.importorskip("torch")
    from model import LSTMAutoencoder

    model = LSTMAutoencoder(input_dim=FEATURE_DIM, hidden_dim=16, latent_dim=4)
    x = torch.randn(3, 10, FEATURE_DIM)
    reconstruction = model(x)
    assert reconstruction.shape == x.shape

    scores = model.score(x)
    assert scores.shape == (3,)
    assert bool((scores >= 0).all())

    numpy_scores = model.score(np.random.rand(2, 10, FEATURE_DIM).astype(np.float32))
    assert numpy_scores.shape == (2,)
