"""Stage 3 feature extraction: tracking data to fixed-size feature sequences.

This module converts the outputs of Stage 1 (per-frame bounding boxes with
track IDs from RT-DETR and ByteTrack) and Stage 2 (an optional behavior label
per track per frame from VideoMAE) into the fixed-length feature sequences
that the Stage 3 LSTM autoencoder consumes.

Input format
------------
``frames`` is an iterable of per-frame records. Each record is either a dict
with keys ``frame``, ``track_id``, ``x``, ``y``, ``w``, ``h`` and an optional
``behavior`` key, or a tuple/list in the positional order
``(frame, track_id, x, y, w, h[, behavior])``. Bounding boxes use pixel
coordinates with ``x, y`` at the top-left corner. ``behavior`` may be a
string label such as ``"eating"`` or ``None`` when Stage 2 has no label.

Output
------
``extract_sequences`` returns ``(sequences, meta)`` where ``sequences`` is a
float32 NumPy array of shape ``(num_windows, window, FEATURE_DIM)`` and
``meta`` describes every window (track id, start and end frame) plus the
feature names and the parameters that were used.

The 12 features (see ``FEATURE_NAMES``)
---------------------------------------
0.  centroid_speed: centroid movement per frame, as a fraction of the frame
    diagonal.
1.  speed_variance_contribution: squared deviation of the current speed from
    the mean speed inside the window. Spikes when motion is erratic.
2.  displacement_from_window_start: distance between the current centroid
    and the centroid at the start of the window, as a fraction of the frame
    diagonal.
3.  direction_change: angle between consecutive velocity vectors, divided
    by pi, so 0 means straight motion and 1 means a full reversal.
4.  bbox_area_normalized: bounding box area divided by frame area. A proxy
    for distance to the camera and for posture changes.
5.  bbox_area_change_rate: relative area change versus the previous frame,
    clipped to the range -5 to 5.
6.  bbox_aspect_ratio: width divided by height, clipped to 10. Lying down
    versus standing changes this value strongly.
7.  nearest_neighbor_distance: distance to the closest other track in the
    same frame, as a fraction of the frame diagonal. 1.0 when the track is
    alone in the frame.
8.  local_track_density: number of other tracks within 0.25 frame diagonals,
    divided by 10 and clipped to 1.0. High values mean crowding or huddling.
9.  stillness: 1.0 when the centroid speed is at or below 0.002 frame
    diagonals per frame, else 0.0. Long stillness in a normally active bird
    is a classic illness cue.
10. behavior_change_rate: fraction of steps inside the window, up to the
    current step, where the Stage 2 behavior label changed.
11. behavior_stability: fraction of steps inside the window, up to the
    current step, whose behavior label equals the current label.

Features 10 and 11 are 0.0 for tracks that carry no behavior labels at all.

Missing data policy
-------------------
Detections can drop for a few frames. Inside one track, gaps are filled by
linear interpolation of the centroid and box size, and the behavior label is
carried forward. If a gap is longer than ``max_gap`` frames, the track is
split into separate segments and no window crosses the split. Tracks shorter
than ``window`` are skipped unless ``pad_short`` is true, in which case the
last observation is repeated (which correctly yields zero speed).

Only NumPy is required.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_DIM = 12

FEATURE_NAMES: Tuple[str, ...] = (
    "centroid_speed",
    "speed_variance_contribution",
    "displacement_from_window_start",
    "direction_change",
    "bbox_area_normalized",
    "bbox_area_change_rate",
    "bbox_aspect_ratio",
    "nearest_neighbor_distance",
    "local_track_density",
    "stillness",
    "behavior_change_rate",
    "behavior_stability",
)

_STILLNESS_SPEED_THRESHOLD = 0.002  # fraction of frame diagonal per frame
_DENSITY_RADIUS_FRACTION = 0.25     # fraction of frame diagonal
_DENSITY_SCALE = 10.0               # neighbour count that maps to 1.0
_ASPECT_RATIO_CLIP = 10.0
_AREA_CHANGE_CLIP = 5.0

__all__ = ["FEATURE_DIM", "FEATURE_NAMES", "extract_sequences"]


def _parse_record(record: Any) -> Tuple[int, Any, float, float, float, float, Any]:
    """Normalise one input record to (frame, track_id, x, y, w, h, behavior)."""
    if isinstance(record, dict):
        frame = record.get("frame", record.get("frame_idx", record.get("frame_index")))
        track_id = record.get("track_id", record.get("id"))
        if record.get("bbox") is not None:
            x, y, w, h = record["bbox"]
        else:
            x, y, w, h = record["x"], record["y"], record["w"], record["h"]
        behavior = record.get("behavior", record.get("behavior_label"))
    else:
        values = list(record)
        if len(values) == 6:
            frame, track_id, x, y, w, h = values
            behavior = None
        elif len(values) >= 7:
            frame, track_id, x, y, w, h, behavior = values[:7]
        else:
            raise ValueError(
                "tuple records need 6 or 7 fields: "
                "(frame, track_id, x, y, w, h[, behavior])"
            )
    if frame is None or track_id is None:
        raise ValueError("record is missing a frame index or a track_id: %r" % (record,))
    return int(frame), track_id, float(x), float(y), float(w), float(h), behavior


class _Segment:
    """One continuous run of a single track, resampled onto every frame."""

    def __init__(self, track_id: Any) -> None:
        self.track_id = track_id
        self.frames: List[int] = []
        self.cx: List[float] = []
        self.cy: List[float] = []
        self.w: List[float] = []
        self.h: List[float] = []
        self.behavior: List[Any] = []
        self.has_behavior = False


def _build_segments(
    parsed: List[Tuple[int, Any, float, float, float, float, Any]],
    max_gap: Optional[int],
) -> List[_Segment]:
    """Group records by track and resample each track onto its frame range."""
    by_track: Dict[Any, Dict[int, Tuple[float, float, float, float, Any]]] = {}
    for frame, track_id, x, y, w, h, behavior in parsed:
        by_track.setdefault(track_id, {})[frame] = (
            x + w / 2.0,
            y + h / 2.0,
            w,
            h,
            behavior,
        )

    segments: List[_Segment] = []
    for track_id in sorted(by_track, key=lambda t: str(t)):
        observed = by_track[track_id]
        obs_frames = sorted(observed)

        # Split the track where the detection gap is too large.
        runs: List[List[int]] = [[obs_frames[0]]]
        for prev, cur in zip(obs_frames, obs_frames[1:]):
            if max_gap is not None and cur - prev > max_gap:
                runs.append([cur])
            else:
                runs[-1].append(cur)

        for run in runs:
            seg = _Segment(track_id)
            run_idx = np.array(run, dtype=np.int64)
            full_frames = np.arange(run[0], run[-1] + 1, dtype=np.int64)
            cx_obs = np.array([observed[f][0] for f in run], dtype=np.float64)
            cy_obs = np.array([observed[f][1] for f in run], dtype=np.float64)
            w_obs = np.array([observed[f][2] for f in run], dtype=np.float64)
            h_obs = np.array([observed[f][3] for f in run], dtype=np.float64)

            if len(run) == 1:
                seg.frames = run[:]
                seg.cx = [float(cx_obs[0])]
                seg.cy = [float(cy_obs[0])]
                seg.w = [float(w_obs[0])]
                seg.h = [float(h_obs[0])]
                seg.behavior = [observed[run[0]][4]]
            else:
                seg.frames = [int(f) for f in full_frames]
                seg.cx = [float(v) for v in np.interp(full_frames, run_idx, cx_obs)]
                seg.cy = [float(v) for v in np.interp(full_frames, run_idx, cy_obs)]
                seg.w = [float(v) for v in np.interp(full_frames, run_idx, w_obs)]
                seg.h = [float(v) for v in np.interp(full_frames, run_idx, h_obs)]
                last_behavior: Any = None
                behavior_by_frame: List[Any] = []
                obs_set = set(run)
                for f in seg.frames:
                    if f in obs_set and observed[f][4] is not None:
                        last_behavior = observed[f][4]
                    behavior_by_frame.append(last_behavior)
                seg.behavior = behavior_by_frame

            seg.has_behavior = any(b is not None for b in seg.behavior)
            segments.append(seg)
    return segments


def _frame_centroids(
    parsed: List[Tuple[int, Any, float, float, float, float, Any]],
) -> Dict[int, List[Tuple[Any, float, float]]]:
    """Map each frame index to the observed (track_id, cx, cy) points."""
    per_frame: Dict[int, List[Tuple[Any, float, float]]] = {}
    for frame, track_id, x, y, w, h, _behavior in parsed:
        per_frame.setdefault(frame, []).append((track_id, x + w / 2.0, y + h / 2.0))
    return per_frame


def _base_features(
    seg: _Segment,
    per_frame: Dict[int, List[Tuple[Any, float, float]]],
    frame_w: float,
    frame_h: float,
    diag: float,
) -> Dict[str, np.ndarray]:
    """Per-timestep features that do not depend on the window position."""
    n = len(seg.frames)
    cx = np.asarray(seg.cx, dtype=np.float64)
    cy = np.asarray(seg.cy, dtype=np.float64)
    w = np.asarray(seg.w, dtype=np.float64)
    h = np.asarray(seg.h, dtype=np.float64)

    step_x = np.zeros(n)
    step_y = np.zeros(n)
    if n > 1:
        step_x[1:] = np.diff(cx)
        step_y[1:] = np.diff(cy)
    speed = np.hypot(step_x, step_y) / diag

    direction_change = np.zeros(n)
    if n > 2:
        dot = step_x[2:] * step_x[1:-1] + step_y[2:] * step_y[1:-1]
        norms = np.hypot(step_x[2:], step_y[2:]) * np.hypot(step_x[1:-1], step_y[1:-1])
        cos_angle = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 1e-12)
        direction_change[2:] = np.arccos(np.clip(cos_angle, -1.0, 1.0)) / math.pi

    area = np.maximum(w * h, 1.0)
    area_norm = area / float(frame_w * frame_h)
    area_change = np.zeros(n)
    if n > 1:
        area_change[1:] = (area[1:] - area[:-1]) / area[:-1]
        area_change = np.clip(area_change, -_AREA_CHANGE_CLIP, _AREA_CHANGE_CLIP)
    aspect = np.clip(w / np.maximum(h, 1e-6), 0.0, _ASPECT_RATIO_CLIP)

    nn_distance = np.ones(n)
    density = np.zeros(n)
    radius_px = _DENSITY_RADIUS_FRACTION * diag
    for i, frame in enumerate(seg.frames):
        others = [
            (ox, oy) for (tid, ox, oy) in per_frame.get(frame, []) if tid != seg.track_id
        ]
        if not others:
            continue
        dists = [math.hypot(ox - cx[i], oy - cy[i]) for (ox, oy) in others]
        nn_distance[i] = min(dists) / diag
        near = sum(1 for d in dists if d <= radius_px)
        density[i] = min(near / _DENSITY_SCALE, 1.0)

    stillness = (speed <= _STILLNESS_SPEED_THRESHOLD).astype(np.float64)

    return {
        "cx": cx,
        "cy": cy,
        "speed": speed,
        "direction_change": direction_change,
        "area_norm": area_norm,
        "area_change": area_change,
        "aspect": aspect,
        "nn_distance": nn_distance,
        "density": density,
        "stillness": stillness,
    }


def _window_features(
    base: Dict[str, np.ndarray],
    behavior: Sequence[Any],
    has_behavior: bool,
    start: int,
    window: int,
    diag: float,
) -> np.ndarray:
    """Assemble the (window, FEATURE_DIM) matrix for one window."""
    sl = slice(start, start + window)
    speed = base["speed"][sl]

    mean_speed = float(speed.mean()) if window else 0.0
    variance_contribution = (speed - mean_speed) ** 2

    displacement = (
        np.hypot(base["cx"][sl] - base["cx"][start], base["cy"][sl] - base["cy"][start])
        / diag
    )

    if has_behavior:
        labels = list(behavior[sl])
        change_rate = np.zeros(window)
        stability = np.zeros(window)
        changes = 0
        for t in range(window):
            if t > 0 and labels[t] != labels[t - 1]:
                changes += 1
            change_rate[t] = changes / float(t + 1)
            same = sum(1 for k in range(t + 1) if labels[k] == labels[t])
            stability[t] = same / float(t + 1)
    else:
        change_rate = np.zeros(window)
        stability = np.zeros(window)

    columns = [
        speed,
        variance_contribution,
        displacement,
        base["direction_change"][sl],
        base["area_norm"][sl],
        base["area_change"][sl],
        base["aspect"][sl],
        base["nn_distance"][sl],
        base["density"][sl],
        base["stillness"][sl],
        change_rate,
        stability,
    ]
    return np.stack(columns, axis=1)


def extract_sequences(
    frames: Iterable[Any],
    window: int = 32,
    stride: int = 16,
    feature_dim: int = FEATURE_DIM,
    frame_size: Tuple[float, float] = (640, 480),
    max_gap: Optional[int] = 30,
    pad_short: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract fixed-length feature sequences from tracking records.

    Args:
        frames: Iterable of per-frame records (see module docstring).
        window: Timesteps per sequence. At 30 FPS, 32 steps is about 1 second.
        stride: Step between consecutive window starts, in frames.
        feature_dim: Must equal FEATURE_DIM (12). Kept as a parameter so
            callers and checkpoints can validate the layout explicitly.
        frame_size: (width, height) of the video frame in pixels, used to
            normalise distances and areas.
        max_gap: Split a track when detections are missing for more than
            this many frames. Use None to interpolate across any gap.
        pad_short: When true, tracks shorter than ``window`` still produce
            one window by repeating their last observation.

    Returns:
        (sequences, meta). ``sequences`` is a float32 array with shape
        (num_windows, window, FEATURE_DIM). ``meta`` holds the feature
        names, the parameters used, whether any behavior labels were seen,
        and one entry per window with track id and frame range.
    """
    if feature_dim != FEATURE_DIM:
        raise ValueError(
            "feature_dim must be %d for this extractor, got %r" % (FEATURE_DIM, feature_dim)
        )
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")

    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    diag = math.hypot(frame_w, frame_h)

    parsed = [_parse_record(r) for r in frames]
    meta: Dict[str, Any] = {
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": FEATURE_DIM,
        "window": window,
        "stride": stride,
        "frame_size": (frame_w, frame_h),
        "max_gap": max_gap,
        "behavior_available": any(r[6] is not None for r in parsed),
        "windows": [],
    }
    if not parsed:
        meta["num_sequences"] = 0
        return np.zeros((0, window, FEATURE_DIM), dtype=np.float32), meta

    per_frame = _frame_centroids(parsed)
    segments = _build_segments(parsed, max_gap=max_gap)

    all_windows: List[np.ndarray] = []
    for seg in segments:
        n = len(seg.frames)
        padded = False
        if n < window:
            if not pad_short:
                continue
            pad = window - n
            seg.frames = seg.frames + [seg.frames[-1]] * pad
            seg.cx = seg.cx + [seg.cx[-1]] * pad
            seg.cy = seg.cy + [seg.cy[-1]] * pad
            seg.w = seg.w + [seg.w[-1]] * pad
            seg.h = seg.h + [seg.h[-1]] * pad
            seg.behavior = seg.behavior + [seg.behavior[-1]] * pad
            n = window
            padded = True

        base = _base_features(seg, per_frame, frame_w, frame_h, diag)
        for start in range(0, n - window + 1, stride):
            feats = _window_features(base, seg.behavior, seg.has_behavior, start, window, diag)
            all_windows.append(feats.astype(np.float32))
            end_frame = seg.frames[start + window - 1]
            meta["windows"].append(
                {
                    "track_id": seg.track_id,
                    "start_frame": int(seg.frames[start]),
                    "end_frame": int(end_frame if not padded else seg.frames[window - 1]),
                    "padded": bool(padded),
                }
            )

    if all_windows:
        sequences = np.stack(all_windows, axis=0)
    else:
        sequences = np.zeros((0, window, FEATURE_DIM), dtype=np.float32)
    meta["num_sequences"] = int(sequences.shape[0])
    return sequences, meta
