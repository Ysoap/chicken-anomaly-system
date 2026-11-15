# Stage 3: Anomaly Detection (LSTM Autoencoder)

Stage 3 is the last stage of the chicken farm monitoring pipeline. It
answers one question: **does this chicken's recent movement pattern look
unusual compared to normal flock behavior?** When the answer stays "yes"
for long enough, the integration pipeline can raise an alert so a sick or
injured bird is checked early.

The stage is unsupervised. It is trained only on sequences of normal
behavior, so it does not need labeled examples of every possible illness.

## Where Stage 3 sits in the pipeline

```
RTSP stream
  -> Stage 1: RT-DETR + ByteTrack     (bounding boxes + track IDs, per frame)
  -> Stage 2: VideoMAE                (behavior label per track per clip)
  -> Stage 3: LSTM Autoencoder        (anomaly score per track window)  <-- here
  -> Integration: FastAPI + Firebase  (alerts to the mobile app)
```

Stage 3 runs at a slower cadence than the earlier stages. Stage 1 tracks
at full frame rate and Stage 2 classifies short clips, while Stage 3
scores about one window per second per track, which is fast enough for
health monitoring and cheap enough to run on CPU.

## Inputs

The feature extractor accepts per-frame tracking records. Each record is
a dict (or a tuple in the same order) with:

| Field      | Type  | Meaning                                            |
|------------|-------|----------------------------------------------------|
| frame      | int   | Frame index in the video.                          |
| track_id   | int or str | Track identity from Stage 1 (ByteTrack).      |
| x, y       | float | Bounding box top-left corner, in pixels.           |
| w, h       | float | Bounding box width and height, in pixels.          |
| behavior   | str, optional | Behavior label from Stage 2 for this track in this frame, for example "eating", "drinking", "walking", "resting". May be None. |

Records do not need to be sorted; the extractor groups them by track and
sorts by frame itself.

### Windowing and missing data

Per track, records are resampled onto every frame from first to last
sighting. Short detection gaps are filled by linear interpolation of the
centroid and box size, and the behavior label is carried forward. If a
gap is longer than `max_gap` frames (default 30), the track is split and
no window crosses the split. Sliding windows of `window` frames (default
32, about 1 second at 30 FPS) are then cut with the given `stride`
(default 16). Tracks shorter than one window are skipped unless
`pad_short=True` is passed.

## The 12 features

Every timestep of every window is described by 12 features. Distances are
normalised by the frame diagonal and areas by the frame area, so the
features are comparable across camera setups after retraining.

| #  | Feature | Meaning |
|----|---------|---------|
| 0  | centroid_speed | Centroid movement per frame, as a fraction of the frame diagonal. |
| 1  | speed_variance_contribution | Squared deviation of the current speed from the mean speed in the window. Spikes when motion turns erratic. |
| 2  | displacement_from_window_start | Distance from the centroid at the window start, as a fraction of the frame diagonal. |
| 3  | direction_change | Angle between consecutive velocity vectors divided by pi. 0 is straight, 1 is a full reversal. |
| 4  | bbox_area_normalized | Box area divided by frame area. A proxy for camera distance and posture. |
| 5  | bbox_area_change_rate | Relative area change versus the previous frame, clipped to -5 to 5. |
| 6  | bbox_aspect_ratio | Width divided by height, clipped to 10. Changes strongly between standing and lying. |
| 7  | nearest_neighbor_distance | Distance to the closest other track in the same frame, as a fraction of the frame diagonal. 1.0 when alone. |
| 8  | local_track_density | Other tracks within 0.25 frame diagonals, divided by 10 and clipped to 1.0. High values mean crowding or huddling (berkumpul). |
| 9  | stillness | 1.0 when speed is at or below 0.002 diagonals per frame, else 0.0. Long stillness is a classic illness cue (ayam diam terlalu lama). |
| 10 | behavior_change_rate | Fraction of steps in the window so far where the Stage 2 label changed. |
| 11 | behavior_stability | Fraction of steps in the window so far that share the current label. |

Features 10 and 11 are 0.0 for tracks that carry no Stage 2 labels, so
Stage 3 still works (with less signal) when Stage 2 output is missing.

## Model architecture

`model.py` defines `LSTMAutoencoder`:

* Encoder: an LSTM reads the (window, 12) feature sequence; its final
  hidden state is projected through a linear layer to a latent vector
  (default 16 dimensions). This bottleneck is the point of the design:
  the model must compress normal behavior patterns instead of copying
  its input.
* Decoder: the latent vector is mapped back to the hidden size, repeated
  at every timestep, decoded by a second LSTM, and projected back to 12
  features per timestep.
* Anomaly score: the mean squared error between the input sequence and
  its reconstruction, computed per sequence by `model.score(...)`.

`input_dim`, `hidden_dim`, `latent_dim`, and `num_layers` are all
configurable and are stored inside every checkpoint.

## Training

Prepare a dataset of feature sequences from **normal** recordings only,
either with the extractor (`extract_sequences` from
`feature_extractor.py`, saved as `.npy`) or as a numeric `.csv` feature
stream. Then:

```bash
python backend/stage3_anomaly/train.py \
    --data data/stage3/normal_sequences.npy \
    --epochs 50 --hidden-dim 64 --latent-dim 16
```

What training does:

1. Loads `.npy` or `.csv` sequences (streams are cut into windows of
   length `--window`; mixed lengths are truncated to the shortest).
2. Splits off a validation set (`--val-frac`, default 0.15).
3. Standardises features with the training split's mean and standard
   deviation. The statistics are stored in the checkpoint and reused by
   inference, so never rescale by hand at run time.
4. Trains with Adam and MSE loss, printing one line per epoch.
5. Sets the anomaly threshold to the `--threshold-percentile` percentile
   (default 99) of validation reconstruction errors.
6. Saves `models/stage3_lstm_ae.pt` containing the weights, the model
   config, the window length, the threshold, and the normalisation
   statistics.

Training runs on CPU for small datasets and uses CUDA automatically when
available (`--device` overrides).

## Inference

### Library API (for the integration pipeline)

```python
from backend.stage3_anomaly.inference import Stage3Scorer

scorer = Stage3Scorer("backend/stage3_anomaly/models/stage3_lstm_ae.pt")

# From raw Stage 1 + Stage 2 output (list of per-frame records):
results = scorer.score_frames(tracking_records)
# -> [{"index": 0, "track_id": 7, "start_frame": 120, "end_frame": 151,
#      "score": 0.0031, "is_anomaly": False}, ...]

# From precomputed feature sequences, shape (num, window, 12):
results = scorer.score_sequences(feature_array)
# -> [{"index": 0, "score": 0.0210, "is_anomaly": True}, ...]
```

Load the scorer once at pipeline startup and reuse it. `score_frames`
uses the window stored in the checkpoint by default, with a stride of
half the window. One scoring pass per track per second is the intended
cadence: collect the newest frames, extract windows, score, and let the
integration layer decide how many consecutive anomalous windows make an
alert (a single spike is often a tracking glitch, not a sick bird).

### CLI

```bash
python backend/stage3_anomaly/inference.py \
    --checkpoint backend/stage3_anomaly/models/stage3_lstm_ae.pt \
    --input tracking.json \
    --output scores.json
```

Accepted inputs: a `.npy` file of feature sequences, a `.json` file with
a list of tracking records (or an object with a `records`, `frames`, or
`sequences` key), or a `.csv` file (a `track_id` column means tracking
records; anything else is treated as a numeric feature stream). Output
is JSON with the threshold, the number of windows and anomalies, and one
result object per window including `track_id` and frame range when the
input was tracking data.

## Evaluation

`eval_test_set.py` scores a labeled test set and reports recall, false
positive rate, and accuracy against the pipeline targets from the
project README: **recall above 90 percent** and **false positive rate
below 5 percent**.

```bash
python backend/stage3_anomaly/eval_test_set.py \
    --checkpoint backend/stage3_anomaly/models/stage3_lstm_ae.pt \
    --data data/stage3/test_set.csv
```

Accepted formats: a long-format CSV (one row per timestep with `seq_id`
and `label` columns, 0 normal and 1 anomaly, plus one column per
feature), a wide-format CSV (one row per sequence with a `label` column
and the flattened sequence), or a `.npy` array plus a `--labels` file.
The printed table shows PASS or FAIL per target together with the
confusion counts, so a failing run says exactly which side is weak:
missed anomalies (low recall) or too many false alarms (high FPR).

## File map

| File | Purpose |
|------|---------|
| feature_extractor.py | Tracking records to (num, window, 12) feature sequences. NumPy only. |
| model.py | LSTMAutoencoder definition and per-sequence scoring. |
| train.py | Training CLI. Writes checkpoints to models/. |
| inference.py | Stage3Scorer library API plus a scoring CLI. |
| eval_test_set.py | Labeled evaluation CLI against the pipeline targets. |
| models/ | Trained checkpoints (created by train.py, not committed with data). |
| ../../tests/test_stage3_features.py | Pytest coverage for the extractor and the model. |

Run the tests from the repository root:

```bash
pytest tests/test_stage3_features.py
```

## Integration notes and limitations

* The checkpoint is the contract. It carries the feature layout
  (feature_names), the normalisation statistics, and the threshold, so
  the pipeline must load the checkpoint rather than hard-coding values.
  If the extractor's feature set ever changes, retrain; old checkpoints
  will not match new features.
* Stage 3 detects *unusual*, not *sick*. A bird can score high because
  it is ill, because it was startled, or because tracking glitched.
  Treat the score as a triage signal for a human check (cek kondisi
  ayam), and consider requiring several consecutive anomalous windows
  before alerting.
* Threshold choice matters. The 99th percentile of validation errors is
  a conservative default. If evaluation shows too many false alarms on
  the real barn footage, raise the percentile; if recall is low, lower
  it or collect more varied normal training data.
* Behavior features depend on Stage 2. Without labels the model still
  works from motion and flock geometry alone, but expect weaker
  separation for behavior-driven anomalies such as not eating.
* Retraining cadence: retrain when the camera position, breed, or age
  group changes, because all of these shift what "normal" looks like.
