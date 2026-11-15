# Stage 3 Training Notes: First Real Training Run

Date: 2026-11-15. Environment: CPU only, PyTorch 2.15.0+cpu, Python 3.12.
Trained with the repo's own `train.py`. Nothing in this run was pushed to
GitHub; the checkpoint and all working files live in
`~/workspace/chicken-stage3/training/`.

## Data sources

All three files describe the same sample video, fCam16 (72 frames at
10 fps, 7.2 seconds), and were downloaded from the repo:

- Stage 1 tracking output: `backend/stage1_detection/results/tracks.json`
  on branch `stage1/detection-setup` (543 KB). A dict with `fps` (10.0)
  and `frames`; each track entry has `track_id` and a `bbox` in xyxy
  pixel coordinates plus a detection score. 72 frames, 91 distinct
  tracks, 2227 track observations. It carries no behavior fields.
- Stage 2 merged tracks: `merged_tracks_fCam16.json` in
  `backend/stage2_behavior/results/infer_fcam16_merged/` on branch
  `stage2/behavior-setup` (305 KB). Same layout as the Stage 1 output:
  72 frames, 81 distinct tracks, 2272 track observations.
- Stage 2 behavior predictions: `track_predictions.json` in the same
  folder (44 KB). 120 clips over 42 tracks. Each clip has `track_id`,
  `start_frame`, `end_frame`, `label` and `confidence`. Clips are
  16 frames long with a stride of 8, so labels are per clip, never per
  frame. Label distribution: RESTING 66, ACTIVE 46, INGESTIVE 8.

## Which tracks the labels belong to, and the mapping used

The Stage 2 clips were computed on the merged track set, not on the raw
Stage 1 tracks. A coverage check proves it: 120 of 120 clips match a
merged track over more than 90 percent of their frames, versus 24 of
120 for the raw Stage 1 tracks. The merged tracks are therefore the
primary tracking source for this run, and the raw Stage 1 file was used
as comparison evidence only.

Mapping applied by `training/prepare_sequences.py`:

- Boxes are converted from xyxy to x, y, w, h.
- For each (track, frame) pair, the behavior label comes from the
  covering clip with the highest confidence; ties are broken by the
  clip whose center is closest to the frame.
- Frames covered by no clip of their track get `behavior = None`. The
  feature extractor forward-fills labels inside a continuous track
  segment, so a track keeps its last clip label until a newer clip
  covers it.
- Result: 2272 records, of which 1368 carry a direct clip label.

## Frame size

The extractor default of 640x480 would be wrong here. The Stage 2
README states the boxes live in a 1400x900 coordinate space (scaled to
960x618 for the annotated preview video), and the observed box maxima
(x2 = 1401, y2 = 901 across all files) agree. This run uses
`frame_size = (1400, 900)`.

## Sequences

- `extract_sequences` with window 32, stride 16, max gap 30.
- Output: 99 sequences of shape (32, 12), produced by 44 distinct
  tracks. Source coverage: frames 0 to 71 of one video.
- Split with seed 42 and validation fraction 0.15, using the same
  procedure as the train.py defaults, so the saved split files
  (`training/train_sequences.npy`, `training/val_sequences.npy`) are
  the split train.py used internally: 84 train, 15 validation.

## Training

- Command: repo `train.py` on `training/sequences_all.npy`, 50 epochs,
  batch size 32, Adam with learning rate 1e-3, MSE loss. Features
  standardized with the mean and standard deviation of the training
  split; both statistics are stored in the checkpoint.
- Model: LSTM autoencoder, input 12, hidden 64, latent 16, 1 layer.
  56,156 trainable parameters.
- Total training time: 19 seconds on this CPU.
- Loss curve (full log in `training/training_log.txt`): epoch 1 train
  1.0054, validation 0.9647; epoch 10 train 0.8453, validation 0.8316;
  epoch 25 train 0.6655, validation 0.7533; epoch 40 train 0.5871,
  validation 0.7191; epoch 50 train 0.5544, validation 0.7061. The
  validation loss settles near 0.70 over the last epochs while the
  training loss keeps creeping down, a mild overfitting gap that more
  data, not more epochs, would address.
- Threshold: 1.113618, the 99th percentile of the 15 validation
  reconstruction errors. With only 15 validation sequences that
  percentile is effectively the maximum, so treat the threshold as
  provisional.

## Evaluation (details in `training/eval_summary.json`)

Normal validation sequences, reconstruction error: mean 0.7061, median
0.6656, p95 1.0367, p99 1.1136, max 1.1328. One of 15 normal windows
exceeds the threshold, an empirical false positive rate of 6.7 percent
on this small set (the target is below 5 percent; the sample is too
small to call that a pass or a fail).

Perturbed copies of the validation sequences (synthetic perturbations
of the feature sequences, labeled here as perturbed-sequence
detection; this is not recall on real anomalies, because the dataset
contains no labeled real anomalies):

- Frozen motion (speed forced to zero, stillness forced to 1):
  flagged 0 of 15. Error mean 0.3884, median 0.3789, max 0.6993. The
  model reconstructs frozen windows better than average real windows.
  Resting is the majority behavior in this data (66 of 120 clips), so
  a motionless window looks normal to this checkpoint. Stillness
  anomalies are invisible to it in its current form.
- Sudden teleport (a jump of 0.35 frame diagonals in one frame):
  flagged 15 of 15. Error mean 12.6469, median 12.6336, max 14.0639,
  roughly 18 times the normal mean.
- Speed spike (five timesteps of frantic, erratic motion):
  flagged 15 of 15. Error mean 169.85, median 25.04, max 1485.19.
- Pooled: flagged 30 of 45, a 66.7 percent perturbed detection rate,
  driven entirely by the motion anomalies; pooled error mean 60.96,
  median 11.87, max 1485.19, against a normal mean of 0.71. The
  separation for motion anomalies is large and clean; the separation
  for stillness is negative.

## Inference performance on this CPU

Measured through `Stage3Scorer.score_sequences` (200 repetitions after
warmup), the path the integration pipeline uses:

- Single window: mean 5.17 ms, median 1.01 ms, p95 14.89 ms per
  window, about 193 windows per second.
- One batched pass over all 99 extracted windows: 0.071 seconds,
  0.72 ms per window.
- Pipeline target is about 1 scoring pass per second, one pass being
  the newest window of each live track. Even scoring 44 tracks one
  window at a time costs roughly 0.23 seconds at the mean latency, so
  this checkpoint has large headroom against the target on CPU.

## Checkpoint

- Location: `~/workspace/chicken-stage3/training/stage3_lstm_ae.pt`
- Size: 230,789 bytes (about 225 KB).
- Contents, in the train.py format: `state_dict`, `config`
  (input 12, hidden 64, latent 16, layers 1), `window` 32,
  `threshold` 1.113618, `threshold_percentile` 99.0, `norm_mean`,
  `norm_std`, `feature_names`.

## Limitations, stated plainly

- One video. All training data comes from a single 7.2 second sample
  (fCam16) with 99 windows and only 15 validation windows. Every
  number above is a first estimate, not a validated result.
- The behavior labels are Stage 2 model predictions, not ground truth,
  and they exist for only 42 of the 81 merged tracks.
- The frozen-motion blind spot above is the most important finding:
  with resting as the dominant normal class, this feature set and
  checkpoint cannot flag a bird that stops moving, which is one of
  the illness cues Stage 3 exists to catch. Fixing that needs either
  behavior-aware scoring (stillness judged against the Stage 2 label
  and its duration) or training data where long stillness in active
  periods is represented as abnormal.
- This checkpoint is a first model for integration wiring: it proves
  the extractor, training, checkpoint format, scorer and latency all
  work end to end on real pipeline data. Production training should
  use many more videos and longer tracks, a larger held-out set for a
  stable threshold, and a review of the stillness features before the
  threshold is trusted.
- Environment note: the venv for this run lives at
  `/home/hatch/train-venv`, not `/tmp/train-venv`, because /tmp is a
  512 MB tmpfs here and the first install attempt failed with
  "No space left on device". Reproduction scripts are in
  `training/prepare_sequences.py` and `training/evaluate.py`.
