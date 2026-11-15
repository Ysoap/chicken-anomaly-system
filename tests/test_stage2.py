"""Smoke test Stage 2 — mapping, clip builder, data prep, plumbing VideoMAE.

Jalankan dari root repo:  python -m pytest tests/test_stage2.py -v
Test ini ringan kecuali ``test_inference_plumbing_pretrained`` yang
mengunduh checkpoint VideoMAE-Small (~90 MB, sekali saja, lalu cache)
dan menjalankannya di CPU. Tidak butuh GPU dan tidak butuh dataset asli
(ChickenAct): semua fixture dibuat sintetis di tmp_path, kecuali test
plumbing yang memakai bobot pretrained Kinetics.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE2 = os.path.join(ROOT, "backend", "stage2_behavior")
DATA_PREP = os.path.join(STAGE2, "data_prep")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- mapping
def test_mapping_15_ke_4_lengkap():
    lm = _load("s2_label_mapping",
               os.path.join(DATA_PREP, "label_mapping.py"))
    assert lm.CLASS_NAMES_4 == ["INGESTIVE", "ACTIVE", "RESTING", "AGONISTIC"]
    # 15 kelas AKTUAL di ChickenAct.zip (bukan daftar whitepaper).
    counts = {c: 0 for c in lm.CLASS_NAMES_4}
    for name15 in ("feeding", "drinking", "litter_pecking",
                   "litter_scratching", "walking", "running",
                   "wing_flapping", "resting", "standing", "preening",
                   "dust_bathing", "stretching", "head_scratching",
                   "body_shaking"):
        counts[lm.map_label15(name15)] += 1
    assert counts == {"INGESTIVE": 4, "ACTIVE": 3, "RESTING": 7,
                      "AGONISTIC": 0}, counts
    # pooping dikecualikan (bukan state perilaku).
    assert lm.map_label15("pooping") is None
    assert lm.map_label15_to_id("pooping") is None
    # Alias whitepaper tetap dikenali (meski tidak ada di zip).
    assert lm.map_label15("eating") == "INGESTIVE"
    assert lm.map_label15("sitting") == "RESTING"
    assert lm.map_label15("fighting") == "AGONISTIC"
    assert lm.ACTUAL_LABELS_15 == sorted([
        "body_shaking", "drinking", "dust_bathing", "feeding",
        "head_scratching", "litter_pecking", "litter_scratching",
        "pooping", "preening", "resting", "running", "standing",
        "stretching", "walking", "wing_flapping"])


def test_mapping_normalisasi_dan_unknown():
    lm = _load("s2_label_mapping",
               os.path.join(DATA_PREP, "label_mapping.py"))
    assert lm.map_label15("Head Shaking") == "AGONISTIC"
    assert lm.map_label15("dustbathing") == "RESTING"
    assert lm.map_label15("pecking ground") == "INGESTIVE"
    assert lm.map_label15_to_id("walking") == 1
    with pytest.raises(KeyError):
        lm.map_label15("kelas_tidak_ada")


# ------------------------------------------------------- util VideoMAE
def test_sample_frames_uniform():
    vu = _load("s2_videomae_utils", os.path.join(STAGE2, "videomae_utils.py"))
    idx = vu.sample_frames_uniform(100)
    assert len(idx) == 16 and idx[0] == 0 and idx[-1] == 99
    idx = vu.sample_frames_uniform(16)
    assert idx == list(range(16))
    idx = vu.sample_frames_uniform(5)
    assert len(idx) == 16 and idx[:5] == [0, 1, 2, 3, 4] and idx[-1] == 4
    with pytest.raises(ValueError):
        vu.sample_frames_uniform(0)


def test_preprocess_frames_bentuk():
    vu = _load("s2_videomae_utils", os.path.join(STAGE2, "videomae_utils.py"))
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
              for _ in range(16)]
    t = vu.preprocess_frames(frames)
    assert tuple(t.shape) == (16, 3, 224, 224)
    assert str(t.dtype) == "torch.float32"
    assert bool(np.isfinite(t.numpy()).all())


# ------------------------------------------------------- clip builder
def _synthetic_video_and_tracks(tmp_path, coord_scale=1.0):
    """Video 40 frame 320x240 + tracks.json 2 track yang hadir penuh."""
    import cv2

    video = str(tmp_path / "synth.mp4")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (320, 240))
    for i in range(40):
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        cv2.rectangle(frame, (50 + i, 80), (110 + i, 160), (255, 255, 255), -1)
        cv2.rectangle(frame, (180, 40 + i), (230, 100 + i), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()

    frames = []
    for i in range(40):
        frames.append({"frame": i, "name": f"frame_{i:04d}", "tracks": [
            {"track_id": 1,
             "bbox": [(50 + i) * coord_scale, 80 * coord_scale,
                      (110 + i) * coord_scale, 160 * coord_scale],
             "score": 0.95},
            {"track_id": 2,
             "bbox": [180 * coord_scale, (40 + i) * coord_scale,
                      230 * coord_scale, (100 + i) * coord_scale],
             "score": 0.90},
        ]})
    tracks_path = str(tmp_path / "tracks.json")
    with open(tracks_path, "w") as f:
        json.dump({"fps": 10.0, "score_threshold": 0.3, "frames": frames}, f)
    return video, tracks_path


def test_clip_builder_video_sintetis(tmp_path):
    cb = _load("s2_clip_builder", os.path.join(STAGE2, "clip_builder.py"))
    video, tracks_path = _synthetic_video_and_tracks(tmp_path)
    frames, fps = cb.read_video_frames(video)
    assert len(frames) == 40 and fps == 10.0
    _tfps, per_track, (mx, my) = cb.load_tracks(tracks_path)
    assert sorted(per_track) == [1, 2]
    scale = cb.detect_coord_scale(320, 240, mx, my)
    assert scale == (1.0, 1.0)
    clips = cb.build_clips(frames, per_track, clip_len=16, stride=8)
    # 2 track x jendela mulai 0, 8, 16, 24 = 8 klip.
    assert len(clips) == 8, f"klip: {len(clips)}"
    for c in clips:
        assert c["frames"].shape == (16, 224, 224, 3)
        assert c["frames"].dtype == np.uint8
        assert c["end_frame"] - c["start_frame"] == 15


def test_clip_builder_skala_koordinat(tmp_path):
    """Bbox di ruang koordinat lebih besar dari video -> skala <1 terdeteksi."""
    cb = _load("s2_clip_builder", os.path.join(STAGE2, "clip_builder.py"))
    # Kasus nyata Stage 1: track 1400x900, video re-encode 960x618.
    scale = cb.detect_coord_scale(960, 618, 1400.29, 900.46)
    assert scale[0] == scale[1]  # seragam (aspect dipertahankan re-encode)
    assert abs(scale[0] - 960 / 1400.29) < 1e-9
    # Kasus sintetis: bbox 2x ukuran video, tapi tidak mencapai ujung
    # ruang koordinat (max x2=460, max y2=320) -> skala dari batas teramati.
    video, tracks_path = _synthetic_video_and_tracks(tmp_path, coord_scale=2.0)
    frames, _fps = cb.read_video_frames(video)
    _tfps, per_track, (mx, my) = cb.load_tracks(tracks_path)
    assert mx > 320  # bbox keluar dari ukuran video
    scale = cb.detect_coord_scale(320, 240, mx, my)
    assert scale == (min(320 / mx, 240 / my),) * 2  # (s, s), seragam
    assert 0 < scale[0] < 1
    clips = cb.build_clips(frames, per_track, clip_len=16, stride=8, scale=scale)
    assert len(clips) == 8
    assert clips[0]["frames"].shape == (16, 224, 224, 3)


# ------------------------------------------------------- data prep
def _fake_chickenact_tree(tmp_path):
    """Pohon dataset palsu: 3 kelas x 4 klip mp4 kecil (20 frame)."""
    import cv2

    root = tmp_path / "ChickenAct"
    for cls in ("eating", "walking", "resting"):
        d = root / cls
        d.mkdir(parents=True)
        for k in range(4):
            p = str(d / f"clip_{k:02d}.mp4")
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"),
                                10, (64, 64))
            for _ in range(20):
                w.write(np.full((64, 64, 3), 100, dtype=np.uint8))
            w.release()
    return str(root)


def test_data_prep_manifest_fake_tree(tmp_path):
    dp = _load("s2_prepare_chickenact",
               os.path.join(DATA_PREP, "prepare_chickenact.py"))
    root = _fake_chickenact_tree(tmp_path)
    by_class = dp.find_clips(root)
    assert sorted(by_class) == ["eating", "resting", "walking"]
    assert all(len(v) == 4 for v in by_class.values())
    manifest, dist, check = dp.build_manifest(by_class, root, verify_n=1)
    assert dist["total_clips"] == 12
    assert dist["per_label4"] == {"INGESTIVE": 4, "ACTIVE": 4, "RESTING": 4}
    assert dist["excluded"] == {}
    n_train = sum(1 for r in manifest if r["split"] == "train")
    n_val = sum(1 for r in manifest if r["split"] == "val")
    n_test = sum(1 for r in manifest if r["split"] == "test")
    # 4 klip/kelas, val 12% -> 1, test 14% -> 1, train 2.
    assert (n_train, n_val, n_test) == (6, 3, 3)
    for r in manifest:
        assert set(r) >= {"path", "label15", "label4", "label4_id", "split"}
        assert r["label4_id"] == {"INGESTIVE": 0, "ACTIVE": 1,
                                  "RESTING": 2}[r["label4"]]
    assert all(c["checked"][0]["ok"] for c in check.values())
    assert check["eating"]["checked"][0]["n_frames"] == 20


def test_data_prep_pooping_dikecualikan(tmp_path):
    """Klip pooping tidak masuk manifest; AGONISTIC tetap 0."""
    import cv2

    dp = _load("s2_prepare_chickenact",
               os.path.join(DATA_PREP, "prepare_chickenact.py"))
    root = tmp_path / "ChickenAct"
    for cls in ("feeding", "pooping"):
        d = root / cls
        d.mkdir(parents=True)
        for k in range(2):
            p = str(d / f"clip_{k:02d}.mp4")
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"),
                                10, (64, 64))
            for _ in range(20):
                w.write(np.full((64, 64, 3), 100, dtype=np.uint8))
            w.release()
    by_class = dp.find_clips(str(root))
    manifest, dist, _check = dp.build_manifest(by_class, str(root),
                                               verify_n=0)
    assert dist["total_clips"] == 2  # hanya feeding
    assert dist["excluded"] == {"pooping": 2}
    assert "AGONISTIC" not in dist["per_label4"]
    assert dist["classes_4_present"] == ["INGESTIVE"]
    assert all(r["label4"] == "INGESTIVE" for r in manifest)


def test_extract_frames_fake(tmp_path):
    dp = _load("s2_prepare_chickenact",
               os.path.join(DATA_PREP, "prepare_chickenact.py"))
    ef = _load("s2_extract_frames",
               os.path.join(DATA_PREP, "extract_frames.py"))
    root = _fake_chickenact_tree(tmp_path)
    by_class = dp.find_clips(root)
    manifest, _dist, _check = dp.build_manifest(by_class, root, verify_n=0)
    out_dir = str(tmp_path / "frames")
    r = ef.extract_one(manifest[0], data_root=root, out_dir=out_dir)
    assert "error" not in r
    assert len(r["frame_files"]) == 16 and r["padded"] is False
    assert all(os.path.exists(os.path.join(out_dir, f))
               for f in r["frame_files"])


# ------------------------------------------------------- plumbing VideoMAE
def test_inference_plumbing_pretrained():
    """Load checkpoint Kinetics + klasifikasi 1 klip sintetis (CPU).

    Labelnya TIDAK ADA ARTINYA sebelum fine-tune — test ini hanya
    membuktikan plumbing: model ter-load, bentuk input/output benar.
    Skip kalau model tidak bisa diunduh (tidak ada internet).
    """
    try:
        vu = _load("s2_videomae_utils",
                   os.path.join(STAGE2, "videomae_utils.py"))
        model, _proc = vu.load_model(vu.HF_CHECKPOINT, num_labels=4,
                                     device="cpu")
    except Exception as e:  # noqa: BLE001 - jaringan/HF tidak tersedia
        pytest.skip(f"checkpoint VideoMAE tidak bisa di-load: {e}")
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
              for _ in range(16)]
    tensor = vu.preprocess_frames(frames)
    label_id, label, probs = vu.predict_clip(model, tensor, device="cpu")
    assert probs.shape == (4,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert 0 <= label_id <= 3 and label in vu.CLASS_NAMES_4
