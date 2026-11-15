"""Tests for the API service and the integration pipeline pieces.

Covered here:
- BufferManager windowing (backend/integration/buffer_manager.py)
- FirebaseClient no-op mode (backend/api/firebase_client.py)
- FastAPI endpoints (backend/api/main.py) via TestClient

Nothing here needs PyTorch, a camera, or Firebase credentials. The
FastAPI tests are skipped automatically when fastapi or httpx is not
installed.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "backend", "integration"),
    os.path.join(_REPO_ROOT, "backend", "api"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from buffer_manager import BufferManager  # noqa: E402
from firebase_client import FirebaseClient  # noqa: E402


def make_records(track_id, frames, behavior=None):
    """Synthetic tracking records in the Stage 3 extractor shape."""
    records = []
    for frame in frames:
        record = {
            "frame": frame,
            "track_id": track_id,
            "x": 100.0 + frame,
            "y": 120.0,
            "w": 46.0,
            "h": 40.0,
        }
        if behavior is not None:
            record["behavior"] = behavior
        records.append(record)
    return records


# ----------------------------------------------------------------------
# BufferManager
# ----------------------------------------------------------------------
def test_buffer_emits_complete_windows_with_stride():
    buffer = BufferManager(window=8, stride=4)
    buffer.add_frame_records(make_records("t1", range(24)))
    windows = buffer.pop_ready_windows()
    # Window starts at 0, 4, 8, 12, 16; the start at 16 covers 16..23.
    assert len(windows) == 5
    for window_records in windows:
        assert len(window_records) == 8
        assert {r["track_id"] for r in window_records} == {"t1"}
        frames = [r["frame"] for r in window_records]
        assert frames == sorted(frames)
    assert [w[0]["frame"] for w in windows] == [0, 4, 8, 12, 16]
    # Frames 20..23 stay buffered: they are the start of the next
    # window, which still needs frames 24..27 to complete.
    counts = buffer.pending_counts()
    assert counts["records_buffered"] == 4
    assert counts["per_track"] == {"t1": 4}


def test_buffer_waits_for_a_full_window():
    buffer = BufferManager(window=8, stride=8)
    buffer.add_frame_records(make_records("t1", range(6)))
    assert buffer.pop_ready_windows() == []
    counts = buffer.pending_counts()
    assert counts["tracks"] == 1
    assert counts["records_buffered"] == 6
    assert counts["per_track"] == {"t1": 6}


def test_buffer_closes_window_across_a_small_gap():
    buffer = BufferManager(window=8, stride=8)
    # Frames 0..6 observed, frame 7 missing (detection gap), then a much
    # later frame proves the window is over. 7 of 8 slots is above the
    # default 0.8 coverage, so the window is emitted with 7 records.
    buffer.add_frame_records(make_records("t1", list(range(7)) + [20]))
    windows = buffer.pop_ready_windows()
    assert len(windows) == 1
    assert [r["frame"] for r in windows[0]] == list(range(7))


def test_buffer_groups_records_by_track():
    buffer = BufferManager(window=4, stride=4)
    buffer.add_frame_records(make_records("hen-a", range(4)))
    buffer.add_frame_records(make_records("hen-b", range(4)))
    windows = buffer.pop_ready_windows()
    assert len(windows) == 2
    by_track = {w[0]["track_id"]: w for w in windows}
    assert set(by_track) == {"hen-a", "hen-b"}
    assert all(len(w) == 4 for w in windows)


def test_buffer_cap_drops_oldest_records():
    buffer = BufferManager(window=32, stride=16, max_records=10)
    buffer.add_frame_records(make_records("t1", range(25)))
    counts = buffer.pending_counts()
    assert counts["records_buffered"] <= 10
    assert counts["dropped_records"] >= 15


# ----------------------------------------------------------------------
# FirebaseClient no-op mode
# ----------------------------------------------------------------------
def test_firebase_client_noop_without_credentials(monkeypatch):
    monkeypatch.delenv("FIREBASE_CREDENTIALS", raising=False)
    monkeypatch.delenv("FIREBASE_DATABASE_URL", raising=False)
    client = FirebaseClient()
    assert client.enabled is False
    assert client.push_anomaly_event({"score": 1.5}) is False
    assert client.update_current_status({"current": None}) is False
    assert client.send_alert("title", "body", topic="anomalies") is False
    status = client.status
    assert status["enabled"] is False


# ----------------------------------------------------------------------
# FastAPI endpoints
# ----------------------------------------------------------------------
@pytest.fixture()
def api_client(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.delenv("STAGE3_CHECKPOINT", raising=False)
    monkeypatch.delenv("FIREBASE_CREDENTIALS", raising=False)
    from fastapi.testclient import TestClient

    from backend.api.main import PipelineState, create_app

    state = PipelineState(firebase_client=FirebaseClient())
    return TestClient(create_app(state))


def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_status_before_any_scoring(api_client):
    response = api_client.get("/api/v1/status")
    assert response.status_code == 200
    body = response.json()
    assert body["stage3_checkpoint_loaded"] is False
    assert body["firebase"]["enabled"] is False
    assert body["windows_scored"] == 0
    assert body["anomalies_detected"] == 0


def test_current_before_any_scoring(api_client):
    response = api_client.get("/api/v1/anomaly/current")
    assert response.status_code == 404
    assert response.json()["current"] is None


def test_history_before_any_scoring(api_client):
    response = api_client.get("/api/v1/anomaly/history")
    assert response.status_code == 200
    assert response.json() == {"count": 0, "results": []}


def test_score_without_checkpoint_is_a_clear_503(api_client):
    response = api_client.post(
        "/api/v1/pipeline/score",
        json={"records": make_records("t1", range(40))},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["stage3_checkpoint_loaded"] is False
    assert "STAGE3_CHECKPOINT" in body["detail"]
