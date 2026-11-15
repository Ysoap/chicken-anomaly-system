"""FastAPI service for the chicken anomaly detection pipeline.

Run it with::

    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health                    liveness probe, no dependencies
GET  /api/v1/anomaly/current    latest scored window (score, threshold,
                                is_anomaly flag, timestamp)
GET  /api/v1/anomaly/history    recent scored windows, newest last,
                                bounded by an in-memory ring buffer
POST /api/v1/pipeline/score     score raw tracking records with Stage 3
GET  /api/v1/status             pipeline status (checkpoint loaded,
                                firebase enabled, windows scored)

Design notes
------------
- All mutable state lives in :class:`PipelineState`, a plain object that
  can be built and tested without starting a server.
- The Stage 3 scorer is imported lazily inside :func:`_load_scorer`, so
  this module imports cleanly and serves /health and /api/v1/status on
  machines without PyTorch. The scorer is loaded only when the
  ``STAGE3_CHECKPOINT`` environment variable points at a checkpoint.
- ``POST /api/v1/pipeline/score`` accepts tracking records in the shape
  the Stage 3 feature extractor consumes: dicts with keys ``frame``,
  ``track_id``, ``x``, ``y``, ``w``, ``h`` and an optional ``behavior``.
  The body may be a JSON list of records or an object with a ``records``
  (or ``frames``) key. When no scorer is configured it answers 503 with
  a JSON body explaining that the checkpoint is not configured.
- The integration pipeline (backend/integration/pipeline.py) can either
  run in the same process and share a PipelineState, or call
  ``POST /api/v1/pipeline/score`` over HTTP.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:  # package import (uvicorn backend.api.main:app)
    from .firebase_client import FirebaseClient
except ImportError:  # direct module import fallback
    from firebase_client import FirebaseClient  # type: ignore

logger = logging.getLogger(__name__)

ENV_CHECKPOINT = "STAGE3_CHECKPOINT"
DEFAULT_HISTORY_SIZE = 200
DEFAULT_ALERT_TOPIC = "anomalies"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineState:
    """Shared, thread-safe state between the pipeline and the API.

    The pipeline records scored windows with :meth:`record_result` and
    reads them back through the endpoint helpers. Everything is guarded
    by one lock; the history is a bounded deque so memory stays flat.
    """

    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        scorer: Any = None,
        scorer_error: Optional[str] = None,
        firebase_client: Optional[FirebaseClient] = None,
        alert_topic: str = DEFAULT_ALERT_TOPIC,
    ) -> None:
        self._lock = threading.Lock()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._windows_scored = 0
        self._anomalies_detected = 0
        self.scorer = scorer
        self.scorer_error = scorer_error
        self.firebase = firebase_client or FirebaseClient()
        self.alert_topic = alert_topic

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @property
    def scorer_configured(self) -> bool:
        return self.scorer is not None

    def score_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score tracking records with Stage 3 and record the results.

        Raises RuntimeError when no scorer is configured; callers turn
        that into a 503 response.
        """
        if self.scorer is None:
            raise RuntimeError(
                "Stage 3 scorer is not configured: set the %s environment "
                "variable to a trained checkpoint path." % ENV_CHECKPOINT
            )
        results = self.scorer.score_frames(records)
        for result in results:
            self.record_result(result)
        return results

    def record_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Store one scored window, update counters, fire side effects.

        Side effects (Firebase status write, anomaly event, push alert)
        are best effort: the Firebase client never raises, and nothing
        here is allowed to break the request or the pipeline loop.
        """
        entry = dict(result)
        entry.setdefault("threshold", self._scorer_threshold())
        entry.setdefault("timestamp", _utc_now_iso())
        with self._lock:
            self._history.append(entry)
            self._windows_scored += 1
            if entry.get("is_anomaly"):
                self._anomalies_detected += 1
            is_anomaly = bool(entry.get("is_anomaly"))
        try:
            self.firebase.update_current_status(self._status_payload(entry))
            if is_anomaly:
                self.firebase.push_anomaly_event(entry)
                self.firebase.send_alert(
                    title="Chicken anomaly detected",
                    body=(
                        "Anomaly score %.4f above threshold %.4f "
                        "(track %s, frames %s to %s)."
                        % (
                            entry.get("score", 0.0),
                            entry.get("threshold", 0.0),
                            entry.get("track_id", "?"),
                            entry.get("start_frame", "?"),
                            entry.get("end_frame", "?"),
                        )
                    ),
                    topic=self.alert_topic,
                )
        except Exception as exc:  # FirebaseClient already swallows its
            logger.warning("Firebase side effect failed: %s", exc)  # own errors
        return entry

    def _scorer_threshold(self) -> Optional[float]:
        if self.scorer is not None:
            return float(getattr(self.scorer, "threshold", 0.0))
        return None

    def _status_payload(self, current: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            windows = self._windows_scored
            anomalies = self._anomalies_detected
        return {
            "current": current,
            "windows_scored": windows,
            "anomalies_detected": anomalies,
            "updated_at": _utc_now_iso(),
        }

    # ------------------------------------------------------------------
    # Reads used by the endpoints
    # ------------------------------------------------------------------
    def current(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._history)
        if limit is not None:
            items = items[-max(0, limit):]
        return items

    def status(self) -> Dict[str, Any]:
        with self._lock:
            windows = self._windows_scored
            anomalies = self._anomalies_detected
        return {
            "stage3_checkpoint_loaded": self.scorer is not None,
            "stage3_error": self.scorer_error,
            "firebase": self.firebase.status,
            "windows_scored": windows,
            "anomalies_detected": anomalies,
            "history_size": len(self._history),
        }


def _load_scorer() -> Any:
    """Import and build the Stage 3 scorer, or return None.

    Deliberately defensive: any failure (no checkpoint configured, torch
    missing, checkpoint unreadable) leaves the API running without a
    scorer instead of crashing at import or startup time.
    """
    checkpoint = os.environ.get(ENV_CHECKPOINT)
    if not checkpoint:
        logger.info(
            "No %s set: pipeline scoring disabled, status endpoints only.",
            ENV_CHECKPOINT,
        )
        return None, None
    try:
        from backend.stage3_anomaly.inference import Stage3Scorer
    except Exception:
        try:
            from stage3_anomaly.inference import Stage3Scorer  # type: ignore
        except Exception as exc:
            message = "could not import Stage3Scorer: %s" % exc
            logger.warning("Stage 3 disabled: %s", message)
            return None, message
    try:
        scorer = Stage3Scorer(checkpoint)
        logger.info("Stage 3 scorer loaded from %s", checkpoint)
        return scorer, None
    except Exception as exc:
        message = "could not load checkpoint %s: %s" % (checkpoint, exc)
        logger.warning("Stage 3 disabled: %s", message)
        return None, message


def create_app(state: Optional[PipelineState] = None) -> FastAPI:
    """Build the FastAPI app. Pass a PipelineState to share or test."""
    if state is None:
        scorer, scorer_error = _load_scorer()
        state = PipelineState(scorer=scorer, scorer_error=scorer_error)

    app = FastAPI(title="Chicken Anomaly Detection API", version="1.0.0")
    app.state.pipeline = state

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "time": _utc_now_iso()}

    @app.get("/api/v1/anomaly/current")
    def anomaly_current() -> Any:
        current = state.current()
        if current is None:
            return JSONResponse(
                status_code=404,
                content={
                    "detail": "No window has been scored yet.",
                    "current": None,
                },
            )
        return {
            "score": current.get("score"),
            "threshold": current.get("threshold"),
            "is_anomaly": current.get("is_anomaly"),
            "track_id": current.get("track_id"),
            "start_frame": current.get("start_frame"),
            "end_frame": current.get("end_frame"),
            "timestamp": current.get("timestamp"),
        }

    @app.get("/api/v1/anomaly/history")
    def anomaly_history(limit: Optional[int] = None) -> Dict[str, Any]:
        items = state.history(limit=limit)
        return {"count": len(items), "results": items}

    @app.post("/api/v1/pipeline/score")
    async def pipeline_score(request: Request) -> Any:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"detail": "Request body must be valid JSON."},
            )
        if isinstance(payload, dict):
            records = payload.get("records", payload.get("frames"))
        else:
            records = payload
        if not isinstance(records, list):
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        "Expected a JSON list of tracking records, or an "
                        "object with a 'records' key. Each record needs "
                        "frame, track_id, x, y, w, h and may carry a "
                        "behavior label."
                    )
                },
            )
        if not state.scorer_configured:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": (
                        "Stage 3 checkpoint is not configured on this "
                        "server. Set the %s environment variable to a "
                        "trained checkpoint path and restart the API."
                        % ENV_CHECKPOINT
                    ),
                    "stage3_checkpoint_loaded": False,
                },
            )
        try:
            results = state.score_records(records)
        except Exception as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": "Scoring failed: %s" % exc},
            )
        return {
            "scored": len(results),
            "anomalies": sum(1 for r in results if r.get("is_anomaly")),
            "results": results,
        }

    @app.get("/api/v1/status")
    def pipeline_status() -> Dict[str, Any]:
        return state.status()

    return app


app = create_app()

__all__ = ["app", "create_app", "PipelineState", "ENV_CHECKPOINT"]
