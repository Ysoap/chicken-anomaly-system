"""Track window buffer for the integration pipeline.

Stage 1 produces per-frame tracking records (frame, track_id, x, y, w, h,
plus an optional Stage 2 behavior label). Stage 3 consumes fixed-length
windows of consecutive records per track. This module sits between them:
it collects records as they stream in, keeps them grouped by track, and
emits completed windows in exactly the record format the Stage 3 feature
extractor accepts.

Windowing model
---------------
Per track, records are kept in frame order. A window covers ``window``
consecutive frame slots (default 32). Windows start at the track's first
buffered frame and then every ``stride`` frames (default 16). A window is
ready once every frame slot in it has been observed, or once a record
arrives beyond the window end and the observed coverage reaches
``min_coverage`` (default 0.8), which tolerates short detection gaps:
the Stage 3 extractor interpolates small gaps itself. After a window is
emitted, records older than the next window start are dropped, so a
long-running track uses bounded memory.

A second, global bound caps the total buffered records across all tracks
(``max_records``, default 20000). When the cap is exceeded the oldest
records are dropped first and the drop is counted in
:attr:`BufferManager.dropped_records`.

The buffer is thread-safe: one thread can feed frames while another pops
ready windows (the pipeline scores in its own rhythm, about once per
second, while frames arrive at camera rate).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, Hashable, List, Tuple

DEFAULT_WINDOW = 32
DEFAULT_STRIDE = 16
DEFAULT_MAX_RECORDS = 20000
DEFAULT_MIN_COVERAGE = 0.8


class _TrackBuffer:
    """Buffered records for one track, keyed by frame index."""

    def __init__(self) -> None:
        self.records: Dict[int, Dict[str, Any]] = {}
        self.next_start: int | None = None  # frame index of next window start

    def add(self, record: Dict[str, Any]) -> None:
        frame = int(record["frame"])
        self.records[frame] = record
        if self.next_start is None or frame < self.next_start:
            if self.next_start is None:
                self.next_start = frame

    def frames(self) -> List[int]:
        return sorted(self.records)


class BufferManager:
    """Collects tracking records and emits Stage 3 ready windows.

    Args:
        window: Number of consecutive frame slots per window.
        stride: Frame step between consecutive window starts.
        max_records: Global cap on buffered records across all tracks.
        min_coverage: Fraction of a window's frame slots that must be
            observed before the window may be emitted when later frames
            prove the missing ones are detection gaps.
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
        max_records: int = DEFAULT_MAX_RECORDS,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> None:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        self.window = window
        self.stride = stride
        self.max_records = max_records
        self.min_coverage = min_coverage
        self.dropped_records = 0
        self._tracks: Dict[Hashable, _TrackBuffer] = {}
        self._total = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def add_frame_records(self, records: List[Dict[str, Any]]) -> int:
        """Add all tracking records observed in one or more frames.

        Each record must be a dict with at least ``frame`` and
        ``track_id`` keys, in the Stage 3 extractor record shape. Returns
        the number of records accepted.
        """
        accepted = 0
        with self._lock:
            for record in records:
                if not isinstance(record, dict):
                    continue
                if "frame" not in record or "track_id" not in record:
                    continue
                track = self._tracks.setdefault(record["track_id"], _TrackBuffer())
                is_new = int(record["frame"]) not in track.records
                track.add(record)
                if is_new:
                    self._total += 1
                accepted += 1
            self._enforce_cap()
        return accepted

    def _enforce_cap(self) -> None:
        """Drop the globally oldest records until under the cap."""
        while self._total > self.max_records:
            oldest: Tuple[int, Hashable] | None = None
            for track_id, track in self._tracks.items():
                if not track.records:
                    continue
                first = min(track.records)
                if oldest is None or first < oldest[0]:
                    oldest = (first, track_id)
            if oldest is None:
                break
            frame, track_id = oldest
            track = self._tracks[track_id]
            del track.records[frame]
            self._total -= 1
            self.dropped_records += 1
            if track.next_start is not None and track.next_start <= frame:
                track.next_start = frame + 1
            if not track.records:
                del self._tracks[track_id]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def pop_ready_windows(self) -> List[List[Dict[str, Any]]]:
        """Remove and return every completed window.

        Each window is a list of tracking records (frame order) for one
        track, ready to pass to ``Stage3Scorer.score_frames`` or to the
        Stage 3 feature extractor.
        """
        ready: List[List[Dict[str, Any]]] = []
        with self._lock:
            for track_id in list(self._tracks):
                track = self._tracks[track_id]
                while True:
                    window_records = self._take_window(track)
                    if window_records is None:
                        break
                    ready.append(window_records)
                if not track.records:
                    del self._tracks[track_id]
        return ready

    def _take_window(self, track: _TrackBuffer) -> List[Dict[str, Any]] | None:
        if track.next_start is None or not track.records:
            return None
        start = track.next_start
        end = start + self.window - 1
        observed = [f for f in track.records if start <= f <= end]
        max_frame = max(track.records)
        complete = len(observed) == self.window
        closable = (
            max_frame > end
            and len(observed) >= self.min_coverage * self.window
        )
        if not (complete or closable):
            return None
        window_records = [track.records[f] for f in sorted(observed)]
        # Advance the start and drop records the next window cannot use.
        track.next_start = start + self.stride
        for frame in [f for f in track.records if f < track.next_start]:
            del track.records[frame]
            self._total -= 1
        return window_records

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def pending_counts(self) -> Dict[str, Any]:
        """Snapshot of what is buffered, for status endpoints and logs."""
        with self._lock:
            per_track = {
                track_id: len(track.records)
                for track_id, track in self._tracks.items()
            }
            return {
                "tracks": len(self._tracks),
                "records_buffered": self._total,
                "per_track": per_track,
                "dropped_records": self.dropped_records,
                "window": self.window,
                "stride": self.stride,
            }


__all__ = ["BufferManager", "DEFAULT_WINDOW", "DEFAULT_STRIDE"]
