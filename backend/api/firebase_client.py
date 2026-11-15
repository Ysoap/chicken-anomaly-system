"""Firebase client wrapper for the chicken anomaly detection backend.

The pipeline talks to Firebase in three ways:

- Anomaly events are appended to the Realtime Database path
  ``/anomalies/events`` so the mobile app can show an alert feed.
- The latest pipeline status is written to ``/status/current`` so the
  mobile dashboard can show the current anomaly score at a glance.
- Push notifications are sent through Firebase Cloud Messaging when a
  window scores above the Stage 3 anomaly threshold.

Credentials are never hardcoded. Pass a service account JSON path to the
constructor or set the ``FIREBASE_CREDENTIALS`` environment variable.
When no credentials are available, or when the ``firebase-admin``
package is not installed, the client runs in a disabled no-op mode: it
logs what it would have sent and returns ``False`` instead of raising.
That keeps local development, CI and the pipeline demo working on
machines without Firebase access.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

STATUS_PATH = "/status/current"
EVENTS_PATH = "/anomalies/events"

ENV_CREDENTIALS = "FIREBASE_CREDENTIALS"
ENV_DATABASE_URL = "FIREBASE_DATABASE_URL"


class FirebaseClient:
    """Small defensive wrapper around firebase-admin.

    Attributes:
        enabled: True only when firebase-admin was initialised with real
            credentials. Every public method is a safe no-op returning
            False when this is False.
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        database_url: Optional[str] = None,
        app_name: str = "chicken-anomaly",
    ) -> None:
        self.credentials_path = credentials_path or os.environ.get(ENV_CREDENTIALS)
        self.database_url = database_url or os.environ.get(ENV_DATABASE_URL)
        self.app_name = app_name
        self.enabled = False
        self._app = None
        self._init_error: Optional[str] = None
        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialise(self) -> None:
        if not self.credentials_path:
            self._init_error = "no credentials path provided"
            logger.info(
                "Firebase disabled: set %s or pass credentials_path to enable it.",
                ENV_CREDENTIALS,
            )
            return
        try:
            import firebase_admin
            from firebase_admin import credentials
        except ImportError:
            self._init_error = "firebase-admin package is not installed"
            logger.warning(
                "Firebase disabled: firebase-admin is not installed, "
                "continuing in no-op mode."
            )
            return
        try:
            # Reuse an existing app with the same name when the client is
            # constructed twice in one process (tests, hot reloads).
            try:
                self._app = firebase_admin.get_app(self.app_name)
            except ValueError:
                options: Dict[str, Any] = {}
                if self.database_url:
                    options["databaseURL"] = self.database_url
                cred = credentials.Certificate(self.credentials_path)
                self._app = firebase_admin.initialize_app(
                    cred, options or None, name=self.app_name
                )
            self.enabled = True
            logger.info("Firebase enabled (app name: %s).", self.app_name)
        except Exception as exc:  # bad path, bad JSON, network, anything
            self._init_error = "%s: %s" % (type(exc).__name__, exc)
            logger.warning(
                "Firebase disabled: could not initialise (%s). "
                "Continuing in no-op mode.",
                self._init_error,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def status(self) -> Dict[str, Any]:
        """Small status dict for the /api/v1/status endpoint."""
        return {
            "enabled": self.enabled,
            "app_name": self.app_name,
            "database_url_configured": bool(self.database_url),
            "error": self._init_error,
        }

    def push_anomaly_event(self, event: Dict[str, Any]) -> bool:
        """Append one anomaly event under /anomalies/events.

        Returns True when the event was written, False in no-op mode or
        on failure. Never raises.
        """
        if not self.enabled:
            logger.info("Firebase no-op: would push anomaly event %s", event)
            return False
        try:
            from firebase_admin import db

            ref = db.reference(EVENTS_PATH, app=self._app)
            ref.push(event)
            return True
        except Exception as exc:
            logger.warning("Failed to push anomaly event: %s", exc)
            return False

    def update_current_status(self, status: Dict[str, Any]) -> bool:
        """Overwrite /status/current with the latest pipeline status."""
        if not self.enabled:
            logger.info("Firebase no-op: would update status %s", status)
            return False
        try:
            from firebase_admin import db

            ref = db.reference(STATUS_PATH, app=self._app)
            ref.set(status)
            return True
        except Exception as exc:
            logger.warning("Failed to update current status: %s", exc)
            return False

    def send_alert(
        self,
        title: str,
        body: str,
        topic: Optional[str] = None,
        token: Optional[str] = None,
    ) -> bool:
        """Send an FCM push notification to a topic or a device token.

        Exactly one of ``topic`` or ``token`` should be given. The mobile
        app subscribes to the "anomalies" topic by default. Returns True
        when the message was sent, False in no-op mode or on failure.
        """
        if not self.enabled:
            logger.info(
                "Firebase no-op: would send alert %r to topic=%s token=%s",
                title,
                topic,
                "set" if token else None,
            )
            return False
        if not topic and not token:
            logger.warning("send_alert called without a topic or token.")
            return False
        try:
            from firebase_admin import messaging

            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                topic=topic,
                token=token,
            )
            messaging.send(message, app=self._app)
            return True
        except Exception as exc:
            logger.warning("Failed to send alert: %s", exc)
            return False


__all__ = ["FirebaseClient", "STATUS_PATH", "EVENTS_PATH"]
