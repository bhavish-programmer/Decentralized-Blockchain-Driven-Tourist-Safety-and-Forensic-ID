"""
Lightweight client for the local Fabric gateway service.
Safe-by-default: optional, short timeouts, and non-blocking helpers.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request

from loguru import logger


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def fabric_enabled() -> bool:
    return _truthy(os.getenv("ENABLE_FABRIC", "false"))


def _require_success() -> bool:
    return _truthy(os.getenv("FABRIC_REQUIRE_SUCCESS", "false"))


def _gateway_url() -> str:
    return os.getenv("FABRIC_GATEWAY_URL", "http://localhost:7059").rstrip("/")


def _timeout() -> float:
    try:
        return float(os.getenv("FABRIC_TIMEOUT_SECONDS", "3"))
    except (TypeError, ValueError):
        return 3.0


def _is_idempotent_error(msg: str) -> bool:
    lower = msg.lower()
    return "already exists" in lower or "already linked" in lower or "already recorded" in lower


def _post_json(path: str, payload: dict) -> dict:
    url = f"{_gateway_url()}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        try:
            detail = err.read().decode("utf-8")
        except Exception:
            detail = ""
        raise RuntimeError(f"Fabric gateway HTTP {err.code}: {detail or err.reason}")
    except urllib.error.URLError as err:
        raise RuntimeError(f"Fabric gateway unreachable: {getattr(err, 'reason', err)}")


def _get_json(path: str) -> dict:
    url = f"{_gateway_url()}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        try:
            detail = err.read().decode("utf-8")
        except Exception:
            detail = ""
        raise RuntimeError(f"Fabric gateway HTTP {err.code}: {detail or err.reason}")
    except urllib.error.URLError as err:
        raise RuntimeError(f"Fabric gateway unreachable: {getattr(err, 'reason', err)}")


def _call(path: str, payload: dict) -> dict | None:
    if not fabric_enabled():
        return None

    try:
        return _post_json(path, payload)
    except Exception as exc:
        msg = str(exc)
        if _is_idempotent_error(msg):
            logger.info("Fabric already recorded: {}", msg)
            return None
        if _require_success():
            raise
        logger.warning("Fabric call failed: {}", msg)
        return None


def register_did(did: str, pii_hash: str, face_hash: str | None, timestamp: str | None = None) -> dict | None:
    payload = {
        "did": did,
        "piiHash": pii_hash,
        "faceHash": face_hash or "",
        "timestamp": timestamp or "",
    }
    return _call("/fabric/register", payload)


def link_tracking_session(
    session_id: str, did: str, confidence: float, timestamp: str | None = None
) -> dict | None:
    payload = {
        "sessionId": session_id,
        "did": did,
        "confidence": float(confidence),
        "timestamp": timestamp or "",
    }
    return _call("/fabric/link", payload)


def link_tracking_session_async(
    session_id: str, did: str, confidence: float, timestamp: str | None = None
) -> None:
    if not fabric_enabled():
        return

    threading.Thread(
        target=link_tracking_session,
        args=(session_id, did, confidence, timestamp),
        daemon=True,
    ).start()


def gateway_health(force: bool = False) -> dict | None:
    if not fabric_enabled() and not force:
        return None
    try:
        return _get_json("/health")
    except Exception as exc:
        msg = str(exc)
        if _require_success() and not force:
            raise
        logger.warning("Fabric health check failed: {}", msg)
        return None


def query_did(did: str) -> dict | None:
    if not did:
        return None
    if not fabric_enabled():
        return None
    try:
        return _get_json(f"/fabric/did/{did}")
    except Exception as exc:
        msg = str(exc)
        if _require_success():
            raise
        logger.warning("Fabric DID query failed: {}", msg)
        return None


def query_link(session_id: str) -> dict | None:
    if not session_id:
        return None
    if not fabric_enabled():
        return None
    try:
        return _get_json(f"/fabric/link/{session_id}")
    except Exception as exc:
        msg = str(exc)
        if _require_success():
            raise
        logger.warning("Fabric link query failed: {}", msg)
        return None
