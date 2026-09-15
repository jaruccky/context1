from __future__ import annotations

import hashlib
import uuid


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def hash_call(tool: str, arguments: dict) -> str:
    """Stable hash of a tool call, used for duplicate-call detection."""
    payload = f"{tool}:{sorted(arguments.items())}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
