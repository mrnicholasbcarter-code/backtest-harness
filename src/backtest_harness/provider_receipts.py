"""Provider-neutral receipts for deterministic backtest evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "1"


def build_backtest_receipt(
    *,
    run_id: str,
    inputs: Any,
    config: Any,
    outcome: str,
    evidence_refs: tuple[str, ...] = (),
    provenance: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, informational backtest-provider receipt."""
    if not run_id.strip() or not outcome.strip():
        raise ValueError("run_id and outcome must be non-empty")
    _reject_sensitive(provenance or {})
    _reject_sensitive(details or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "provider": "verdict-backtest",
        "provider_version": "0.2.0",
        "inputs_hash": _hash(inputs),
        "config_hash": _hash(config),
        "outcome": outcome,
        "provenance": dict(provenance or {}),
        "evidence_refs": list(evidence_refs),
        "details": dict(details or {}),
    }


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _reject_sensitive(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower().replace("-", "_") in {
                "api_key",
                "authorization",
                "password",
                "secret",
                "token",
            }:
                raise ValueError(f"sensitive receipt field rejected: {key}")
            _reject_sensitive(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_sensitive(child)


__all__ = ["SCHEMA_VERSION", "build_backtest_receipt"]
