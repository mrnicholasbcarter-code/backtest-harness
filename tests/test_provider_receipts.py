"""Receipt contract tests mirroring verdict-core's conformance template (ADR-021)."""

from __future__ import annotations

import pytest

from backtest_harness.provider_receipts import (
    PROVIDER_VERSION,
    SCHEMA_VERSION,
    build_backtest_receipt,
    canonical_hash,
)


def test_receipt_is_deterministic_and_hashes_inputs() -> None:
    receipt = build_backtest_receipt(
        run_id="run-1",
        inputs={"value": 1, "name": "sample"},
        config={"threshold": 0.5},
        outcome="success",
        provenance={"source": "fixture", "authority": "observed"},
        evidence_refs=("evidence-1",),
        details={"approved": True},
    )
    same = build_backtest_receipt(
        run_id="run-1",
        inputs={"name": "sample", "value": 1},
        config={"threshold": 0.5},
        outcome="success",
        provenance={"source": "fixture", "authority": "observed"},
        evidence_refs=("evidence-1",),
        details={"approved": True},
    )

    assert receipt == same
    assert receipt["inputs_hash"].startswith("sha256:")
    assert receipt["config_hash"].startswith("sha256:")
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["provider"] == "verdict-backtest"
    assert receipt["provider_version"] == PROVIDER_VERSION


def test_receipt_rejects_sensitive_metadata() -> None:
    with pytest.raises(ValueError, match="sensitive"):
        build_backtest_receipt(
            run_id="run-1",
            inputs={},
            config={},
            outcome="unknown",
            provenance={"api_key": "must-not-persist"},
        )
    with pytest.raises(ValueError, match="sensitive"):
        build_backtest_receipt(
            run_id="run-1",
            inputs={},
            config={},
            outcome="unknown",
            details={"nested": [{"Token": "x"}]},
        )


def test_receipt_rejects_empty_identifiers() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_backtest_receipt(run_id="  ", inputs={}, config={}, outcome="success")
    with pytest.raises(ValueError, match="non-empty"):
        build_backtest_receipt(run_id="run-1", inputs={}, config={}, outcome="")


def test_canonical_hash_is_order_invariant_and_strict() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash([1.5, "x"]).startswith("sha256:")
    with pytest.raises(TypeError):
        canonical_hash(object())
