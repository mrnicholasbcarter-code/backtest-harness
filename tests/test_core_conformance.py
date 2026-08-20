"""Boundary conformance against real verdict-core contracts (ADR-021, BACK-001).

Runs only when ``verdict`` (verdict-core) is installed — CI executes it in the
``compat-gate`` job.  Exercises the exported payloads through Core's own
``ProviderReceipt``, ``VerificationResult``, and ``EvidenceChainLink``
validators so drift fails here rather than at replay time.
"""

from __future__ import annotations

import pytest

verdict_contracts = pytest.importorskip("verdict.contracts")
verdict_receipts = pytest.importorskip("verdict.provider_receipts")

from backtest_harness.evidence import (  # noqa: E402
    build_failure_evidence,
    run_counterfactual,
    to_evidence_chain_link,
    to_verification_result,
)
from backtest_harness.provider_receipts import canonical_hash  # noqa: E402


def _evidence():
    return run_counterfactual(
        run_id="conformance-1",
        trade_returns=[0.02, -0.01, 0.03, 0.005, -0.02, 0.015, 0.01, -0.005],
        starting_equity=1000.0,
        seed=7,
        dataset_ref="dataset://fixtures/returns-v1",
        num_simulations=100,
        trades_per_sim=25,
        walk_forward_splits=4,
        fee_config={"model": "bounded_profit"},
        fee_trades=[(50.0, 100.0)],
    )


def test_canonical_hash_matches_core() -> None:
    payload = {"b": [1, 2.5, "x"], "a": {"nested": True, "n": None}}
    assert canonical_hash(payload) == verdict_receipts.canonical_hash(payload)


def test_receipt_round_trips_through_core_provider_receipt() -> None:
    receipt = _evidence()["receipt"]
    core_receipt = verdict_receipts.ProviderReceipt.from_dict(receipt)
    assert core_receipt.to_dict() == receipt


def test_verification_result_export_passes_core_contract() -> None:
    evidence = _evidence()
    payload = to_verification_result(evidence, policy_requirement="BACK-001")
    result = verdict_contracts.VerificationResult.from_dict(payload)
    assert result.status == "passed"
    assert result.check_type == "custom"

    failed = build_failure_evidence(run_id="conformance-2", outcome="failure", reason="x")
    failed_result = verdict_contracts.VerificationResult.from_dict(to_verification_result(failed))
    assert failed_result.status == "failed"


def test_evidence_chain_link_export_passes_core_contract() -> None:
    evidence = _evidence()
    link_payload = to_evidence_chain_link(
        evidence,
        decision="allow",
        policy="policy://backtest/counterfactual",
        envelope_hash=canonical_hash({"envelope": "conformance"}),
        runtime="verdict-node",
        model="none",
        timestamp="2026-08-20T00:00:00Z",
    )
    link = verdict_contracts.EvidenceChainLink.from_dict(link_payload)
    assert link.provider == "verdict-backtest"
    assert link.outcome == "success"
