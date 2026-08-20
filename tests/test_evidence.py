"""Counterfactual provider and evidence-export tests (BACK-001)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from backtest_harness.evidence import (
    COUNTERFACTUAL_OUTCOMES,
    build_failure_evidence,
    run_counterfactual,
    to_evidence_chain_link,
    to_verification_result,
)
from backtest_harness.provider_receipts import canonical_hash

RETURNS = [0.02, -0.01, 0.03, 0.005, -0.02, 0.015, 0.01, -0.005, 0.02, -0.01]
ENVELOPE_HASH = canonical_hash({"envelope": "test"})
TIMESTAMP = "2026-08-20T00:00:00Z"


def _run(**overrides):
    kwargs = {
        "run_id": "run-1",
        "trade_returns": RETURNS,
        "starting_equity": 1000.0,
        "seed": 42,
        "dataset_ref": "dataset://fixtures/returns-v1",
        "num_simulations": 200,
        "trades_per_sim": 50,
        "walk_forward_splits": 5,
    }
    kwargs.update(overrides)
    return run_counterfactual(**kwargs)


class TestRunCounterfactual:
    def test_same_seed_reproduces_identical_evidence(self):
        first = _run()
        second = _run()
        assert first == second
        assert first["results_hash"] == second["results_hash"]
        assert first["receipt"]["inputs_hash"] == second["receipt"]["inputs_hash"]
        assert first["receipt"]["config_hash"] == second["receipt"]["config_hash"]

    def test_different_seed_changes_results(self):
        assert _run(seed=1)["results_hash"] != _run(seed=2)["results_hash"]

    def test_records_seed_fee_model_code_version_and_dataset(self):
        evidence = _run(
            fee_config={
                "model": "bounded_profit",
                "percent_of_profit": 0.07,
                "maximum_fee_cents": 0.05,
            },
            fee_trades=[(50.0, 100.0), (60.0, 0.0)],
        )
        receipt = evidence["receipt"]
        # The receipt binds the results by canonical hash.
        assert evidence["results_hash"] in receipt["evidence_refs"]
        assert evidence["results_hash"] == canonical_hash(evidence["results"])
        # Bundle is JSON-compatible (portable evidence).
        json.dumps(evidence)
        results = evidence["results"]
        assert set(results) == {"monte_carlo", "tearsheet", "walk_forward", "fees"}
        assert results["fees"] == {
            "model": "bounded_profit",
            "params": {"percent_of_profit": 0.07, "maximum_fee_cents": 0.05},
            "n_trades": 2,
            "total_fee": pytest.approx(0.05),  # capped fee + zero on a loss
            "mean_fee": pytest.approx(0.025),
        }
        assert len(results["walk_forward"]) == 4  # n_splits - 1 expanding folds

    def test_flat_maker_taker_fee_summary(self):
        evidence = _run(
            fee_config={"model": "flat_maker_taker", "maker_bps": 0.0, "taker_bps": 0.001},
            fee_trades=[(1000.0, False), (1000.0, True)],
        )
        fees = evidence["results"]["fees"]
        assert fees["total_fee"] == pytest.approx(1.0)
        assert fees["n_trades"] == 2

    def test_does_not_leak_global_rng_state(self):
        np.random.seed(7)
        expected = np.random.random()
        np.random.seed(7)
        _run()
        assert np.random.random() == expected

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"trade_returns": []}, "empty"),
            ({"trade_returns": [0.01, float("nan")]}, "finite"),
            ({"trade_returns": ["not-a-number"]}, "numeric"),
            ({"run_id": " "}, "run_id"),
            ({"dataset_ref": ""}, "dataset_ref"),
            ({"starting_equity": 0.0}, "starting_equity"),
            ({"num_simulations": 0}, "positive"),
            ({"walk_forward_splits": 1}, "walk_forward_splits"),
            ({"walk_forward_splits": 99}, "walk_forward_splits"),
            ({"fee_trades": [(1.0, 2.0)]}, "fee_config"),
            ({"fee_config": {"model": "mystery"}}, "fee_config.model"),
            ({"fee_config": {"model": "bounded_profit", "bogus": 1.0}}, "malformed"),
        ],
    )
    def test_malformed_inputs_are_rejected(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _run(**overrides)

    def test_sensitive_provenance_is_rejected(self):
        with pytest.raises(ValueError, match="sensitive"):
            _run(provenance={"api_key": "must-not-persist"})


class TestFailureEvidence:
    @pytest.mark.parametrize(
        "outcome", ["failure", "timeout", "denied", "unknown", "cancelled", "error"]
    )
    def test_non_success_states_are_explicit(self, outcome):
        evidence = build_failure_evidence(
            run_id="run-1", outcome=outcome, reason="provider unavailable"
        )
        assert evidence["receipt"]["outcome"] == outcome
        assert evidence["results"] is None
        assert evidence["results_hash"] is None

    def test_success_cannot_be_fabricated(self):
        with pytest.raises(ValueError, match="run_counterfactual"):
            build_failure_evidence(run_id="run-1", outcome="success", reason="nope")

    def test_unknown_outcome_and_empty_reason_rejected(self):
        with pytest.raises(ValueError, match="unknown outcome"):
            build_failure_evidence(run_id="run-1", outcome="mystery", reason="x")
        with pytest.raises(ValueError, match="reason"):
            build_failure_evidence(run_id="run-1", outcome="failure", reason=" ")


class TestVerificationResultExport:
    def test_success_maps_to_passed_with_bound_digests(self):
        evidence = _run()
        payload = to_verification_result(evidence, policy_requirement="BACK-001")
        assert payload["status"] == "passed"
        assert payload["check_type"] == "custom"
        assert payload["provenance"] == "verdict-backtest"
        assert evidence["results_hash"] in payload["artifact_digests"]
        assert all(d.startswith("sha256:") for d in payload["artifact_digests"])

    @pytest.mark.parametrize(
        ("outcome", "status"),
        [
            ("failure", "failed"),
            ("denied", "failed"),
            ("error", "failed"),
            ("timeout", "unknown"),
            ("unknown", "unknown"),
            ("cancelled", "skipped"),
        ],
    )
    def test_non_success_never_maps_to_passed(self, outcome, status):
        evidence = build_failure_evidence(run_id="run-1", outcome=outcome, reason="x")
        assert to_verification_result(evidence)["status"] == status

    def test_advisory_details_cannot_upgrade_status(self):
        # A tampered receipt that *claims* success in advisory fields still
        # exports from the recorded outcome, so provider data cannot weaken
        # a hard policy gate.
        evidence = copy.deepcopy(
            build_failure_evidence(run_id="run-1", outcome="failure", reason="x")
        )
        evidence["receipt"]["details"]["status"] = "passed"
        evidence["receipt"]["details"]["approved"] = True
        assert to_verification_result(evidence)["status"] == "failed"

    def test_outcome_outside_allowlist_is_rejected(self):
        evidence = copy.deepcopy(_run())
        evidence["receipt"]["outcome"] = "totally-approved"
        with pytest.raises(ValueError, match="unknown outcome"):
            to_verification_result(evidence)

    def test_tampered_digest_is_rejected(self):
        evidence = copy.deepcopy(_run())
        evidence["results_hash"] = "sha256:not-a-digest"
        with pytest.raises(ValueError, match="digest"):
            to_verification_result(evidence)

    def test_malformed_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            to_verification_result("not-a-mapping")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="receipt"):
            to_verification_result({"results": {}})


class TestEvidenceChainExport:
    def _link(self, evidence, **overrides):
        kwargs = {
            "decision": "allow",
            "policy": "policy://backtest/counterfactual",
            "envelope_hash": ENVELOPE_HASH,
            "runtime": "verdict-node",
            "model": "none",
            "timestamp": TIMESTAMP,
        }
        kwargs.update(overrides)
        return to_evidence_chain_link(evidence, **kwargs)

    def test_link_embeds_verification_and_outcome(self):
        evidence = _run()
        link = self._link(evidence)
        assert link["provider"] == "verdict-backtest"
        assert link["outcome"] == "success"
        assert link["verification"] == [to_verification_result(evidence)]
        assert link["outcome"] in COUNTERFACTUAL_OUTCOMES

    def test_provider_cannot_mint_decision_authority(self):
        evidence = _run()
        with pytest.raises(ValueError, match="decision"):
            self._link(evidence, decision=" ")
        with pytest.raises(ValueError, match="policy"):
            self._link(evidence, policy="")
        with pytest.raises(ValueError, match="envelope_hash"):
            self._link(evidence, envelope_hash="sha256:short")
        with pytest.raises(ValueError, match="previous_hash"):
            self._link(evidence, previous_hash="bogus")
        with pytest.raises(ValueError, match="timestamp"):
            self._link(evidence, timestamp="yesterday")

    def test_failure_link_carries_failed_verification(self):
        evidence = build_failure_evidence(run_id="run-1", outcome="timeout", reason="x")
        link = self._link(evidence)
        assert link["outcome"] == "timeout"
        assert link["verification"][0]["status"] == "unknown"
