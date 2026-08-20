"""Deterministic counterfactual evaluation and Core-compatible evidence export.

This module exposes Monte Carlo, fee, walk-forward, and tear-sheet results as
verification / counterfactual evidence (BACK-001, ADR-021):

* :func:`run_counterfactual` — seeded, reproducible evaluation of a historical
  returns series.  Pure in-process math: no network, no live execution, no
  order placement.
* :func:`build_failure_evidence` — explicit receipts for failure, timeout,
  denial, unavailable, and unknown states so an absent result is never read as
  a pass.
* :func:`to_verification_result` — export as a verdict-core
  ``VerificationResult`` payload.  Status derives only from the recorded
  outcome; a non-success outcome can never map to ``"passed"``, and advisory
  detail fields cannot override the mapping.
* :func:`to_evidence_chain_link` — export as a verdict-core
  ``EvidenceChainLink`` payload.  All decision-authority fields (decision,
  policy, envelope hash, model, timestamp) MUST be supplied by the caller;
  this provider is evidence-only and cannot grant or fabricate authority.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from backtest_harness.analytics import split_walk_forward, tearsheet
from backtest_harness.fee_models import BoundedProfitFeeModel, FlatMakerTakerModel
from backtest_harness.monte_carlo import MonteCarloSimulator
from backtest_harness.provider_receipts import (
    PROVIDER_VERSION,
    build_backtest_receipt,
    canonical_hash,
)

EVIDENCE_SCHEMA_VERSION = "1"
PROVIDER_NAME = "verdict-backtest"

# Outcome allowlist mirroring verdict-core ``_OUTCOME_VALUES``.  Failure,
# timeout, denial, and unknown states are explicit, never inferred.
COUNTERFACTUAL_OUTCOMES = frozenset(
    {
        "success",
        "failure",
        "partial",
        "denied",
        "unknown",
        "cancelled",
        "timeout",
        "error",
        "skipped",
    }
)

# Deterministic outcome -> VerificationResult.status mapping.  ``"passed"``
# appears exactly once: only a recorded ``"success"`` can produce it, so no
# advisory/provider data can weaken a hard policy gate by upgrading a
# non-success run.  Inconclusive states stay ``"unknown"`` (verdict-core keeps
# ``unknown`` distinct from ``passed`` for the same reason).
_OUTCOME_TO_STATUS: Mapping[str, str] = {
    "success": "passed",
    "failure": "failed",
    "denied": "failed",
    "error": "failed",
    "partial": "unknown",
    "timeout": "unknown",
    "unknown": "unknown",
    "cancelled": "skipped",
    "skipped": "skipped",
}

# Mirrors verdict-core's digest and ISO-8601 timestamp patterns so exports
# fail loudly at this boundary even when verdict-core is not installed.
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ISO_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)

_FEE_MODELS = ("bounded_profit", "flat_maker_taker")


def run_counterfactual(
    *,
    run_id: str,
    trade_returns: Sequence[float],
    starting_equity: float,
    seed: int,
    dataset_ref: str,
    num_simulations: int = 1000,
    trades_per_sim: int = 250,
    walk_forward_splits: int = 5,
    periods_per_year: int = 252,
    fee_config: Mapping[str, Any] | None = None,
    fee_trades: Sequence[tuple[float, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a seeded, reproducible counterfactual backtest evaluation.

    Records inputs, the random seed, the fee model, the code version, and the
    dataset reference in a receipt whose ``evidence_refs`` bind it to the
    canonical hash of the produced results.  Identical arguments produce an
    identical evidence bundle.  Pure in-process computation only.

    Args:
        run_id: Unique identifier for this evaluation run.
        trade_returns: Historical per-trade returns as fractions.
        starting_equity: Initial equity in base currency units.
        seed: Random seed for the Monte Carlo resampling (recorded).
        dataset_ref: Caller-supplied reference identifying the input dataset.
        num_simulations: Monte Carlo path count.
        trades_per_sim: Trades per simulated path.
        walk_forward_splits: Number of walk-forward blocks (>= 2).
        periods_per_year: Annualization factor for tear-sheet statistics.
        fee_config: Optional fee model description, e.g.
            ``{"model": "bounded_profit", "percent_of_profit": 0.07,
            "maximum_fee_cents": 0.05}`` or
            ``{"model": "flat_maker_taker", "maker_bps": 0.0,
            "taker_bps": 0.001}``.
        fee_trades: Trades evaluated under ``fee_config``:
            ``(entry_cents, payout_cents)`` pairs for ``bounded_profit`` or
            ``(trade_volume, is_maker)`` pairs for ``flat_maker_taker``.
        provenance: Optional redacted provenance mapping for the receipt.

    Returns:
        Evidence bundle ``{"schema_version", "receipt", "results",
        "results_hash"}``.

    Raises:
        ValueError: On malformed input (empty/NaN returns, non-positive
            equity or simulation counts, unknown fee model, bad splits).
    """
    returns = _validate_returns(trade_returns)
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not dataset_ref.strip():
        raise ValueError("dataset_ref must be non-empty")
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    if num_simulations <= 0 or trades_per_sim <= 0:
        raise ValueError("num_simulations and trades_per_sim must be positive")
    if not 2 <= walk_forward_splits <= returns.size:
        raise ValueError("walk_forward_splits must be >= 2 and <= the number of return periods")
    fee_summary = _evaluate_fees(fee_config, fee_trades)

    monte_carlo = _seeded_monte_carlo(
        returns=returns,
        starting_equity=starting_equity,
        seed=seed,
        num_simulations=num_simulations,
        trades_per_sim=trades_per_sim,
    )
    sheet = tearsheet(returns, periods_per_year=periods_per_year)
    walk_forward = _walk_forward_report(returns, walk_forward_splits, periods_per_year)

    results: dict[str, Any] = {
        "monte_carlo": monte_carlo,
        "tearsheet": sheet,
        "walk_forward": walk_forward,
        "fees": fee_summary,
    }
    results_hash = canonical_hash(results)
    inputs = {
        "dataset_ref": dataset_ref,
        "trade_returns_hash": canonical_hash([float(v) for v in returns]),
        "n_periods": int(returns.size),
    }
    config = {
        "seed": int(seed),
        "starting_equity": float(starting_equity),
        "num_simulations": int(num_simulations),
        "trades_per_sim": int(trades_per_sim),
        "walk_forward_splits": int(walk_forward_splits),
        "periods_per_year": int(periods_per_year),
        "fee_config": dict(fee_config) if fee_config is not None else None,
        "code_version": PROVIDER_VERSION,
    }
    receipt = build_backtest_receipt(
        run_id=run_id,
        inputs=inputs,
        config=config,
        outcome="success",
        evidence_refs=(results_hash,),
        provenance=provenance,
        details={"kind": "counterfactual_backtest"},
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "receipt": receipt,
        "results": results,
        "results_hash": results_hash,
    }


def build_failure_evidence(
    *,
    run_id: str,
    outcome: str,
    reason: str,
    dataset_ref: str = "",
    config: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an explicit non-success state as an evidence bundle.

    Use for failure, timeout, denied, cancelled, or unknown states so the
    absence of results is itself auditable.  ``outcome`` must be a
    non-success member of the outcome allowlist; fabricated results are
    impossible because the bundle carries none.
    """
    if outcome not in COUNTERFACTUAL_OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome}")
    if outcome == "success":
        raise ValueError("success evidence must come from run_counterfactual")
    if not reason.strip():
        raise ValueError("reason must be non-empty")
    receipt = build_backtest_receipt(
        run_id=run_id,
        inputs={"dataset_ref": dataset_ref},
        config={**dict(config or {}), "code_version": PROVIDER_VERSION},
        outcome=outcome,
        provenance=provenance,
        details={"kind": "counterfactual_backtest", "reason": reason},
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "receipt": receipt,
        "results": None,
        "results_hash": None,
    }


def to_verification_result(
    evidence: Mapping[str, Any],
    *,
    check_name: str = "backtest_counterfactual",
    policy_requirement: str = "",
    command: str = "",
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Export an evidence bundle as a Core ``VerificationResult`` payload.

    ``status`` derives solely from the receipt outcome via the fixed
    non-upgradable mapping; advisory fields in ``details`` or the receipt
    cannot override it.
    """
    receipt = _require_receipt(evidence)
    outcome = receipt["outcome"]
    if outcome not in _OUTCOME_TO_STATUS:
        raise ValueError(f"unknown outcome: {outcome}")
    digests = [receipt["inputs_hash"], receipt["config_hash"]]
    results_hash = evidence.get("results_hash")
    if results_hash is not None:
        digests.append(results_hash)
    for digest in digests:
        if not _DIGEST_PATTERN.match(str(digest)):
            raise ValueError(f"invalid artifact digest: {digest}")
    # Core contracts reject nested nulls in generic payload maps, so absent
    # values are omitted rather than carried as ``None``.
    details: dict[str, Any] = {"receipt": dict(receipt)}
    if results_hash is not None:
        details["results_hash"] = results_hash
    payload: dict[str, Any] = {
        "check_name": check_name,
        "check_type": "custom",
        "status": _OUTCOME_TO_STATUS[outcome],
        "details": details,
        "artifact_digests": digests,
        "command": command,
        "runtime": f"backtest_harness=={PROVIDER_VERSION}",
        "provenance": PROVIDER_NAME,
        "policy_requirement": policy_requirement,
        "raw_output": "",
        "schema_version": "1",
    }
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return payload


def to_evidence_chain_link(
    evidence: Mapping[str, Any],
    *,
    decision: str,
    policy: str,
    envelope_hash: str,
    runtime: str,
    model: str,
    timestamp: str,
    previous_hash: str = "",
    tools: Sequence[str] = (),
    changes: Sequence[str] = (),
) -> dict[str, Any]:
    """Export an evidence bundle as a Core ``EvidenceChainLink`` payload.

    Every decision-authority field is caller-supplied and validated; this
    provider only contributes the verification payload and its outcome.  It
    cannot mint decisions, policies, or envelopes.
    """
    receipt = _require_receipt(evidence)
    for name, value in (
        ("decision", decision),
        ("policy", policy),
        ("runtime", runtime),
        ("model", model),
    ):
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
    if not _DIGEST_PATTERN.match(envelope_hash):
        raise ValueError(f"invalid envelope_hash: {envelope_hash}")
    if previous_hash and not _DIGEST_PATTERN.match(previous_hash):
        raise ValueError(f"invalid previous_hash: {previous_hash}")
    if not _ISO_TIMESTAMP.match(timestamp):
        raise ValueError(f"invalid timestamp: {timestamp}")
    outcome = receipt["outcome"]
    if outcome not in COUNTERFACTUAL_OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome}")
    return {
        "decision": decision,
        "policy": policy,
        "envelope_hash": envelope_hash,
        "runtime": runtime,
        "provider": PROVIDER_NAME,
        "model": model,
        "tools": list(tools),
        "changes": list(changes),
        "verification": [to_verification_result(evidence)],
        "outcome": outcome,
        "timestamp": timestamp,
        "previous_hash": previous_hash,
        "schema_version": "1",
    }


def _validate_returns(trade_returns: Sequence[float]) -> np.ndarray:
    try:
        returns = np.asarray(trade_returns, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trade_returns must be numeric: {exc}") from exc
    if returns.size == 0:
        raise ValueError("trade_returns must not be empty")
    if not np.all(np.isfinite(returns)):
        raise ValueError("trade_returns must contain only finite values")
    return returns


def _seeded_monte_carlo(
    *,
    returns: np.ndarray,
    starting_equity: float,
    seed: int,
    num_simulations: int,
    trades_per_sim: int,
) -> dict[str, Any]:
    """Run the simulator under a saved/restored seeded global RNG state."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return MonteCarloSimulator.simulate_equity_paths(
            trade_returns_pct=returns,
            starting_equity=starting_equity,
            num_simulations=num_simulations,
            trades_per_sim=trades_per_sim,
        )
    finally:
        np.random.set_state(state)


def _walk_forward_report(
    returns: np.ndarray, n_splits: int, periods_per_year: int
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(
        split_walk_forward(returns, n_splits=n_splits), start=1
    ):
        test_sheet = tearsheet(returns[test_idx], periods_per_year=periods_per_year)
        folds.append(
            {
                "fold": fold,
                "train_size": int(train_idx.size),
                "test_size": int(test_idx.size),
                "test_total_return": test_sheet["total_return"],
                "test_sharpe": test_sheet["sharpe"],
                "test_max_drawdown": test_sheet["max_drawdown"],
            }
        )
    return folds


def _evaluate_fees(
    fee_config: Mapping[str, Any] | None,
    fee_trades: Sequence[tuple[float, Any]] | None,
) -> dict[str, Any] | None:
    if fee_config is None:
        if fee_trades:
            raise ValueError("fee_trades requires fee_config")
        return None
    config = dict(fee_config)
    model_name = config.pop("model", None)
    if model_name not in _FEE_MODELS:
        raise ValueError(f"fee_config.model must be one of {sorted(_FEE_MODELS)}")
    trades = list(fee_trades or ())
    try:
        if model_name == "bounded_profit":
            bounded = BoundedProfitFeeModel(**config)
            fees = [bounded.calculate_fee(float(entry), float(payout)) for entry, payout in trades]
        else:
            flat = FlatMakerTakerModel(**config)
            fees = [
                flat.calculate_fee(float(volume), bool(is_maker)) for volume, is_maker in trades
            ]
    except TypeError as exc:
        raise ValueError(f"malformed fee_config or fee_trades: {exc}") from exc
    total = float(sum(fees))
    return {
        "model": model_name,
        "params": {key: float(value) for key, value in config.items()},
        "n_trades": len(fees),
        "total_fee": total,
        "mean_fee": total / len(fees) if fees else 0.0,
    }


def _require_receipt(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")
    receipt = evidence.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("evidence.receipt must be a mapping")
    for field in ("outcome", "inputs_hash", "config_hash"):
        if field not in receipt:
            raise ValueError(f"evidence.receipt missing field: {field}")
    return receipt


__all__ = [
    "COUNTERFACTUAL_OUTCOMES",
    "EVIDENCE_SCHEMA_VERSION",
    "PROVIDER_NAME",
    "build_failure_evidence",
    "run_counterfactual",
    "to_evidence_chain_link",
    "to_verification_result",
]
