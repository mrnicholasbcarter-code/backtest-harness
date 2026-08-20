"""Backtest Harness - Monte Carlo Simulation Engine & Fee Evaluator.

A high-performance Monte Carlo simulation engine for predictive market backtesting
with built-in fee models for Kalshi (bounded profit) and Polymarket (flat maker-taker).

Example:
    >>> import numpy as np
    >>> from backtest_harness import MonteCarloSimulator, BoundedProfitFeeModel, tearsheet
    >>>
    >>> # Simulate equity paths
    >>> returns = np.array([0.05, -0.10, 0.02, 0.15, -0.05])
    >>> stats = MonteCarloSimulator.simulate_equity_paths(
    ...     trade_returns_pct=returns,
    ...     starting_equity=1000.0,
    ...     num_simulations=5000,
    ...     trades_per_sim=250
    ... )
    >>> print(f"Risk of Ruin: {stats['prob_ruin'] * 100:.1f}%")
    >>>
    >>> # Calculate Kalshi-style fees
    >>> fee_model = BoundedProfitFeeModel(percent_of_profit=0.07, maximum_fee_cents=0.05)
    >>> fee = fee_model.calculate_fee(entry_cents=50.0, payout_cents=100.0)
    >>>
    >>> # Tearsheet
    >>> t = tearsheet(returns, periods_per_year=252)
    >>> print(f"Sharpe: {t['sharpe']:.2f}")
"""

from backtest_harness.analytics import split_walk_forward, tearsheet
from backtest_harness.evidence import (
    build_failure_evidence,
    run_counterfactual,
    to_evidence_chain_link,
    to_verification_result,
)
from backtest_harness.fee_models import (
    BoundedProfitFeeModel,
    FeeModel,
    FlatMakerTakerModel,
)
from backtest_harness.monte_carlo import MonteCarloSimulator
from backtest_harness.provider_receipts import (
    PROVIDER_VERSION,
    build_backtest_receipt,
    canonical_hash,
)

__version__ = PROVIDER_VERSION

__all__ = [
    "BoundedProfitFeeModel",
    "FeeModel",
    "FlatMakerTakerModel",
    "MonteCarloSimulator",
    "build_backtest_receipt",
    "build_failure_evidence",
    "canonical_hash",
    "run_counterfactual",
    "split_walk_forward",
    "tearsheet",
    "to_evidence_chain_link",
    "to_verification_result",
]
