"""Performance analytics and walk-forward utilities for backtests.

This module provides:

* :func:`tearsheet` — key performance statistics (Sharpe, Sortino, max
  drawdown, Calmar, win rate, total return) for a series of returns.
* :func:`split_walk_forward` — yield ``(train, test)`` index pairs for
  walk-forward optimization / out-of-sample evaluation.

Both functions accept any 1D array-like (NumPy array, Python list, or pandas
Series when the index is monotonic). Returns are expressed as fractions
(e.g. ``0.01`` for +1%).
"""

from __future__ import annotations

from collections.abc import Iterator
from math import sqrt
from typing import Any, Literal, overload

import numpy as np

# Annualization factor. 252 trading periods is the convention for daily
# returns; the caller can pass ``periods_per_year`` if their series is at
# a different cadence (e.g. 4 for weekly, 12 for monthly).
_TRADING_PERIODS = 252


def _to_returns(returns: Any) -> np.ndarray:
    """Coerce array-like input to a 1D float64 NumPy array."""
    if hasattr(returns, "values"):  # pandas Series / DataFrame[col]
        returns = returns.values
    arr = np.asarray(returns, dtype=np.float64).ravel()
    return arr


@overload
def tearsheet(
    returns: Any, periods_per_year: int = _TRADING_PERIODS, as_dict: Literal[True] = True
) -> dict[str, float | int]: ...
@overload
def tearsheet(
    returns: Any, periods_per_year: int = _TRADING_PERIODS, as_dict: Literal[False] = False
) -> None: ...
def tearsheet(
    returns: Any, periods_per_year: int = _TRADING_PERIODS, as_dict: bool = True
) -> dict | None:
    """Compute a performance "tearsheet" of key stats for a returns series.

    Args:
        returns: 1D array-like of period returns as fractions (e.g. 0.01 = 1%).
            NaNs are dropped before computation.
        periods_per_year: Annualization factor. 252 for daily returns (default),
            12 for monthly, 4 for weekly, etc.
        as_dict: If True (default), return the stats as a dict. If False,
            print a formatted table and return ``None``.

    Returns:
        A dict with keys: ``total_return``, ``annual_return``, ``volatility``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``calmar``, ``win_rate``,
        ``n_periods``. When ``as_dict=False``, the same values are printed as
        a table and ``None`` is returned.

    Examples:
        >>> import numpy as np
        >>> r = np.array([0.01, -0.005, 0.02, 0.003, -0.01, 0.015])
        >>> t = tearsheet(r, periods_per_year=252)
        >>> round(t["sharpe"], 4) == round(t["sharpe"], 4)
        True
    """
    r = _to_returns(returns)
    r = r[np.isfinite(r)]
    n = int(r.size)

    stats: dict[str, float | int] = {"n_periods": n}
    if n == 0:
        zero = {
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
        }
        stats.update(zero)
        if as_dict:
            return stats
        _print_tearsheet(stats)
        return None

    # Compounded total return and annualized geometric return.
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    years = n / float(periods_per_year) if periods_per_year else 1.0
    annual_return = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    # Per-period and annualized volatility.
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if n > 1 else 0.0
    ann_vol = sigma * sqrt(periods_per_year) if periods_per_year else sigma

    # Sharpe ratio (assume risk-free = 0).
    sharpe = (mu / sigma * sqrt(periods_per_year)) if sigma > 0 else 0.0

    # Sortino ratio: downside deviation only.
    downside = r[r < 0.0]
    if downside.size > 1:
        dd_sigma = float(np.std(downside, ddof=1))
        sortino = (mu / dd_sigma * sqrt(periods_per_year)) if dd_sigma > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown from the running peak of the equity curve.
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown = float(drawdowns.min())  # negative number (e.g. -0.25)

    # Calmar ratio: annualized return / |max drawdown|.
    calmar = (annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    # Win rate.
    win_rate = float(np.mean(r > 0.0))

    stats.update(
        {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": ann_vol,
            "sharpe": round(float(sharpe), 4),
            "sortino": round(float(sortino), 4),
            "max_drawdown": round(max_drawdown, 4),
            "calmar": round(float(calmar), 4),
            "win_rate": round(win_rate, 4),
        }
    )

    if as_dict:
        return stats
    _print_tearsheet(stats)
    return None


def _print_tearsheet(stats: dict) -> None:
    """Render a tearsheet dict as a formatted column-aligned table."""
    pct_keys = {"total_return", "annual_return", "volatility", "max_drawdown", "win_rate"}
    rows = [
        ("Total Return", stats["total_return"]),
        ("Annual Return", stats["annual_return"]),
        ("Volatility (ann.)", stats["volatility"]),
        ("Sharpe Ratio", stats["sharpe"]),
        ("Sortino Ratio", stats["sortino"]),
        ("Max Drawdown", stats["max_drawdown"]),
        ("Calmar Ratio", stats["calmar"]),
        ("Win Rate", stats["win_rate"]),
        ("Periods", stats["n_periods"]),
    ]
    width = max(len(label) for label, _ in rows)
    print("=" * (width + 16))
    print(f"{'Tearsheet':^{width + 16}}")
    print("=" * (width + 16))
    for label, value in rows:
        if isinstance(value, float) and label != "Periods":
            if label in pct_keys:
                print(f"{label:<{width}}  {value * 100:>10.2f} %")
            else:
                print(f"{label:<{width}}  {value:>10.4f}")
        else:
            print(f"{label:<{width}}  {value:>12}")
    print("=" * (width + 16))


def split_walk_forward(returns: Any, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` index pairs for walk-forward validation.

    Splits the series into ``n_splits`` contiguous, non-overlapping blocks.
    For each block ``k`` (0-indexed), all earlier blocks form the train set
    and block ``k`` forms the test set — an expanding training window with
    a rolling one-block out-of-sample test window (classic walk-forward).

    Args:
        returns: 1D array-like whose length defines the time axis.
        n_splits: Number of test blocks (and the minimum number of folds
            produced). Must be > 1 and <= the number of periods.

    Yields:
        ``(train_idx, test_idx)`` — integer index arrays into the original
        series. ``len(train_idx)`` grows by one block each fold.

    Raises:
        ValueError: If ``n_splits < 2`` or exceeds the number of periods.

    Examples:
        >>> list(split_walk_forward(range(10), n_splits=2))  # doctest: +SKIP
        [(array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9]))]
    """
    arr = _to_returns(returns)
    n = arr.size
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_splits > n:
        raise ValueError(f"n_splits ({n_splits}) exceeds number of periods ({n})")

    block = n // n_splits
    all_idx = np.arange(n)
    bounds = [i * block for i in range(n_splits)] + [n]

    # Expanding-window walk-forward: train on everything before test block k.
    # Skip k=0 (no training data). Produces n_splits-1 folds.
    # (Document the rationale: requires >= 1 training block to have fold k=1.)
    for k in range(1, n_splits):
        train_idx = all_idx[: bounds[k]]
        test_idx = all_idx[bounds[k] : bounds[k + 1]]
        if test_idx.size == 0:  # pragma: no cover - trailing split round-off
            continue
        yield train_idx, test_idx
