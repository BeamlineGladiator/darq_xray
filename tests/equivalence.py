"""Assert a chunked implementation is budget-independent (not a pytest file).

Every phase-5 conversion uses this: run the same callable at several memory
budgets and require bit-identical results. It is the executable form of the
guarantee that a laptop and the workstation produce the same data product.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DIVISORS = (1, 2, 3, 7, 1000)


def _materialise(result):
    """Normalise a result to something comparable with array_equal."""
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (int, float, np.floating, np.integer)):
        return np.asarray(result)
    # A generator/iterable of (slice, block) pairs, as two_pass yields.
    blocks = [block for _sl, block in result]
    return np.concatenate(blocks, axis=0) if blocks else np.asarray([])


def assert_budget_independent(fn, dset, *, budgets=None, nbytes=None) -> None:
    """Run *fn(dset, budget_bytes=b)* at several budgets; require identical bits.

    Compares with ``np.array_equal(..., equal_nan=True)`` so NaN placement is
    part of the guarantee — a chunked path that moves a NaN has changed the
    product just as surely as one that changes a number.
    """
    total = nbytes if nbytes is not None else dset.nbytes
    budgets = budgets or [max(1, int(total // d)) for d in DEFAULT_DIVISORS]
    reference = _materialise(fn(dset, budget_bytes=budgets[0]))
    for budget in budgets[1:]:
        candidate = _materialise(fn(dset, budget_bytes=budget))
        assert candidate.shape == reference.shape, (
            f"budget {budget}: shape {candidate.shape} != {reference.shape}"
        )
        assert np.array_equal(candidate, reference, equal_nan=True), (
            f"budget {budget} produced different bits than budget {budgets[0]}"
        )
