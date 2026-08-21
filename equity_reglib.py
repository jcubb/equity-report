# -*- coding: utf-8 -*-
"""
Compatibility shim. The regression implementations that used to live here now live
in the `fast-regression` package: https://github.com/jcubb/fast-regression

    pip install git+https://github.com/jcubb/fast-regression.git

WHY THIS MOVED
--------------
This file was one of four near-identical copies of the same code (here,
`equity-dashboard/core/equity_reglib.py`, `factor/reglib.py`, and the upstream
`fast-regression/reglib.py`). Copies drift, and these had: by August 2026 the
upstream had picked up five fixes that none of the forks had, including two that
were silently costing real performance and one that was silently producing wrong
numbers.

  * `@njit` does NOT release the GIL. `GroupFastRolling2sOOSOLS` fans out over ten
    worker threads, and without `nogil=True` measured 0.98x -- ten threads doing
    exactly nothing. Upstream now measures 2.07x on the same workload.
  * `np.linalg.solve` returns silent garbage on a rank-deficient design rather than
    raising. Every solve site upstream is now guarded by a tolerance-checked
    Cholesky.
  * A homogeneous float DataFrame's `.to_numpy()` is F-ordered and `np.insert`
    preserves that, which made every rolling row slice non-contiguous and cost
    ~1.3x. Designs are built C-ordered upstream.
  * `FastOLS.cleanVars` ran a full dropna / index-intersection / triple-`.loc`
    realignment on every call even when there was nothing to align.
  * `if isinstance(y, pd.DataFrame): y_pd = y.iloc[:, 0]` was dead code -- the very
    next line reassigned `y_pd` from the original `y`, so a DataFrame `y` reached
    the kernel as a 2-D array.

Upstream also carries 36 correctness tests and 65 asserted speed floors; this file
had neither.

BEHAVIOUR CHANGE TO BE AWARE OF
-------------------------------
`nb_ols` now RAISES `numpy.linalg.LinAlgError` on a rank-deficient design where it
previously returned silent garbage. Inside `GroupFastRolling2sOOSOLS` that lands in
the existing per-column `try/except`, so an affected column is now ABSENT from
`.results` rather than present-and-wrong. That is the better failure, but it is a
visible change -- and with `verbose=False` it is swallowed silently. Pass
`verbose=True` if a column goes missing unexpectedly.

The public API is unchanged: same names, same signatures, same properties.
"""
try:
    from reglib import (
        FastOLS,
        FastRolling2sOOSOLS,
        GroupFastRolling2sOOSOLS,
        nb_ols,
        nb_roll_2s_oos_ols,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "equity_reglib now re-exports from the 'fast-regression' package, which is "
        "not installed. Install it with:\n"
        "    pip install git+https://github.com/jcubb/fast-regression.git\n"
        "or reinstall this project's dependencies:\n"
        "    pip install -r requirements.txt"
    ) from exc

__all__ = [
    "nb_ols",
    "nb_roll_2s_oos_ols",
    "FastOLS",
    "FastRolling2sOOSOLS",
    "GroupFastRolling2sOOSOLS",
]
