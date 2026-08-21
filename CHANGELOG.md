# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `equity_reglib.py` is now a thin shim that re-exports from the
  [fast-regression](https://github.com/jcubb/fast-regression) package, added to
  `requirements.txt` as a git dependency. The public API is unchanged — same names,
  same signatures, same properties.
- Scaffold placeholder URLs in `README.md` and `setup.py` now point at the real
  repository; `setup.py`'s Documentation link used `blob/main` on a repo whose default
  branch is `master`.

### Fixed
Picked up from upstream, none of which this repo's vendored copy had:
- **Threading now does something.** `@njit` does not release the GIL.
  `GroupFastRolling2sOOSOLS` fans out over ten worker threads and without
  `nogil=True` measured **0.98x** — ten threads doing nothing at all. Now 2.07x on
  the same workload.
- **Rank-deficient designs no longer return silent garbage.** `np.linalg.solve` does
  not detect rank deficiency; every solve site upstream is guarded by a
  tolerance-checked Cholesky.
- **~1.3x on the rolling kernels** from memory layout: a homogeneous float
  DataFrame's `.to_numpy()` is Fortran-ordered and `np.insert` preserves that, which
  made every rolling row slice non-contiguous. Designs are built C-ordered now.
- `FastOLS.cleanVars` re-aligned indexes on every call even when there was nothing to
  align.
- A dead `isinstance(y, pd.DataFrame)` branch meant a DataFrame `y` reached the kernel
  as a 2-D array.

### Note on behaviour
`nb_ols` now raises `numpy.linalg.LinAlgError` on a rank-deficient design rather than
returning silent garbage. Inside `GroupFastRolling2sOOSOLS` this is caught per column,
so an affected ticker is **absent** from `.results` rather than present-and-wrong. That
is the better failure, but it is a visible change, and it is swallowed unless you pass
`verbose=True`.

## [1.0.0] - 2025-10-09

### Added
- Initial release of equity factor attribution tool
- Command-line interface with argument parsing
- Factor regression analysis using rolling out-of-sample methodology
- Risk attribution by sector and factor
- Comprehensive PDF report generation
- Support for 6-factor model (Beta, Quality, Value, Momentum, Size, Min Volatility)
- High-performance Numba-optimized regression implementations
- Sector allocation and selection analysis
- Marginal tracking error calculations
- Interactive plotting with adjustText for label positioning

### Features
- Fast rolling 2-stage OLS regression with out-of-sample predictions
- Factor attribution with allocation and selection effects
- Risk model using exponentially weighted covariance matrices
- Multi-threaded portfolio analysis
- Method chaining support for pandas operations
- Comprehensive error handling and data validation

### Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- statsmodels >= 0.13.0
- adjustText >= 0.7.3
- numba >= 0.56.0

### Documentation
- Complete README with usage examples
- Contributing guidelines
- MIT License
- Setup.py for package installation

## [Unreleased]

### Planned
- Additional factor models
- Alternative risk models
- Enhanced visualization options
- Performance benchmarking
- Unit test suite
- Jupyter notebook examples