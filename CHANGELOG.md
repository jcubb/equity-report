# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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