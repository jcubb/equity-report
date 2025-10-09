# Contributing to Equity Factor Attribution Report

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/equity-report.git
   cd equity-report
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

Before submitting changes:

1. Test your changes with sample data
2. Ensure all existing functionality still works
3. Add tests for new features when applicable
4. Check for performance regressions in numerical computations

## Submitting Changes

1. Commit your changes with clear, descriptive commit messages
2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Create a Pull Request on GitHub
4. Describe your changes and their purpose clearly

## Areas for Contribution

We welcome contributions in these areas:

### New Features
- Additional factor models
- Alternative risk models
- New visualization types
- Performance optimizations
- Data source integrations

### Documentation
- Code documentation improvements
- Usage examples
- Tutorial notebooks
- Performance benchmarks

### Bug Fixes
- Edge case handling
- Data validation improvements
- Numerical stability enhancements

### Performance
- Algorithm optimizations
- Memory usage improvements
- Parallel processing enhancements

## Code Structure

- `factor_reg.py`: Main analysis script and CLI
- `equity_risklib.py`: Risk and attribution classes
- `equity_reglib.py`: Regression implementations (Numba-optimized)
- `chartlib.py`: Plotting utilities

## Key Considerations

- **Performance**: This tool processes large datasets, so maintain efficiency
- **Numerical Stability**: Financial computations require careful handling of edge cases
- **Data Alignment**: Ensure robust handling of missing data and date alignment
- **Backward Compatibility**: Avoid breaking existing functionality

## Questions?

If you have questions or need clarification, please:
1. Check existing issues on GitHub
2. Open a new issue for discussion
3. Reach out to maintainers

Thank you for contributing!