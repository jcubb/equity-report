"""
Equity Factor Attribution Report Generator

A comprehensive tool for analyzing portfolio performance through factor attribution
and risk decomposition relative to benchmark indices.

This package provides:
- Factor regression analysis using rolling out-of-sample methodology
- Risk attribution by sector and factor
- Comprehensive reporting with visualizations
- High-performance implementations using Numba optimization

Main modules:
- factor_reg: Main analysis script and CLI interface
- equity_risklib: Risk and attribution analysis classes  
- equity_reglib: Optimized regression implementations
- chartlib: Plotting and visualization utilities

Example usage:
    from factor_reg import main
    main(['--pfile', 'my_portfolio.csv'])

For command line usage:
    python factor_reg.py --pfile my_portfolio.csv --db /path/to/data
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
from . import equity_risklib
from . import equity_reglib  
from . import chartlib
from .factor_reg import main

__all__ = [
    'equity_risklib',
    'equity_reglib', 
    'chartlib',
    'main'
]