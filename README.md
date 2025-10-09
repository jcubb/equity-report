# Equity Factor Attribution Report

A Python-based tool for generating comprehensive equity factor attribution reports. This project analyzes portfolio performance relative to a benchmark (S&P 500) through factor decomposition and risk attribution analysis.

## Features

- **Factor Attribution Analysis**: Decomposes portfolio returns into alpha and factor components
- **Rolling Factor Regression**: Performs out-of-sample rolling regressions using optimized algorithms
- **Risk Analysis**: Calculates marginal tracking error by factor and sector
- **Visualization**: Generates comprehensive one-page PDF reports with multiple charts
- **Sector Analysis**: Provides allocation and selection effects by sector

## Factor Models

The analysis uses the following factor exposures:
- **Beta**: Market exposure (SPY)
- **Quality**: Quality factor differential (QUAL - SPY)
- **Value**: Value factor differential (VLUE - SPY) 
- **Momentum**: Momentum factor differential (MTUM - SPY)
- **Size**: Size factor differential (SIZE - SPY)
- **Min Volatility**: Low volatility factor differential (USMV - SPY)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/equity-report.git
cd equity-report
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have access to the required data files in your data hub directory.

## Usage

### Command Line Interface

```bash
python factor_reg.py --db "path/to/data-hub" --pfile "portfolio.csv"
```

**Arguments:**
- `--db, -d`: Path to data directory containing factor and market data (default: system-specific)
- `--pfile, -p`: Portfolio CSV file name (default: "ComstockFund_24Q3.csv")

### Portfolio File Format

Your portfolio CSV should contain the following columns:
- `Company`: Company name
- `Shares`: Number of shares held
- `Value`: Market value of holding
- `Ticker`: Stock ticker symbol

### Example

```bash
python factor_reg.py --db "C:/data/financial-data" --pfile "my_portfolio.csv"
```

This will generate:
- A comprehensive factor attribution analysis
- PDF report: `my_portfolio.pdf`
- Console output showing regression timing and statistics

## Output

The tool generates a one-page PDF report containing:

1. **Factor Contribution to Relative Performance**: Bar chart showing how each factor contributed to portfolio vs benchmark performance
2. **Relative Factor Betas**: Factor exposures relative to benchmark
3. **Marginal Tracking Error**: Risk contribution by factor (annualized)
4. **Alpha Allocation and Selection by Sector**: Combined view of sector allocation effects and relative weights
5. **Security-Level Analysis**: Top/bottom quartile holdings analysis by alpha selection and returns

## Data Requirements

The system expects pickled data files in your data directory:
- `spdrfactors.pickle`: SPDR factor ETF returns
- `sprtns.pickle`: Individual stock returns
- `spsect.pickle`: Stock sector classifications
- `sp500_history.pickle`: S&P 500 index weights over time

## Configuration

Key parameters can be modified in the script:
- `window = 100`: Rolling regression window (weeks)
- `horiz = 12`: Attribution horizon (weeks)
- `halflife = 52`: Risk model half-life (weeks)
- `start_date_data = '2012-09-01'`: Data start date

## Dependencies

- pandas
- numpy 
- matplotlib
- seaborn
- statsmodels
- adjustText
- numba
- concurrent.futures

## Project Structure

```
equity-report/
├── factor_reg.py          # Main analysis script
├── equity_risklib.py      # Risk and attribution analysis classes
├── equity_reglib.py       # Fast regression implementations
├── chartlib.py            # Plotting utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore rules
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Performance Notes

The system uses Numba-optimized functions for fast rolling regressions, capable of processing thousands of securities efficiently. Timing information is displayed in the console output.

## Troubleshooting

**Common Issues:**

1. **Missing sector data**: Securities without sector classifications are dropped from benchmark analysis
2. **Data alignment**: Ensure all data files have consistent date indices
3. **Memory usage**: Large universes may require increased memory allocation

For additional support, please open an issue on GitHub.