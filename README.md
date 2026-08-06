# Machine Learning for Financial Models

A collection of two ML projects where I apply regression and classification to
government-yield risk and loan default. The combined report explains the data,
calculations, models, and results.

## Folders

- `market_risk` contains German government-yield forecasting.
- `credit_risk` contains twelve-month loan-default classification.

The main PDF is `financial_ml_portfolio.pdf`. Each project folder contains its code,
data or download instructions, result workbook, and a short README. Report source
and generated figures are kept locally in `extras`.

## Run the studies

Create an environment from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the studies from the repository root:

```bash
.venv/bin/python credit_risk/analysis.py
.venv/bin/python market_risk/market_risk.py
```

The market data come from the Deutsche Bundesbank. The credit script downloads the
public Go & Grow workbook when it is missing. The MIT licence applies to the code.
Third-party data are covered by their providers' terms, as described in
`DATA_NOTICE.md`.
