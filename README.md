# Machine Learning for Financial Models

A collection of two ML projects where I apply regression and classification to
government-yield risk and loan default. The notebooks contain the data features,
model fitting, predictions, and results.

## Folders

- `market_risk`: [German government-yield forecasting](market_risk/German_Bund.ipynb).
- `credit_risk`: [Twelve-month loan default](credit_risk/credit_risk_step_by_step.ipynb).

The notebooks provide the step-by-step analyses and saved outputs.
`financial_ml_portfolio.pdf` presents both studies in report form, and
`market_risk/market_results.xlsx` contains the market backtests and forecasts.

## Run the studies

Create an environment from the repository root (Python 3.11):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook
```

Open a notebook in its project folder and run the cells in order. German-yield
data are included. The credit notebook prepares `credit_data.xlsx` on its first
run, using the existing loan workbook or downloading it if missing. Preparation
can take several minutes; later runs reuse the file. Both loan workbooks stay
local and are excluded from Git.

The only supporting Python file is `credit_risk/prepare_data.py`. It downloads
and filters loan data; all modelling and evaluation are in the notebooks.

The market data come from the Deutsche Bundesbank. `credit_risk/prepare_data.py`
downloads the public Go & Grow workbook when it is missing. The MIT licence
applies to the code. Third-party data are covered by their providers' terms, as
described in [DATA_NOTICE.md](DATA_NOTICE.md).
