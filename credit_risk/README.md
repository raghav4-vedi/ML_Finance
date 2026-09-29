# Credit risk

This project predicts whether a loan defaults within twelve months. Start with
[credit_risk_step_by_step.ipynb](credit_risk_step_by_step.ipynb), which compares
four logistic-regression feature sets with default gradient boosting. Denmark is
held out for a separate country check.

## Files

- `credit_risk_step_by_step.ipynb` contains the step-by-step analysis and saved results.
- `prepare_data.py` downloads the source workbook and prepares the notebook input.
- `credit_data.xlsx` is prepared locally for the notebook and excluded from Git.
- `loan_dataset_investor.xlsx` is the downloaded source data. It stays local because
  it is larger than GitHub's 100 MB file limit.

## Run

Use the environment setup in the [main README](../README.md), then open the notebook
and run its cells in order. Its first code cell calls `prepare_data()` in
`prepare_data.py` if `credit_data.xlsx` is missing. A newer source workbook may change
the sample and results.

To prepare the data separately, run this from the repository root:

```bash
.venv/bin/python credit_risk/prepare_data.py
```

The source is the public Go & Grow loan dataset:
https://sabanners001.blob.core.windows.net/statistics/public/loan_dataset_investor.xlsx

The dataset is supplied by Go & Grow and is not covered by this repository's MIT
licence. See [DATA_NOTICE.md](../DATA_NOTICE.md) for its source and terms.

The PDF in the repository root presents the study in report form.
