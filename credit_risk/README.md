# Credit risk

This project predicts whether a loan defaults within twelve months. It compares
three logistic-regression feature sets with gradient boosting.

## Files

- `analysis.py` downloads the source workbook when needed and runs the analysis.
- `loan_dataset_investor.xlsx` is the downloaded source data. It stays local because
  it is larger than GitHub's 100 MB file limit.
- `credit_results.xlsx` contains `model_results` and `calibration_checks`.

## Run

From this folder:

```bash
python3 analysis.py
```

The source is the public Go & Grow loan dataset:
https://sabanners001.blob.core.windows.net/statistics/public/loan_dataset_investor.xlsx

The dataset is supplied by Go & Grow and is not covered by this repository's MIT
licence. See `DATA_NOTICE.md` in the repository root.

The combined report in the repository root contains the full explanation and
results. The script saves its report figure in the local `extras` folder.
