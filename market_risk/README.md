# Market risk

This project predicts unusually large increases in the German ten-year government
yield with quantile regression.

## Files

- `market_risk.py` runs the German government-yield analysis.
- `market_data.xlsx` contains the `german_yields` input worksheet.
- `market_results.xlsx` contains `german_yield_backtests` and
  `german_yield_forecasts`.

## Run

From this folder:

```bash
python3 market_risk.py
```

The observations come from the Deutsche Bundesbank. Its reuse terms require source
attribution and are summarised in `DATA_NOTICE.md` in the repository root. The
combined report contains the full explanation and results. The script saves its
report figure in the local `extras` folder.
