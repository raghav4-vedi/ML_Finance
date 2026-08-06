"""Forecast unusually large increases in the German ten-year government yield."""

# Libraries
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss as quantile_loss


# Files and analysis settings
ROOT = Path(__file__).resolve().parent
EXTRAS_DIR = ROOT.parent / "extras"
DATA_FILE = ROOT / "market_data.xlsx"
RESULTS_FILE = ROOT / "market_results.xlsx"
FIGURE_FILE = EXTRAS_DIR / "var_figure.pdf"

PERCENTILES = (0.975, 0.99)
HISTORY = 1_000
FEATURES = [
    "rv_1d",
    "rv_5d",
    "rv_20d",
    "rv_60d",
    "yield_2y",
    "yield_5y",
    "yield_10y",
    "yield_30y",
    "slope_10y_2y_bp",
    "slope_30y_10y_bp",
    "curvature_2_5_10_bp",
]

MODEL_NAMES = {
    "historical": "Historical percentile",
    "gradient_boosting": "Gradient-boosted quantile regression",
}
PERIOD_NAMES = {
    "validation": "Validation: 2017 to 2020",
    "test": "Test: 2021 onward",
}


# Step 1: Read the saved German government yields

def load_data():
    columns = ["date", "yield_2y", "yield_5y", "yield_10y", "yield_30y"]
    # Reading stored decimals as text keeps repeated tree fits reproducible.
    data = pd.read_excel(DATA_FILE, sheet_name="german_yields", dtype=str)[columns]
    data["date"] = pd.to_datetime(data["date"])
    data[columns[1:]] = data[columns[1:]].astype(float)
    return data.sort_values("date")


# Step 2: Create the features, target, and historical baseline

def create_dataset(data):
    data = data.copy()

    # Yield differences are converted from percentage points to basis points.
    data["change_10y_bp"] = 100 * data["yield_10y"].diff()
    data["slope_10y_2y_bp"] = 100 * (data["yield_10y"] - data["yield_2y"])
    data["slope_30y_10y_bp"] = 100 * (data["yield_30y"] - data["yield_10y"])
    data["curvature_2_5_10_bp"] = 100 * (
        2 * data["yield_5y"] - data["yield_2y"] - data["yield_10y"]
    )

    # Root-mean-square changes describe recent movement size.
    change = data["change_10y_bp"]
    for window in (1, 5, 20, 60):
        data[f"rv_{window}d"] = np.sqrt(change.pow(2).rolling(window).mean())

    # Each row predicts the yield change on the next available date.
    data["target_date"] = data["date"].shift(-1)
    data["target_change_bp"] = data["change_10y_bp"].shift(-1)

    # The baseline is the chosen percentile of the latest 1,000 changes.
    for percentile in PERCENTILES:
        suffix = str(percentile).replace("0.", "")
        data[f"historical_{suffix}"] = change.rolling(HISTORY).quantile(percentile)

    required = FEATURES + [
        "target_date",
        "target_change_bp",
        "historical_975",
        "historical_99",
    ]
    return data.dropna(subset=required)


# Step 3: Fit gradient-boosted quantile regression on earlier years

def make_predictions(data):
    evaluation = data.loc[data["date"] >= "2017-01-01"]
    predictions = evaluation[
        ["date", "target_date", "target_change_bp", "historical_975", "historical_99"]
    ].copy()
    predictions["gradient_boosting_975"] = np.nan
    predictions["gradient_boosting_99"] = np.nan

    last_year = int(evaluation["date"].dt.year.max())
    for year in range(2017, last_year + 1):
        start = pd.Timestamp(year, 1, 1)
        end = pd.Timestamp(year + 1, 1, 1)
        train = data.loc[data["target_date"] < start]
        score = evaluation.loc[
            evaluation["date"].between(start, end, inclusive="left")
        ]
        score_rows = predictions["date"].between(start, end, inclusive="left")

        for percentile in PERCENTILES:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=percentile,
                learning_rate=0.04,
                max_iter=80,
                max_leaf_nodes=10,
                min_samples_leaf=100,
                l2_regularization=3.0,
                early_stopping=False,
                random_state=42,
            )
            model.fit(train[FEATURES], train["target_change_bp"])
            suffix = str(percentile).replace("0.", "")
            predictions.loc[score_rows, f"gradient_boosting_{suffix}"] = (
                model.predict(score[FEATURES])
            )

    return predictions.dropna()


# Step 4: Compare validation and test errors

def evaluate(predictions):
    rows = []
    periods = {
        "validation": predictions["date"] < "2021-01-01",
        "test": predictions["date"] >= "2021-01-01",
    }

    for period, period_rows in periods.items():
        group = predictions.loc[period_rows]
        actual = group["target_change_bp"].to_numpy()
        for percentile in PERCENTILES:
            suffix = str(percentile).replace("0.", "")
            for model_name in MODEL_NAMES:
                forecast = group[f"{model_name}_{suffix}"].to_numpy()
                exceptions = actual > forecast
                rows.append(
                    {
                        "period": period,
                        "percentile": percentile,
                        "model": model_name,
                        "forecasts": len(group),
                        "exceptions": int(exceptions.sum()),
                        "expected_exceptions": len(group) * (1 - percentile),
                        "exception_rate": exceptions.mean(),
                        "quantile_loss": quantile_loss(
                            actual, forecast, alpha=percentile
                        ),
                    }
                )
    return pd.DataFrame(rows)


# Step 5: Save plain result tables and the report figure
def prepare_backtests(results):
    table = results.copy()
    table["period"] = table["period"].map(PERIOD_NAMES)
    table["model"] = table["model"].map(MODEL_NAMES)
    return table.rename(
        columns={
            "period": "Validation or test period",
            "percentile": "Forecast percentile",
            "model": "Model",
            "forecasts": "Number of forecasts",
            "exceptions": "Exceptions",
            "expected_exceptions": "Expected exceptions",
            "exception_rate": "Exception rate",
            "quantile_loss": "Mean quantile loss (basis points)",
        }
    )


def prepare_forecasts(predictions):
    return predictions.rename(
        columns={
            "date": "Forecast date",
            "target_date": "Target date",
            "target_change_bp": "Observed next-day yield change (basis points)",
            "historical_975": "Historical percentile: 97.5% forecast (basis points)",
            "gradient_boosting_975": "Gradient boosting: 97.5% forecast (basis points)",
            "historical_99": "Historical percentile: 99% forecast (basis points)",
            "gradient_boosting_99": "Gradient boosting: 99% forecast (basis points)",
        }
    )


def draw_figure(results):
    test = results.loc[
        (results["period"] == "test") & (results["percentile"] == 0.99)
    ].set_index("model")
    order = ["historical", "gradient_boosting"]
    labels = ["Historical percentile", "Gradient boosting"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(labels, test.loc[order, "exceptions"], color=["0.55", "#0072B2"])
    ax.bar_label(bars)
    ax.axhline(test["expected_exceptions"].iloc[0], color="black", linestyle="--")
    ax.set_ylabel("99% forecast exceptions")
    ax.set_title("German ten-year yield, test period")
    fig.tight_layout()
    fig.savefig(FIGURE_FILE)
    plt.close(fig)


def main():
    data = create_dataset(load_data())
    predictions = make_predictions(data)
    results = evaluate(predictions)

    with pd.ExcelWriter(RESULTS_FILE) as writer:
        prepare_backtests(results).to_excel(
            writer, sheet_name="german_yield_backtests", index=False
        )
        prepare_forecasts(predictions).to_excel(
            writer, sheet_name="german_yield_forecasts", index=False
        )
    EXTRAS_DIR.mkdir(exist_ok=True)
    draw_figure(results)
    print(prepare_backtests(results).to_string(index=False))


if __name__ == "__main__":
    main()
