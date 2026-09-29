"""Download and prepare the loan data used by the notebook."""

from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd


FOLDER = Path(__file__).resolve().parent
SOURCE_FILE = FOLDER / "loan_dataset_investor.xlsx"
OUTPUT_FILE = FOLDER / "credit_data.xlsx"
DATA_URL = (
    "https://sabanners001.blob.core.windows.net/statistics/public/"
    "loan_dataset_investor.xlsx"
)

COLUMNS = [
    "loan_id",
    "country",
    "loan_issued_at",
    "issued_amount",
    "initial_interest_rate",
    "initial_loan_duration",
    "combined_income",
    "has_default_within_12_months",
    "projected_npv_return",
    "customer_risk_rating",
]


def prepare_data():
    if OUTPUT_FILE.exists():
        return OUTPUT_FILE

    if not SOURCE_FILE.exists():
        urlretrieve(DATA_URL, SOURCE_FILE)

    loans = pd.read_excel(
        SOURCE_FILE,
        sheet_name="Loan Dataset",
        usecols=COLUMNS,
    ).dropna()

    loans["loan_issued_at"] = pd.to_datetime(loans["loan_issued_at"])
    loans = loans[
        loans["country"].isin(
            ["Estonia", "Finland", "Latvia", "Netherlands", "Denmark"]
        )
        & loans["loan_issued_at"].between(
            "2023-10-01", "2025-04-01", inclusive="left"
        )
    ]
    loans["has_default_within_12_months"] = loans[
        "has_default_within_12_months"
    ].astype(int)

    loans.to_excel(OUTPUT_FILE, sheet_name="loan_data", index=False)
    return OUTPUT_FILE


if __name__ == "__main__":
    prepare_data()
