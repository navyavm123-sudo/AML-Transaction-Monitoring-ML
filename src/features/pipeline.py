"""Feature engineering and preprocessing pipeline for AML Transaction Monitoring."""

import re
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd


def clean_account_id(val: Any) -> float:
    """Extract numeric value from account string or number for standard scaling.

    Handles formats like 'ACC-891024', 891024, '8724731955', or None.
    """
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(val) else 0.0
    # Extract consecutive digits
    digits = re.sub(r"\D", "", str(val))
    return float(digits) if digits else 0.0


def engineer_features(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
) -> pd.DataFrame:
    """Perform AML domain feature transformations on incoming transactions.

    Transforms raw transaction dictionary/DataFrame to match the feature schema
    produced during model training (XGBoost No SMOTE pipeline):
    - Account ID digitization
    - Non-negative amount clipping and log1p scaling
    - Date & weekend extraction
    - Time & night-time indicator extraction
    - Cross-border transaction indicator
    - Same-currency indicator
    - Laundering_type placeholder (preventing target leakage)

    Args:
        data: Single transaction dict, list of transaction dicts, or pandas DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame with all engineered feature columns.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Unsupported data type for feature engineering: {type(data)}")

    # 1. Clean and standardize Account IDs
    if "Sender_account" in df.columns:
        df["Sender_account"] = df["Sender_account"].apply(clean_account_id)
    else:
        df["Sender_account"] = 0.0

    if "Receiver_account" in df.columns:
        df["Receiver_account"] = df["Receiver_account"].apply(clean_account_id)
    else:
        df["Receiver_account"] = 0.0

    # 2. Transaction Amount
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
        df["Amount"] = np.maximum(df["Amount"], 0.0)
    else:
        df["Amount"] = 0.0

    df["Log_amount"] = np.log1p(df["Amount"])
    amount_lower, amount_upper = -10328.474999999999, 22930.625
    df["Amount_outlier"] = ((df["Amount"] < amount_lower) | (df["Amount"] > amount_upper)).astype(int)

    # 3. Date & Weekend Indicator
    date_col = df["Date"] if "Date" in df.columns else pd.Series(["2023-01-01"] * len(df))
    date_dt = pd.to_datetime(date_col, errors="coerce")
    df["Year"] = date_dt.dt.year.fillna(2022).astype(int)
    df["Month"] = date_dt.dt.month.fillna(1).astype(int)
    df["Day"] = date_dt.dt.day.fillna(1).astype(int)
    day_of_week = getattr(date_dt.dt, "day_of_week", date_dt.dt.dayofweek).fillna(0).astype(int)
    df["DayOfWeek"] = day_of_week
    df["Is_weekend"] = (day_of_week >= 5).astype(int)

    # 4. Time & Night Indicator
    time_col = df["Time"] if "Time" in df.columns else pd.Series(["12:00:00"] * len(df))
    time_series = pd.to_datetime(time_col, format="%H:%M:%S", errors="coerce")
    hour = time_series.dt.hour.fillna(12).astype(int)
    df["Hour"] = hour
    df["Is_night"] = ((hour < 6) | (hour >= 22)).astype(int)

    # Standardize Time format if needed
    if "Time" not in df.columns:
        df["Time"] = "12:00:00"
    else:
        df["Time"] = df["Time"].astype(str).fillna("12:00:00")

    # 5. Currency & Location Indicators
    pay_curr = df["Payment_currency"].astype(str) if "Payment_currency" in df.columns else pd.Series(["Unknown"] * len(df))
    rec_curr = df["Received_currency"].astype(str) if "Received_currency" in df.columns else pd.Series(["Unknown"] * len(df))
    df["Same_currency"] = (pay_curr == rec_curr).astype(int)

    send_loc = df["Sender_bank_location"].astype(str) if "Sender_bank_location" in df.columns else pd.Series(["Unknown"] * len(df))
    rec_loc = df["Receiver_bank_location"].astype(str) if "Receiver_bank_location" in df.columns else pd.Series(["Unknown"] * len(df))
    df["Cross_border"] = (send_loc != rec_loc).astype(int)

    # Categorical defaults
    if "Payment_type" not in df.columns:
        df["Payment_type"] = "Unknown"
    else:
        df["Payment_type"] = df["Payment_type"].astype(str).fillna("Unknown")

    if "Payment_currency" not in df.columns:
        df["Payment_currency"] = pay_curr
    if "Received_currency" not in df.columns:
        df["Received_currency"] = rec_curr
    if "Sender_bank_location" not in df.columns:
        df["Sender_bank_location"] = send_loc
    if "Receiver_bank_location" not in df.columns:
        df["Receiver_bank_location"] = rec_loc

    # 6. Laundering_type handling (prevent leakage, provide valid category for OneHotEncoder)
    if "Laundering_type" not in df.columns or df["Laundering_type"].isna().any():
        existing = df.get("Laundering_type", pd.Series(["Unknown"] * len(df)))
        df["Laundering_type"] = existing.fillna("Unknown").astype(str)

    return df
