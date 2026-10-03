"""Shared feature logic: used by training AND by the app, so they can never drift apart."""
import numpy as np
import pandas as pd

CAT_FEATURES = ["State", "Region", "Category", "Sub-Category"]
NUM_FEATURES = ["Sales", "Quantity", "Discount", "UnitPrice", "LogSales"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Superstore dataframe -> dataframe with all model features."""
    out = df.copy()
    out["Order Date"] = pd.to_datetime(out["Order Date"], format="%m/%d/%Y")
    out["DayOfWeek"] = out["Order Date"].dt.dayofweek
    out["Quarter"] = out["Order Date"].dt.quarter
    out["Month"] = out["Order Date"].dt.month
    out["UnitPrice"] = out["Sales"] / out["Quantity"] / (1 - out["Discount"])
    out["LogSales"] = np.log1p(out["Sales"])
    return out


def build_features(state, region, category, sub_category, quantity, discount, unit_price) -> pd.DataFrame:
    """One order (as entered in the app) -> one-row dataframe in the model's feature format."""
    sales = float(unit_price) * int(quantity) * (1 - float(discount))
    row = {
        "State": state, "Region": region, "Category": category, "Sub-Category": sub_category,
        "Sales": sales, "Quantity": int(quantity), "Discount": float(discount),
        "UnitPrice": float(unit_price), "LogSales": float(np.log1p(sales)),
    }
    return pd.DataFrame([row])[FEATURES]