"""Train the final models and save everything the app needs.

Run:  python train_final.py
Outputs: models/clf.joblib, models/reg.joblib, models/meta.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CAT_FEATURES, FEATURES, NUM_FEATURES, add_training_features

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Sample - Superstore.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

# Hyper-parameters found by RandomizedSearchCV in the notebook
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=1, max_features=0.3, max_depth=12)
HGB_PARAMS = dict(learning_rate=0.03, max_iter=200, min_samples_leaf=10,
                  max_leaf_nodes=31, l2_regularization=1.0)


def make_prep():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CAT_FEATURES),
    ])


def make_reg():
    return Pipeline([("prep", make_prep()),
                     ("model", RandomForestRegressor(n_jobs=-1, random_state=42, **RF_PARAMS))])


def make_clf():
    return Pipeline([("prep", make_prep()),
                     ("model", HistGradientBoostingClassifier(random_state=42, **HGB_PARAMS))])


# ---------- load data ----------
raw = pd.read_csv(DATA, encoding="cp1252")
df = add_training_features(raw)
df["Is_Loss"] = (df["Profit"] < 0).astype(int)
year = df["Order Date"].dt.year          # used only to build the time split, not a model feature

X, y_reg, y_clf = df[FEATURES], df["Profit"], df["Is_Loss"]
idx = np.arange(len(df))

# ---------- 1) honest evaluation: models never see the rows they are scored on ----------
splits = {
    "random_split": train_test_split(idx, test_size=0.2, random_state=42, stratify=y_clf),
    "time_split_2017": (idx[year.values <= 2016], idx[year.values == 2017]),
}
metrics = {}
for scenario, (a, b) in splits.items():
    reg = make_reg().fit(X.iloc[a], y_reg.iloc[a])
    clf = make_clf().fit(X.iloc[a], y_clf.iloc[a])
    pred_profit = reg.predict(X.iloc[b])
    prob_loss = clf.predict_proba(X.iloc[b])[:, 1]

    m = {
        "profit_r2": r2_score(y_reg.iloc[b], pred_profit),
        "profit_mae": mean_absolute_error(y_reg.iloc[b], pred_profit),
        "profit_median_abs_error": median_absolute_error(y_reg.iloc[b], pred_profit),
        "loss_auc": roc_auc_score(y_clf.iloc[b], prob_loss),
    }
    for t in (0.5, 0.15):
        m[f"loss_precision_at_{t}"] = precision_score(y_clf.iloc[b], prob_loss >= t)
        m[f"loss_recall_at_{t}"] = recall_score(y_clf.iloc[b], prob_loss >= t)
    metrics[scenario] = {k: round(float(v), 3) for k, v in m.items()}
    print(scenario, metrics[scenario])

# ---------- 2) final models, trained on ALL the data ----------
reg_final = make_reg().fit(X, y_reg)
clf_final = make_clf().fit(X, y_clf)
joblib.dump(reg_final, OUT / "reg.joblib", compress=3)
joblib.dump(clf_final, OUT / "clf.joblib", compress=3)

# ---------- 3) metadata the app needs (dropdown options, valid ranges, metrics) ----------
# the app derives Region from State and Category from Sub-Category, so these must be one-to-one
assert df.groupby("State")["Region"].nunique().max() == 1
assert df.groupby("Sub-Category")["Category"].nunique().max() == 1

unit_price = df.groupby("Sub-Category")["UnitPrice"]
meta = {
    "sklearn_version": sklearn.__version__,
    "n_training_rows": int(len(df)),
    "training_date_range": [str(df["Order Date"].min().date()), str(df["Order Date"].max().date())],
    "states": sorted(df["State"].unique().tolist()),
    "state_to_region": df.drop_duplicates("State").set_index("State")["Region"].to_dict(),
    "sub_to_category": df.drop_duplicates("Sub-Category").set_index("Sub-Category")["Category"].to_dict(),
    "discount_values": sorted(df["Discount"].unique().round(2).tolist()),
    "quantity_range": [int(df["Quantity"].min()), int(df["Quantity"].max())],
    "sales_p99": round(float(df["Sales"].quantile(0.99)), 2),
    "unit_price_by_sub": {
        s: {"min": round(float(g.min()), 2), "p10": round(float(g.quantile(0.1)), 2),
            "median": round(float(g.median()), 2), "p90": round(float(g.quantile(0.9)), 2),
            "max": round(float(g.max()), 2)}
        for s, g in unit_price
    },
    "thresholds": {"balanced": 0.5, "catch_most_losses": 0.15},
    "metrics": metrics,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print("Saved models + meta.json")