"""Order Profitability Estimator - Streamlit app (Superstore ML project)."""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

from features import build_features

ROOT = Path(__file__).parent
st.set_page_config(page_title="Order Profitability Estimator", page_icon="📦", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    clf = joblib.load(ROOT / "models" / "clf.joblib")
    reg = joblib.load(ROOT / "models" / "reg.joblib")
    return meta, clf, reg


meta, clf, reg = load_artifacts()
if meta["sklearn_version"] != sklearn.__version__:
    st.warning(f"Models were trained with scikit-learn {meta['sklearn_version']} but this environment has "
               f"{sklearn.__version__}. Re-run `python train_final.py` to avoid compatibility issues.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Alert sensitivity")
    mode = st.radio(
        "Loss-alert threshold",
        ["Balanced (0.50)", "Catch most losses (0.15)"],
        help="A lower threshold flags more orders as risky: it catches more real losses but raises more false alarms.",
    )
    threshold = meta["thresholds"]["balanced" if mode.startswith("Balanced") else "catch_most_losses"]
    m = meta["metrics"]["random_split"]
    key = "0.5" if mode.startswith("Balanced") else "0.15"
    st.caption(f"On held-out data at this setting: it caught **{m[f'loss_recall_at_{key}']:.0%}** of loss-making orders; "
               f"**{m[f'loss_precision_at_{key}']:.0%}** of its alerts were real losses.")
    st.divider()
    st.caption("Built with scikit-learn: HistGradientBoosting (loss classifier) and Random Forest (profit estimate), "
               "trained on the Superstore sample dataset.")

# ------------------------------------------------------------------ header
st.title("📦 Order Profitability Estimator")
st.write("Enter the details of an order to estimate **how likely it is to lose money** and its **expected profit**. "
         "Everything below is known when the order is placed.")

# ------------------------------------------------------------------ inputs
left, right = st.columns(2)

with left:
    st.subheader("Product")
    sub = st.selectbox("Sub-Category", sorted(meta["sub_to_category"]), index=sorted(meta["sub_to_category"]).index("Phones"))
    category = meta["sub_to_category"][sub]
    st.caption(f"Category: **{category}**")
    price_info = meta["unit_price_by_sub"][sub]
    # key depends on the sub-category so the default price resets when the sub-category changes
    unit_price = st.number_input(
        "Catalog unit price ($, before discount)", min_value=0.5, max_value=10000.0,
        value=float(round(price_info["median"], 2)), step=1.0, key=f"price_{sub}",
        help=f"Typical range for {sub} in the training data: ${price_info['p10']:,.0f} to ${price_info['p90']:,.0f}.",
    )
    qmin, qmax = meta["quantity_range"]
    quantity = st.number_input("Quantity", min_value=1, max_value=50, value=2, step=1)
    discount_pct = st.slider("Discount (%)", min_value=0, max_value=80, value=0, step=1, format="%d%%",
                             help="The training data only contains discounts of "
                                  + ", ".join(f"{d:.0%}" for d in meta["discount_values"])
                                  + ". Values in between are interpolated by the model.")
    discount = discount_pct / 100

with right:
    st.subheader("Location")
    state = st.selectbox("State", meta["states"], index=meta["states"].index("California"))
    region = meta["state_to_region"][state]
    st.caption(f"Region: **{region}**")

# ------------------------------------------------------------------ prediction
row = build_features(state, region, category, sub, quantity, discount, unit_price)
sales = float(row["Sales"].iloc[0])
p_loss = float(clf.predict_proba(row)[0, 1])
profit_est = float(reg.predict(row)[0])

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Order value (Sales)", f"${sales:,.2f}", help="Unit price x quantity x (1 - discount)")
c2.metric("Probability of a loss", f"{p_loss:.0%}")
c3.metric("Estimated profit", f"-${abs(profit_est):,.2f}" if profit_est < 0 else f"${profit_est:,.2f}")

if p_loss >= threshold:
    st.error(f"⚠️ **Likely loss-making.** Probability {p_loss:.0%} is above the alert threshold ({threshold:.0%}).")
else:
    st.success(f"✅ **Likely profitable.** Probability of a loss is {p_loss:.0%} (alert threshold {threshold:.0%}).")

# ------------------------------------------------------------------ the two models can disagree
margin = profit_est / sales if sales > 0 else 0
if p_loss < threshold and profit_est < 0:
    st.warning("⚠️ **Mixed signal:** the loss probability is below the alert threshold, but the expected profit is "
               "negative. Treat this order as risky.")
elif p_loss >= threshold and 0 < margin < 0.10:
    st.info("ℹ️ **Mixed signal:** the classifier flags a likely loss, but the expected profit is slightly positive. "
            "The order is probably close to break-even.")    

# ------------------------------------------------------------------ input sanity warnings
notes = []
if unit_price < price_info["min"] or unit_price > price_info["max"]:
    notes.append(f"The unit price is outside the range seen for **{sub}** in the training data "
                 f"(${price_info['min']:,.0f} to ${price_info['max']:,.0f}), so the estimate is an extrapolation.")
if quantity > qmax:
    notes.append(f"Quantity is above the largest order seen in training ({qmax}).")
if sales > meta["sales_p99"]:
    notes.append(f"This is a very large order (above ${meta['sales_p99']:,.0f}, the top 1% of training orders). "
                 "The profit estimate is least reliable for orders like this; treat it as a rough guide.")
if discount >= 0.3:
    notes.append("Discounts of 30% or more were loss-making on average in the training data.")
for n in notes:
    st.warning(n)

with st.expander("What drove this prediction? (the order as the model sees it)"):
    st.dataframe(row.T.rename(columns={0: "value"}).astype(str), width="stretch")

# ------------------------------------------------------------------ model card
with st.expander("How accurate is this? (model card)"):
    r, t = meta["metrics"]["random_split"], meta["metrics"]["time_split_2017"]
    card = pd.DataFrame({
        "Random 80/20 split": [r["loss_auc"], r["loss_recall_at_0.5"], r["loss_precision_at_0.5"],
                               r["profit_r2"], r["profit_median_abs_error"]],
        "Train 2014-16, test 2017": [t["loss_auc"], t["loss_recall_at_0.5"], t["loss_precision_at_0.5"],
                                     t["profit_r2"], t["profit_median_abs_error"]],
    }, index=["Loss detection: AUC", "Loss detection: recall (threshold 0.5)", "Loss detection: precision (threshold 0.5)",
              "Profit estimate: R²", "Profit estimate: median absolute error ($)"])
    st.dataframe(card.style.format("{:.3f}"), width="stretch")
    st.markdown(
        f"""
**How to read this**
- The **loss classifier is the more reliable part** (AUC about 0.99, stable across random, time-based and new-product tests).
- The **profit estimate** is accurate for typical orders (median error about $4) but can be far off for rare, very large orders. R² varies a lot between splits because of those orders.
- For products the model has never seen, the profit estimate is noticeably weaker (R² roughly 0.4 to 0.55 in grouped cross-validation).
- Trained on **{meta['n_training_rows']:,} orders** from {meta['training_date_range'][0]} to {meta['training_date_range'][1]} (sample dataset). This is a portfolio demo, not a production pricing system.
"""
    )