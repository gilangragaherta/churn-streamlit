import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================
st.title("📉 Customer Churn Prediction")

st.caption(
    "Snapshot-based churn model using 3-month (90 days) customer behavior. "
    "Each row represents one customer snapshot — NOT a time series model."
)

# ======================================================
# SIDEBAR — MODEL INFO & THRESHOLD
# ======================================================
st.sidebar.header("ℹ️ Model Information")

st.sidebar.markdown(
    """
**Model Type**
- Snapshot-based (90 days)
- Binary churn classification

**Churn Label Definition (Pre-Modeling)**
- p75 inter-transaction ≈ 7 days  
- Std deviation ≈ 7.5 days  
- **Churn defined as ≥ 14 days without transaction**

**Purpose**
- Decision-support model
- Used for prioritization, not deterministic prediction
"""
)

st.sidebar.divider()
st.sidebar.subheader("🎚️ Business Threshold")

PREDICTION_THRESHOLD = st.sidebar.slider(
    "Churn Probability Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    with open("customer_churn_model_xgb.sav", "rb") as f:
        return pickle.load(f)

model = load_model()
MODEL_FEATURES = list(model.feature_names_in_)

# ======================================================
# FEATURE PREPARATION (ROBUST)
# ======================================================
def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Snapshot-based feature preparation.
    MUST match training logic.
    Includes safety-net for missing columns.
    """

    df = df_raw.copy()

    # -------------------------
    # Feature engineering (same as training)
    # -------------------------
    df["avg_transaction_amount"] = (
        df.get("total_amount_90d", 0)
        / df.get("txn_count_90d", 0).replace(0, np.nan)
    ).fillna(0)

    df["txn_trend_30d_vs_60d"] = (
        df.get("txn_count_last_30d", 0)
        / df.get("txn_count_last_60d", 0).replace(0, np.nan)
    ).fillna(0)

    df["log_total_amount_90d"] = np.log1p(df.get("total_amount_90d", 0))
    df["log_total_amount_last_30d"] = np.log1p(df.get("total_amount_last_30d", 0))

    # -------------------------
    # SAFETY NET:
    # ensure all model features exist
    # -------------------------
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[MODEL_FEATURES]

# ======================================================
# CSV TEMPLATE DOWNLOAD
# ======================================================
st.subheader("📥 Download Input CSV Template")

template_columns = [
    "customer_id",
    "txn_count_90d",
    "total_amount_90d",
    "txn_count_last_30d",
    "txn_count_last_60d",
    "total_amount_last_30d",
    "days_since_first_trx",
    "transaction_consistency",
    "value_segment",
    "trx_B",
    "trx_F",
    "trx_O",
    "trx_T"
]

st.download_button(
    "Download CSV Template",
    pd.DataFrame(columns=template_columns).to_csv(index=False).encode("utf-8"),
    file_name="churn_input_template.csv",
    mime="text/csv"
)

st.divider()

# ======================================================
# BATCH PREDICTION
# ======================================================
st.header("📂 Batch Churn Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload customer snapshot data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    if st.button("🚀 Run Batch Churn Prediction", use_container_width=True):

        with st.spinner("Processing snapshot & predicting churn probability..."):
            X = prepare_features(df_raw)
            df_raw["churn_probability"] = model.predict_proba(X)[:, 1]
            df_raw["churn_prediction"] = (
                df_raw["churn_probability"] >= PREDICTION_THRESHOLD
            ).astype(int)

        st.success("✅ Batch churn prediction completed")

        # ==================================================
        # SUMMARY
        # ==================================================
        st.subheader("📊 Churn Overview")

        total_customer = len(df_raw)
        churn_customer = int(df_raw["churn_prediction"].sum())
        churn_rate = churn_customer / total_customer if total_customer else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", total_customer)
        c2.metric("Churn Customers", churn_customer)
        c3.metric("Churn Rate", f"{churn_rate:.2%}")

        # ==================================================
        # PRIORITY CUSTOMERS
        # ==================================================
        st.subheader("🎯 Customers to Prioritize")

        churn_customers = (
            df_raw[df_raw["churn_prediction"] == 1]
            .sort_values("churn_probability", ascending=False)
        )

        display_cols = ["churn_probability"]
        if "customer_id" in churn_customers.columns:
            display_cols.insert(0, "customer_id")

        st.dataframe(churn_customers[display_cols], use_container_width=True)

        # ==================================================
        # THRESHOLD vs VOLUME
        # ==================================================
        st.subheader("📈 Threshold vs Churn Volume Simulation")

        thresholds = np.arange(0.10, 0.91, 0.05)
        sim = []

        for t in thresholds:
            sim.append({
                "threshold": t,
                "churn_customers": int((df_raw["churn_probability"] >= t).sum())
            })

        sim_df = pd.DataFrame(sim)
        st.line_chart(sim_df.set_index("threshold"))

        # ==================================================
        # REVENUE AT RISK
        # ==================================================
        st.subheader("💰 Estimated Revenue at Risk")

        df_raw["revenue_at_risk"] = (
            df_raw["churn_probability"] * df_raw["total_amount_90d"]
        )

        total_revenue = df_raw["total_amount_90d"].sum()
        current_risk = df_raw.loc[
            df_raw["churn_probability"] >= PREDICTION_THRESHOLD,
            "revenue_at_risk"
        ].sum()

        r1, r2, r3 = st.columns(3)
        r1.metric("Total Revenue (90d)", f"Rp {total_revenue:,.0f}")
        r2.metric("Revenue at Risk", f"Rp {current_risk:,.0f}")
        r3.metric("% Revenue at Risk", f"{current_risk / total_revenue:.2%}")

        # ==================================================
        # DOWNLOAD
        # ==================================================
        st.subheader("⬇️ Download Results")

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download ALL Customers",
            df_raw.to_csv(index=False).encode("utf-8"),
            "customer_churn_prediction_all.csv"
        )
        d2.download_button(
            "Download CHURN Customers Only",
            churn_customers.to_csv(index=False).encode("utf-8"),
            "customer_churn_high_risk.csv"
        )

# ======================================================
# MANUAL SINGLE CUSTOMER CHECK
# ======================================================
st.divider()
st.header("🧪 Single Customer Churn Checker (Manual Input)")

with st.form("manual_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        txn_count_90d = st.number_input("Transaction Count (90d)", 0, value=10)
        total_amount_90d = st.number_input("Total Amount (90d)", 0.0, value=1_000_000.0)

    with c2:
        txn_count_last_30d = st.number_input("Transaction Count (Last 30d)", 0, value=3)
        txn_count_last_60d = st.number_input("Transaction Count (Last 60d)", 0, value=6)

    with c3:
        total_amount_last_30d = st.number_input("Total Amount (Last 30d)", 0.0, value=300_000.0)
        days_since_first_trx = st.number_input("Days Since First Transaction", 1, value=120)
        transaction_consistency = st.slider("Transaction Consistency", 0.0, 1.0, 0.5)
        value_segment = st.selectbox("Value Segment", ["Low", "Medium", "High", "Very High"])

    submit = st.form_submit_button("🔍 Check Churn Probability")

if submit:

    manual_df = pd.DataFrame([{
        "txn_count_90d": txn_count_90d,
        "total_amount_90d": total_amount_90d,
        "txn_count_last_30d": txn_count_last_30d,
        "txn_count_last_60d": txn_count_last_60d,
        "total_amount_last_30d": total_amount_last_30d,
        "days_since_first_trx": days_since_first_trx,
        "transaction_consistency": transaction_consistency,
        "value_segment": value_segment,

        # DEFAULT transaction-type flags (manual mode)
        "trx_B": 0,
        "trx_F": 0,
        "trx_O": 0,
        "trx_T": 0
    }])

    X_manual = prepare_features(manual_df)
    prob = model.predict_proba(X_manual)[0, 1]

    st.metric("Churn Probability", f"{prob:.2%}")
    st.metric(
        "Prediction",
        "CHURN" if prob >= PREDICTION_THRESHOLD else "NOT CHURN"
    )

# ======================================================
# FOOTER
# ======================================================
st.caption(
    "Customer Churn Prediction | Snapshot-based Model (90 days) | "
    "Decision-support application"
)
