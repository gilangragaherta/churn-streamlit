import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================================================
# CONFIGURATION
# ======================================================
PREDICTION_THRESHOLD = 0.4  # decision threshold (deployment)

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
# SIDEBAR — MODEL INFO
# ======================================================
st.sidebar.header("ℹ️ Model Information")

st.sidebar.markdown(
    """
    **Model Type**
    - Snapshot-based (90 days)
    - Binary churn classification

    **Churn Label Definition (Pre-Modeling)**
    - Based on transaction inactivity
    - p75 inter-transaction ≈ 7 days
    - Std deviation ≈ 7.5 days
    - **Churn defined as ≥ 14 days without transaction**

    **Deployment Decision Rule**
    - Model outputs churn probability
    - Probability threshold = **0.40**
    - Used to prioritize customers for action
    """
)

st.sidebar.divider()
st.sidebar.markdown("📂 **Upload customer snapshot data (CSV)**")

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
# FEATURE PREPARATION (DEPLOY CONTRACT)
# ======================================================
def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    SNAPSHOT-BASED FEATURE PREPARATION

    - Model trained on 3-month aggregated snapshot
    - NOT a time series model
    - Feature logic MUST match training
    """

    df = df_raw.copy()

    # -------------------------
    # REQUIRED RAW INPUT COLUMNS
    # -------------------------
    required_raw_cols = [
        'txn_count_90d',
        'total_amount_90d',
        'txn_count_last_30d',
        'txn_count_last_60d',
        'total_amount_last_30d',
        'days_since_first_trx',
        'transaction_consistency',
        'value_segment'
    ]

    missing = set(required_raw_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required snapshot columns: {missing}"
        )

    # -------------------------
    # FEATURE ENGINEERING (SAME AS TRAINING)
    # -------------------------
    df["avg_transaction_amount"] = (
        df["total_amount_90d"] / df["txn_count_90d"].replace(0, np.nan)
    ).fillna(0)

    df["txn_trend_30d_vs_60d"] = (
        df["txn_count_last_30d"] / df["txn_count_last_60d"].replace(0, np.nan)
    ).fillna(0)

    df["log_total_amount_90d"] = np.log1p(df["total_amount_90d"])
    df["log_total_amount_last_30d"] = np.log1p(df["total_amount_last_30d"])

    # -------------------------
    # FINAL FEATURE CONTRACT
    # -------------------------
    X = df[MODEL_FEATURES]

    return X

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "📂 Upload customer snapshot file (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    st.divider()

    if st.button("🚀 Run Churn Prediction", use_container_width=True):

        with st.spinner("Processing snapshot & predicting churn probability..."):

            try:
                X = prepare_features(df_raw)

                df_raw["churn_probability"] = model.predict_proba(X)[:, 1]
                df_raw["churn_prediction"] = (
                    df_raw["churn_probability"] >= PREDICTION_THRESHOLD
                ).astype(int)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.success("✅ Churn prediction completed")

        # ======================================================
        # SUMMARY METRICS
        # ======================================================
        st.subheader("📊 Churn Overview")

        total_customer = len(df_raw)
        churn_customer = int(df_raw["churn_prediction"].sum())
        churn_rate = churn_customer / total_customer if total_customer > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total_customer)
        col2.metric("Churn Customers", churn_customer)
        col3.metric("Churn Rate", f"{churn_rate:.2%}")

        st.divider()

        # ======================================================
        # CUSTOMERS TO PRIORITIZE
        # ======================================================
        st.subheader("🎯 Customers to Prioritize")

        churn_customers = (
            df_raw[df_raw["churn_prediction"] == 1]
            .sort_values("churn_probability", ascending=False)
        )

        if churn_customers.empty:
            st.success("🎉 No high-risk churn customers detected")
        else:
            display_cols = ["churn_probability"]
            if "customer_id" in churn_customers.columns:
                display_cols.insert(0, "customer_id")

            st.dataframe(
                churn_customers[display_cols],
                use_container_width=True
            )

        st.divider()

        # ======================================================
        # FULL RESULT
        # ======================================================
        with st.expander("📄 View Full Prediction Result"):
            st.dataframe(df_raw, use_container_width=True)

        # ======================================================
        # DOWNLOAD SECTION
        # ======================================================
        st.subheader("⬇️ Download Results")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.download_button(
                "Download ALL Customers",
                df_raw.to_csv(index=False).encode("utf-8"),
                file_name="customer_churn_prediction_all.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_d2:
            st.download_button(
                "Download CHURN Customers Only",
                churn_customers.to_csv(index=False).encode("utf-8"),
                file_name="customer_churn_high_risk.csv",
                mime="text/csv",
                use_container_width=True
            )

# ======================================================
# FOOTER
# ======================================================
st.caption(
    "Customer Churn Prediction | Snapshot-based Model (90 days) | "
    "Churn label defined using 14-day inactivity threshold"
)
