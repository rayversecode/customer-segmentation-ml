import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Customer Segmentation", layout="wide")



st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    h1 { font-weight: 600; }
    .stButton button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Customer Segmentation")

st.caption("Predict which customer segment a new customer belongs to, using K-Means clustering and classification models.")

case_study = st.radio("Select dataset:", ["Mall Customers (Retail)", "Credit Card Customers (Banking)"], horizontal=True)



st.divider()

if case_study == "Mall Customers (Retail)":
    model = joblib.load("case_study_1_mall_customers/cluster_classifier.pkl")
    scaler = joblib.load("case_study_1_mall_customers/scaler.pkl")

    cluster_names = {
        0: "Mature Moderate Spenders",
        1: "Young Average Earners",
        2: "Young High Spenders (Low Credit)",
        3: "Premium VIP Customers",
        4: "Wealthy Savers"
    }


    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        income = st.number_input("Annual Income (k$)", min_value=15, max_value=140, value=50, step=1)
        spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)
    with col2:
        savings = st.number_input("Estimated Savings (k$)", min_value=5, max_value=120, value=30, step=1)
        credit = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=10)

    if st.button("Predict Segment", type="primary"):
        input_data = np.array([[age, income, spending, savings, credit]])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]
        st.metric("Predicted Segment", cluster_names.get(cluster, "Unknown"), f"Cluster {cluster}")


else:
    model = joblib.load("case_study_2_credit_card/cluster_classifier.pkl")
    scaler = joblib.load("case_study_2_credit_card/scaler.pkl")

    cluster_names = {
        0: "High-Value Active User",
        1: "Standard/Light User"
    }


    st.markdown("**Balance & Purchase Activity**")
    col1, col2, col3 = st.columns(3)
    with col1:
        balance = st.number_input("Balance", min_value=0, max_value=20000, value=1000, step=100)
        purchases = st.number_input("Purchases", min_value=0, max_value=50000, value=1000, step=100)
    with col2:
        oneoff = st.number_input("One-off Purchases", min_value=0, max_value=40000, value=500, step=100)
        installments = st.number_input("Installments Purchases", min_value=0, max_value=25000, value=500, step=100)
    with col3:
        balance_freq = st.number_input("Balance Frequency", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        purchases_freq = st.number_input("Purchases Frequency", min_value=0.0, max_value=1.0, value=0.5, step=0.05)


    st.markdown("**Cash Advances**")
    col1, col2, col3 = st.columns(3)
    with col1:
        cash_advance = st.number_input("Cash Advance", min_value=0, max_value=50000, value=0, step=100)
    with col2:
        cash_advance_freq = st.number_input("Cash Advance Frequency", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with col3:
        cash_advance_trx = st.number_input("Cash Advance Transactions", min_value=0, max_value=130, value=0, step=1)


    st.markdown("**Credit & Payments**")
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_limit = st.number_input("Credit Limit", min_value=0, max_value=30000, value=5000, step=500)
        purchases_trx = st.number_input("Purchases Transactions", min_value=0, max_value=300, value=10, step=1)
    with col2:
        payments = st.number_input("Payments", min_value=0, max_value=50000, value=1000, step=100)
        prc_full_payment = st.number_input("% Full Payment", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with col3:
        min_payments = st.number_input("Minimum Payments", min_value=0, max_value=76000, value=500, step=100)
        tenure = st.number_input("Tenure (months)", min_value=6, max_value=12, value=12, step=1)


    st.write("")
    if st.button("Predict Segment", type="primary"):
        input_data = np.array([[balance, balance_freq, purchases, oneoff, installments,
                                 cash_advance, purchases_freq, cash_advance_freq,
                                 cash_advance_trx, purchases_trx, credit_limit,
                                 payments, min_payments, prc_full_payment, tenure]])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]
        st.metric("Predicted Segment", cluster_names.get(cluster, "Unknown"), f"Cluster {cluster}")





st.divider()
st.caption("Models: K-Means clustering, Decision Tree / Naive Bayes classification (scikit-learn)")