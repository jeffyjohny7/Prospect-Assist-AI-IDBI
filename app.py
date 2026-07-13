import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

from crypto_layer import run_dhe_handshake, encrypt_mock_fi_payload, decrypt_fi_payload
from bsa import generate_synthetic_statement, analyze_cash_flow
from cmab import EpsilonGreedyCMAB

st.set_page_config(page_title="C-PRISM | Prospect Assist AI Bank", layout="wide", page_icon="🏦")

# ---------------- Session state ----------------
if "cmab" not in st.session_state:
    st.session_state.cmab = EpsilonGreedyCMAB()
if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_models():
    intent = joblib.load("models/intent_model.pkl")
    risk = joblib.load("models/risk_model.pkl")
    return intent, risk

intent_bundle, risk_bundle = load_models()

# ---------------- Header ----------------
st.title("🏦 C-PRISM — Contextual Predictive Repayment & Intent Scoring Model")
st.caption("Prospect Assist AI Bank · Live working prototype (IDBI Innovate 2026)")

with st.expander("⚠️ What's real vs. simulated in this demo", expanded=False):
    st.markdown("""
| Component | Status |
|---|---|
| Diffie-Hellman (X25519) key exchange | ✅ **Real** cryptographic handshake, runs live |
| Bank Statement Analyzer (DTI, income, obligations) | ✅ **Real** pandas computation on transaction data |
| Intent model (ESMM stand-in) | ✅ **Real** trained GradientBoosting classifier (Test AUC ≈ 0.75) |
| Risk model (RMT-Net stand-in) | ✅ **Real** trained GradientBoosting classifier (Test AUC ≈ 0.69) |
| CMAB routing | ✅ **Real** epsilon-greedy bandit, learns online as you use the demo |
| Account Aggregator data source | 🟡 **Simulated** — synthetic bank statements stand in for a live Setu/Finvu AA sandbox connection (requires FIU certification, out of scope for a 2-hour build) |
| AI Voice Agent | 🟡 Not implemented in this demo — described in architecture only |
""")

st.divider()

# ---------------- Step 1: Applicant input ----------------
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("1️⃣ Applicant enters the funnel")
    channel = st.selectbox("Acquisition channel", ["whatsapp", "app", "web", "sms"])
    device_tier = st.selectbox("Device tier", ["high", "mid", "low"])
    time_on_page = st.slider("Time on landing page (seconds)", 5, 300, 60)
    loan_amount = st.number_input("Requested loan amount (₹)", 10000, 2000000, 300000, step=10000)
    monthly_income_declared = st.number_input("Approx. monthly income (₹, self-declared)", 8000, 500000, 45000, step=1000)
    hour_of_day = st.slider("Hour of application (24h)", 0, 23, 14)

with col2:
    st.subheader("2️⃣ Consent & Secure Data Fetch")
    st.markdown("Simulates the RBI Account Aggregator consent + FIU↔FIP key exchange.")
    run_fetch = st.button("🔐 Grant AA Consent & Fetch Bank Data", type="primary")

st.divider()

if run_fetch:
    # ---- Real crypto handshake ----
    st.subheader("🔑 Secure Data Acquisition Layer")
    handshake = run_dhe_handshake()
    with st.container(border=True):
        for s in handshake["steps"]:
            st.markdown(f"**{s['step']}**")
            st.code(s["detail"], language="text")
    st.success("Diffie-Hellman handshake complete — shared session key established.")

    # ---- Generate + "encrypt" + "decrypt" synthetic statement using real session key ----
    raw_transactions = generate_synthetic_statement(monthly_income_declared, months=6)
    encrypted_blob = encrypt_mock_fi_payload(handshake["session_key_hex"], {"transactions": raw_transactions})
    st.caption(f"Encrypted FI payload received from FIP (first 80 hex chars): `{encrypted_blob[:80]}...`")
    decrypted = decrypt_fi_payload(handshake["session_key_hex"], encrypted_blob)
    st.success("Payload decrypted with session key. 6 months of transactions received.")

    # ---- Real BSA ----
    st.subheader("📊 Cash-Flow Underwriting Layer (Bank Statement Analyzer)")
    metrics = analyze_cash_flow(decrypted["transactions"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verified Monthly Income", f"₹{metrics['verified_monthly_income']:,.0f}")
    m2.metric("Monthly Obligations", f"₹{metrics['total_monthly_obligations']:,.0f}")
    m3.metric("DTI Ratio", f"{metrics['dti_ratio']:.2f}")
    m4.metric("Income Stability", f"{metrics['income_stability_score']:.2f}")

    cat_df = pd.DataFrame(list(metrics["spend_by_category"].items()), columns=["Category", "Amount"])
    fig = px.bar(cat_df, x="Category", y="Amount", title="6-Month Spend by Category (from real transaction data)")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Real ML inference ----
    st.subheader("🧠 Algorithmic Decision Layer")

    intent_row = pd.DataFrame([{
        "time_on_page": time_on_page,
        "loan_amount_requested": loan_amount,
        "hour_of_day": hour_of_day,
        "channel_app": 1 if channel == "app" else 0,
        "channel_sms": 1 if channel == "sms" else 0,
        "channel_web": 1 if channel == "web" else 0,
        "channel_whatsapp": 1 if channel == "whatsapp" else 0,
        "device_tier_low": 1 if device_tier == "low" else 0,
        "device_tier_mid": 1 if device_tier == "mid" else 0,
    }])
    intent_row = intent_row.reindex(columns=intent_bundle["features"], fill_value=0)
    intent_probability = float(intent_bundle["model"].predict_proba(intent_row)[0, 1])

    risk_row = pd.DataFrame([{
        "dti_ratio": metrics["dti_ratio"],
        "bounce_count_6m": metrics["bounce_count_6m"],
        "income_stability_score": metrics["income_stability_score"],
        "avg_monthly_balance": metrics["avg_monthly_balance"],
        "monthly_income": metrics["verified_monthly_income"],
        "total_obligations": metrics["total_monthly_obligations"],
    }])
    risk_row = risk_row.reindex(columns=risk_bundle["features"], fill_value=0)
    default_probability = float(risk_bundle["model"].predict_proba(risk_row)[0, 1])

    c1, c2 = st.columns(2)
    with c1:
        fig1 = go.Figure(go.Indicator(mode="gauge+number", value=intent_probability * 100,
                                        title={"text": "Intent / Conversion Probability (%)"},
                                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2ca02c"}}))
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Indicator(mode="gauge+number", value=default_probability * 100,
                                        title={"text": "Default / Risk Probability (%)"},
                                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#d62728"}}))
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Real CMAB routing decision ----
    st.subheader("🎯 Dynamic CMAB Routing Decision")
    context = {"intent_probability": intent_probability, "default_probability": default_probability}
    chosen_arm = st.session_state.cmab.select_arm(context)

    action_labels = {
        "STRAIGHT_THROUGH_APPROVAL": "✅ Instant Pre-Approval — funds disbursed",
        "AI_VOICE_ASSIST": "📞 Route to AI Voice Agent for qualification",
        "COLLATERALIZED_OFFER": "🏠 Offer collateralized loan alternative",
        "REQUEST_MORE_DOCS": "📄 Request additional documentation",
    }
    st.info(f"**Routing decision:** {action_labels[chosen_arm]}")

    # simulate a reward for bandit learning based on the same generative logic
    simulated_reward = 1 if (intent_probability > 0.6 and default_probability < 0.3 and chosen_arm in
                              ["STRAIGHT_THROUGH_APPROVAL", "AI_VOICE_ASSIST"]) else np.random.binomial(1, 0.15)
    st.session_state.cmab.update(chosen_arm, simulated_reward)
    st.session_state.history.append({
        "channel": channel, "intent": round(intent_probability, 3),
        "risk": round(default_probability, 3), "dti": metrics["dti_ratio"],
        "action": chosen_arm, "reward": simulated_reward
    })

    st.divider()
    st.subheader("📈 CMAB Learning State (updates live as you run more applicants)")
    snap = st.session_state.cmab.snapshot()
    snap_df = pd.DataFrame(snap).T.reset_index().rename(columns={"index": "arm"})
    st.dataframe(snap_df, use_container_width=True, hide_index=True)

    if st.session_state.history:
        st.subheader("🕓 Session History")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

else:
    st.info("Fill in applicant details and click **Grant AA Consent & Fetch Bank Data** to run the full pipeline.")
