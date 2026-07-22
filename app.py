import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

from crypto_layer import run_dhe_handshake, encrypt_mock_fi_payload, decrypt_fi_payload
from bsa import generate_synthetic_statement, analyze_cash_flow
from cmab import EpsilonGreedyCMAB
import torch, json
from torch_models import ESMM, RMTNet
from explain import shap_factors, adverse_action_notice
from counterfactual import coach, THRESHOLD
from drift import drift_report, anomaly_flags

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
    spec = joblib.load("models/feature_spec.pkl")
    scalers = joblib.load("models/scalers.pkl")
    esmm = ESMM(len(spec["intent_features"]), hidden=128)
    esmm.load_state_dict(torch.load("models/esmm.pt")); esmm.eval()
    rmt = RMTNet(len(spec["risk_features"]), hidden=128)
    rmt.load_state_dict(torch.load("models/rmt.pt")); rmt.eval()
    return intent, risk, spec, scalers, esmm, rmt

intent_bundle, risk_bundle, spec, scalers, esmm_model, rmt_model = load_models()

def torch_score(row_df, feats, mu, sd, model, kind):
    x = torch.tensor(((row_df[feats].astype(float) - mu) / sd).values, dtype=torch.float32)
    with torch.no_grad():
        if kind == "intent":
            _, pcvr, pctcvr = model(x)
            return float(pcvr), float(pctcvr)
        p_rej, p_def = model(x)
        return float(p_rej), float(p_def)

if "drift_rows" not in st.session_state:
    st.session_state.drift_rows = []

# ---------------- Header ----------------
st.title("🏦 C-PRISM — Contextual Predictive Repayment & Intent Scoring Model")
st.caption("Prospect Assist AI Bank · Live working prototype (IDBI Innovate 2026)")

with st.expander("⚠️ What's real vs. simulated in this demo", expanded=False):
    st.markdown("""
| Component | Status |
|---|---|
| Diffie-Hellman (X25519) key exchange | ✅ **Real** cryptographic handshake, runs live |
| Bank Statement Analyzer (DTI, income, obligations) | ✅ **Real** pandas computation on transaction data |
| Intent engine | ✅ **Real PyTorch ESMM** (entire-space multi-task) + naive GBM baseline for comparison |
| Risk engine | ✅ **Real PyTorch RMT-Net** (reject-aware masked-loss + monotonic gating) + naive GBM baseline |
| Explainability | ✅ **Real SHAP** factor attribution + LLM adverse-action notice (template fallback) |
| Counterfactual coach | ✅ **Real** grid search over the risk model for minimal decision-flipping changes |
| Drift monitor | ✅ **Real** KS-test against training distribution, live per session |
| CMAB routing | ✅ **Real** epsilon-greedy bandit, learns online as you use the demo |
| Account Aggregator data source | 🟡 **Simulated** — synthetic bank statements stand in for a live Setu/Finvu AA sandbox connection (requires FIU certification, out of scope for a 2-hour build) |
| AI Voice Agent | 🟡 Not implemented in this demo — described in architecture only |
""")

tab_live, tab_bias, tab_drift = st.tabs(["🏦 Live Pipeline", "🧪 Bias Experiment (why our engine wins)", "📡 Drift & Fraud Monitor"])

with tab_live:
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
        intent_row["dti_ratio"] = metrics["dti_ratio"]
        intent_row["income_stability_score"] = metrics["income_stability_score"]
        intent_row["device_tier_high"] = 1 if device_tier == "high" else 0
        intent_row = intent_row.reindex(columns=intent_bundle["features"], fill_value=0)
        gbm_intent = float(intent_bundle["model"].predict_proba(intent_row)[0, 1])
        _, esmm_pctcvr = torch_score(intent_row, spec["intent_features"],
                                     scalers["mu_i"], scalers["sd_i"], esmm_model, "intent")
        intent_probability = esmm_pctcvr  # bias-corrected engine drives decisions

        risk_row = pd.DataFrame([{
            "dti_ratio": metrics["dti_ratio"],
            "bounce_count_6m": metrics["bounce_count_6m"],
            "income_stability_score": metrics["income_stability_score"],
            "avg_monthly_balance": metrics["avg_monthly_balance"],
            "monthly_income": metrics["verified_monthly_income"],
            "total_obligations": metrics["total_monthly_obligations"],
        }])
        risk_row = risk_row.reindex(columns=risk_bundle["features"], fill_value=0)
        gbm_risk = float(risk_bundle["model"].predict_proba(risk_row)[0, 1])
        _, rmt_p_def = torch_score(risk_row, spec["risk_features"],
                                   scalers["mu_r"], scalers["sd_r"], rmt_model, "risk")
        default_probability = rmt_p_def  # bias-corrected engine drives decisions
        st.caption(f"Dual-engine scores — Intent: GBM baseline {gbm_intent:.2f} vs **ESMM {esmm_pctcvr:.2f}** · "
                   f"Risk: GBM baseline {gbm_risk:.2f} vs **RMT-Net {rmt_p_def:.2f}** (bias-corrected engine used for routing)")
        st.session_state.drift_rows.append({"dti_ratio": metrics["dti_ratio"],
            "income_stability_score": metrics["income_stability_score"],
            "bounce_count_6m": metrics["bounce_count_6m"],
            "monthly_income": metrics["verified_monthly_income"]})

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

        # ---- Explainable underwriting (SHAP + adverse action notice) ----
        st.subheader("🔍 Explainable Decision (SHAP + RBI-style notice)")
        factors = shap_factors(risk_row[risk_bundle["features"]])
        fdf = pd.DataFrame([{"Factor": f, "SHAP impact": round(s, 3), "Value": round(v, 3)}
                            for f, s, v in factors])
        st.dataframe(fdf, use_container_width=True, hide_index=True)
        decision_label = ("APPROVED" if default_probability < THRESHOLD
                          else "REFERRED FOR REVIEW")
        notice, source = adverse_action_notice(decision_label, factors, default_probability)
        st.text_area(f"Adverse action / decision notice ({'LLM-generated' if source=='llm' else 'template — set OPENAI_API_KEY in .env for LLM'})",
                     notice, height=170)

        # ---- Fraud / anomaly flags from BSA ----
        flags = anomaly_flags(metrics)
        if flags:
            for fl in flags:
                st.warning(f"⚠️ {fl}")

        # ---- Counterfactual loan-readiness coach ----
        if default_probability >= THRESHOLD:
            st.subheader("🧭 Loan-Readiness Coach (counterfactuals)")
            st.markdown("Instead of a dead-end rejection, C-PRISM computes the **smallest realistic change** that would make this applicant approvable — converting today's rejection into tomorrow's customer.")
            plans = coach(risk_row.iloc[0].to_dict(), default_probability)
            if plans:
                for p in plans:
                    st.success(f"**{p['plan']}** → risk drops {p['old_risk']:.0%} → {p['new_risk']:.0%} (below {THRESHOLD:.0%} approval bar)")
            else:
                st.caption("No single-lever change flips this decision — multi-factor improvement needed.")

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


with tab_bias:
    st.subheader("Controlled experiment: naive production models vs bias-corrected engine")
    st.markdown("""Because this dataset is **synthetic**, we hold the oracle real banks never have:
the true default outcome of applicants the credit policy **rejected**. That lets us measure exactly
how much Sample Selection Bias (intent) and MNAR reject bias (risk) cost — and how much the
PyTorch **ESMM** and **RMT-Net** engines recover. All models below are really trained; numbers come
from `train_torch.py`, not hardcoded.""")
    res = json.load(open("models/experiment_results.json"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk: default prediction on the FULL population**")
        rdf = pd.DataFrame({"Model": ["Naive GBM (approved-only)", "RMT-Net (reject-aware)"],
            "Full-population AUC": [res["risk"]["naive_gbm_full_pop_auc"], res["risk"]["rmtnet_full_pop_auc"]],
            "Rejected-only AUC": [res["risk"]["naive_gbm_rejected_only_auc"], res["risk"]["rmtnet_rejected_only_auc"]]})
        st.dataframe(rdf.round(3), use_container_width=True, hide_index=True)
        fig = px.bar(rdf, x="Model", y="Full-population AUC", title="Reject-aware learning recovers ranking power")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**The dangerous failure: risk underestimation on rejected applicants**")
        cal = res["risk_calibration_on_rejected"]
        cdf = pd.DataFrame({"Source": ["Actual default rate (oracle)", "Naive GBM predicts", "RMT-Net predicts"],
            "Default rate": [cal["actual_default_rate"], cal["naive_gbm_predicted_rate"], cal["rmtnet_predicted_rate"]]})
        fig2 = px.bar(cdf, x="Source", y="Default rate", title="Naive model says 7% — reality is 29%")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""The naive model, having never seen a rejected applicant's outcome, predicts a
**{cal['naive_gbm_predicted_rate']:.0%}** default rate for them — reality is **{cal['actual_default_rate']:.0%}**.
Deploy that model to expand lending (the NTC market) and it silently approves the riskiest cohort.
RMT-Net instead flags them as high-risk ({cal['rmtnet_predicted_rate']:.0%}) — deliberately conservative
where it has no data, which is exactly the behaviour a prudent underwriter wants.""")
    st.divider()
    st.markdown("**Intent: conversion prediction over ALL impressions (the space the model is used on)**")
    idf = pd.DataFrame({"Model": ["Naive GBM CVR (trained on clicked-only)", "ESMM (entire-space, chain-rule)"],
        "Entire-space AUC": [res["intent"]["naive_gbm_entire_space_auc"], res["intent"]["esmm_entire_space_auc"]]})
    st.dataframe(idf.round(3), use_container_width=True, hide_index=True)
    st.caption("ESMM never trains pCVR directly — it trains pCTR and pCTCVR (both observable for every impression) and derives pCVR, eliminating Sample Selection Bias. The CTnoCVR auxiliary head additionally penalizes clickbait leads.")

with tab_drift:
    st.subheader("Live feature drift vs. training distribution (KS test)")
    st.markdown("Every applicant scored in this session is compared against the training distribution. Drift signals **economic regime change** or **fraud rings probing the model** — both require model review before automated approvals continue.")
    rep = drift_report(st.session_state.drift_rows)
    if rep is None:
        st.info(f"Score at least 3 applicants in the Live Pipeline tab to activate the monitor ({len(st.session_state.drift_rows)} so far).")
    else:
        st.dataframe(rep, use_container_width=True, hide_index=True)
        if (rep["status"] == "🚨 DRIFT").any():
            st.error("Drift detected — in production this would page the MRM (model risk management) team and tighten the approval threshold automatically.")
