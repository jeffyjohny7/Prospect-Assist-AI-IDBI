"""
C-PRISM: Synthetic data generation + REAL model training.
This is not mocked - we generate a plausible synthetic population of loan
applicants, derive labels using a noisy-but-principled generative process,
and train actual scikit-learn classifiers on them. This stands in for the
real historical approved/rejected loan book that a bank would use to train
RMT-Net / ESMM in production.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

np.random.seed(42)
N = 6000

# ---- Simulate applicant features (what CMAB/ESMM would see at top of funnel) ----
channel = np.random.choice(["whatsapp", "app", "web", "sms"], N, p=[0.35, 0.30, 0.25, 0.10])
time_on_page = np.random.gamma(2.0, 40, N).clip(5, 600)          # seconds
device_tier = np.random.choice(["high", "mid", "low"], N, p=[0.3, 0.45, 0.25])
loan_amount_requested = np.random.gamma(3, 40000, N).clip(10000, 2000000)
hour_of_day = np.random.randint(0, 24, N)

# ---- Simulate bank-statement-derived financial features (BSA output) ----
monthly_income = np.random.lognormal(mean=10.5, sigma=0.5, size=N).clip(8000, 500000)
existing_emi = monthly_income * np.random.beta(2, 5, N)
rent = monthly_income * np.random.beta(1.5, 6, N)
total_obligations = existing_emi + rent
dti_ratio = (total_obligations / monthly_income).clip(0, 3)
avg_monthly_balance = monthly_income * np.random.uniform(0.2, 2.0, N)
bounce_count_6m = np.random.poisson(0.6, N)
income_stability_score = np.random.beta(5, 2, N)  # 0-1, higher = more stable salary credits

df = pd.DataFrame({
    "channel": channel,
    "time_on_page": time_on_page,
    "device_tier": device_tier,
    "loan_amount_requested": loan_amount_requested,
    "hour_of_day": hour_of_day,
    "monthly_income": monthly_income,
    "existing_emi": existing_emi,
    "rent": rent,
    "total_obligations": total_obligations,
    "dti_ratio": dti_ratio,
    "avg_monthly_balance": avg_monthly_balance,
    "bounce_count_6m": bounce_count_6m,
    "income_stability_score": income_stability_score,
})

# ---- Generative process for labels (ground truth we then try to learn back) ----
# Intent / conversion probability: driven by engagement signals (ESMM-style CVR)
intent_logit = (
    -1.2
    + 1.8 * (df["channel"] == "whatsapp")
    + 0.9 * (df["channel"] == "app")
    + 0.015 * (df["time_on_page"].clip(0, 300))
    + 0.8 * (df["device_tier"] == "high")
    - 0.4 * (df["device_tier"] == "low")
    - 0.000002 * df["loan_amount_requested"]
    + np.random.normal(0, 0.6, N)
)
p_conversion = 1 / (1 + np.exp(-intent_logit))
converted = np.random.binomial(1, p_conversion)

# Default / risk probability: driven by cash-flow underwriting signals (RMT-Net-style)
risk_logit = (
    -2.0
    + 3.2 * df["dti_ratio"]
    + 0.55 * df["bounce_count_6m"]
    - 2.0 * df["income_stability_score"]
    - 0.0000015 * df["avg_monthly_balance"]
    + np.random.normal(0, 0.5, N)
)
p_default = 1 / (1 + np.exp(-risk_logit))
defaulted = np.random.binomial(1, p_default)

df["converted"] = converted
df["defaulted"] = defaulted

df.to_csv("data/synthetic_applicants.csv", index=False)

# ---- Feature encoding ----
df_enc = pd.get_dummies(df, columns=["channel", "device_tier"], drop_first=True)

intent_features = [c for c in df_enc.columns if c not in
                    ["converted", "defaulted", "monthly_income", "existing_emi", "rent",
                     "total_obligations", "dti_ratio", "avg_monthly_balance",
                     "bounce_count_6m", "income_stability_score"]]
risk_features = ["dti_ratio", "bounce_count_6m", "income_stability_score",
                  "avg_monthly_balance", "monthly_income", "total_obligations"]

X_intent = df_enc[intent_features]
y_intent = df_enc["converted"]
X_risk = df_enc[risk_features]
y_risk = df_enc["defaulted"]

# ---- Train REAL models (ESMM stand-in = intent model, RMT-Net stand-in = risk model) ----
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(X_intent, y_intent, test_size=0.2, random_state=42, stratify=y_intent)
intent_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
intent_model.fit(Xi_tr, yi_tr)
intent_auc = roc_auc_score(yi_te, intent_model.predict_proba(Xi_te)[:, 1])

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk)
risk_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
risk_model.fit(Xr_tr, yr_tr)
risk_auc = roc_auc_score(yr_te, risk_model.predict_proba(Xr_te)[:, 1])

os.makedirs("models", exist_ok=True)
joblib.dump({"model": intent_model, "features": intent_features}, "models/intent_model.pkl")
joblib.dump({"model": risk_model, "features": risk_features}, "models/risk_model.pkl")

print(f"Intent (ESMM stand-in) model trained. Test AUC: {intent_auc:.3f}")
print(f"Risk (RMT-Net stand-in) model trained. Test AUC: {risk_auc:.3f}")
print(f"Synthetic dataset: {N} rows saved to data/synthetic_applicants.csv")
