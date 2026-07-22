"""
C-PRISM data generation + BASELINE (production) model training.
v2 changes for bias-corrected engine:
  1. Labels now follow the funnel CHAIN: impression -> click -> convert
     (convert can only be 1 if click == 1)  -> enables ESMM comparison.
  2. A realistic credit POLICY rejects risky applicants; their default label
     is MASKED (-1, unobserved) -> creates true MNAR bias for RMT-Net to fix.
  3. Because data is synthetic, we keep `defaulted_true` (ground truth even
     for rejected applicants). This is ONLY used for evaluation of the bias
     experiment - never for training. It's the "oracle" real banks never have.
The GBMs trained here are the NAIVE PRODUCTION BASELINE:
  - intent GBM trains CVR on clicked-only samples (classic SSB)
  - risk GBM trains on approved-only samples (classic MNAR)
"""
import numpy as np, pandas as pd, joblib, os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
N = 8000

channel = np.random.choice(["whatsapp","app","web","sms"], N, p=[.35,.30,.25,.10])
time_on_page = np.random.gamma(2.0, 40, N).clip(5, 600)
device_tier = np.random.choice(["high","mid","low"], N, p=[.3,.45,.25])
loan_amount_requested = np.random.gamma(3, 40000, N).clip(10000, 2000000)
hour_of_day = np.random.randint(0, 24, N)
monthly_income = np.random.lognormal(10.5, 0.5, N).clip(8000, 500000)
existing_emi = monthly_income * np.random.beta(2, 5, N)
rent = monthly_income * np.random.beta(1.5, 6, N)
total_obligations = existing_emi + rent
dti_ratio = (total_obligations / monthly_income).clip(0, 3)
avg_monthly_balance = monthly_income * np.random.uniform(0.2, 2.0, N)
bounce_count_6m = np.random.poisson(0.6, N)
income_stability_score = np.random.beta(5, 2, N)

df = pd.DataFrame({"channel":channel,"time_on_page":time_on_page,"device_tier":device_tier,
    "loan_amount_requested":loan_amount_requested,"hour_of_day":hour_of_day,
    "monthly_income":monthly_income,"existing_emi":existing_emi,"rent":rent,
    "total_obligations":total_obligations,"dti_ratio":dti_ratio,
    "avg_monthly_balance":avg_monthly_balance,"bounce_count_6m":bounce_count_6m,
    "income_stability_score":income_stability_score})

# ---- CHAIN label generation: impression -> click -> convert ----
click_logit = (-0.8 + 1.4*(df.channel=="whatsapp") + 0.7*(df.channel=="app")
    + 0.012*df.time_on_page.clip(0,300) + 0.6*(df.device_tier=="high")
    - 0.4*(df.device_tier=="low") + np.random.normal(0,0.5,N))
clicked = np.random.binomial(1, 1/(1+np.exp(-click_logit)))

conv_logit = (-1.0 + 0.010*df.time_on_page.clip(0,300)
    - 0.0000018*df.loan_amount_requested - 1.2*df.dti_ratio
    + 1.0*df.income_stability_score + np.random.normal(0,0.5,N))
p_conv_given_click = 1/(1+np.exp(-conv_logit))
converted = clicked * np.random.binomial(1, p_conv_given_click)   # chain: conv only if click

# ---- Default ground truth (exists for EVERYONE - synthetic oracle) ----
risk_logit = (-2.2 + 2.2*df.dti_ratio + 0.5*df.bounce_count_6m
    - 1.8*df.income_stability_score - 0.0000015*df.avg_monthly_balance
    + 4.0*np.maximum(df.dti_ratio-0.55, 0)          # steep cliff naive model never sees
    + np.random.normal(0,0.5,N))
defaulted_true = np.random.binomial(1, 1/(1+np.exp(-risk_logit)))

# ---- Credit policy: reject risky -> their labels become UNOBSERVED (MNAR) ----
rejected = ((df.dti_ratio > 0.55) | (df.income_stability_score < 0.45)
            | (df.bounce_count_6m >= 3)).astype(int)
defaulted_observed = np.where(rejected==1, -1, defaulted_true)

df["clicked"], df["converted"] = clicked, converted
df["rejected"] = rejected
df["defaulted_true"] = defaulted_true          # oracle: evaluation only
df["defaulted_observed"] = defaulted_observed  # what a real bank sees
os.makedirs("data", exist_ok=True); os.makedirs("models", exist_ok=True)
df.to_csv("data/synthetic_applicants.csv", index=False)

df_enc = pd.get_dummies(df, columns=["channel","device_tier"], drop_first=False)
intent_features = ["time_on_page","loan_amount_requested","hour_of_day",
    "channel_app","channel_sms","channel_web","channel_whatsapp",
    "device_tier_high","device_tier_low","device_tier_mid","dti_ratio","income_stability_score"]
risk_features = ["dti_ratio","bounce_count_6m","income_stability_score",
    "avg_monthly_balance","monthly_income","total_obligations"]
joblib.dump({"intent_features":intent_features,"risk_features":risk_features}, "models/feature_spec.pkl")
df_enc.to_csv("data/encoded.csv", index=False)

# ---- NAIVE BASELINE 1: CVR GBM trained on CLICKED-ONLY (Sample Selection Bias) ----
clk = df_enc[df_enc.clicked==1]
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(clk[intent_features], clk.converted,
    test_size=0.2, random_state=42, stratify=clk.converted)
intent_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
intent_model.fit(Xi_tr, yi_tr)
print(f"[Baseline] CVR GBM (clicked-only) AUC on clicked test: {roc_auc_score(yi_te, intent_model.predict_proba(Xi_te)[:,1]):.3f}")

# ---- NAIVE BASELINE 2: risk GBM trained on APPROVED-ONLY (MNAR) ----
appr = df_enc[df_enc.rejected==0]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(appr[risk_features], appr.defaulted_observed,
    test_size=0.2, random_state=42, stratify=appr.defaulted_observed)
risk_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
risk_model.fit(Xr_tr, yr_tr)
print(f"[Baseline] Risk GBM (approved-only) AUC on approved test: {roc_auc_score(yr_te, risk_model.predict_proba(Xr_te)[:,1]):.3f}")

joblib.dump({"model":intent_model,"features":intent_features}, "models/intent_model.pkl")
joblib.dump({"model":risk_model,"features":risk_features}, "models/risk_model.pkl")
print(f"Dataset: {N} rows | rejected: {rejected.mean():.1%} | clicked: {clicked.mean():.1%} | converted: {converted.mean():.1%}")
