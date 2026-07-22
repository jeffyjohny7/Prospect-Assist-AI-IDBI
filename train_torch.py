"""
Trains the bias-corrected PyTorch engine AND runs the controlled experiment
that real banks can never run: because our data is synthetic, we know the TRUE
default outcome even for REJECTED applicants (`defaulted_true`). We evaluate:
  Naive GBM (approved-only)  vs  RMT-Net (reject-aware)   on the FULL population
  Naive GBM CVR (clicked-only) vs ESMM (entire-space)     on ALL impressions
Results -> models/experiment_results.json (consumed by the Streamlit Bias tab).
"""
import torch, joblib, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from torch_models import ESMM, RMTNet, train_esmm, train_rmtnet

df = pd.read_csv("data/encoded.csv")
spec = joblib.load("models/feature_spec.pkl")
IF, RF = spec["intent_features"], spec["risk_features"]

def std(X):  # simple standardizer, returns tensor + params for serving
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return torch.tensor(((X - mu) / sd).values, dtype=torch.float32), mu, sd

Xi, mu_i, sd_i = std(df[IF].astype(float))
Xr, mu_r, sd_r = std(df[RF].astype(float))
click = torch.tensor(df.clicked.values, dtype=torch.float32)
conv = torch.tensor(df.converted.values, dtype=torch.float32)
rej = torch.tensor(df.rejected.values, dtype=torch.float32)
dobs = torch.tensor(df.defaulted_observed.values, dtype=torch.float32)

print("Training ESMM (entire-space intent)...")
esmm = train_esmm(ESMM(Xi.shape[1], hidden=128), Xi, click, conv, epochs=600)
print("Training RMT-Net (reject-aware risk)...")
rmt = train_rmtnet(RMTNet(Xr.shape[1], hidden=128), Xr, rej, dobs, epochs=800)

torch.save(esmm.state_dict(), "models/esmm.pt")
torch.save(rmt.state_dict(), "models/rmt.pt")
joblib.dump({"mu_i": mu_i, "sd_i": sd_i, "mu_r": mu_r, "sd_r": sd_r}, "models/scalers.pkl")

# ================= THE EXPERIMENT =================
intent_gbm = joblib.load("models/intent_model.pkl")["model"]
risk_gbm = joblib.load("models/risk_model.pkl")["model"]

with torch.no_grad():
    _, pcvr, _ = esmm(Xi)
    _, p_def = rmt(Xr)
pcvr, p_def = pcvr.squeeze().numpy(), p_def.squeeze().numpy()
gbm_cvr = intent_gbm.predict_proba(df[IF])[:, 1]
gbm_def = risk_gbm.predict_proba(df[RF])[:, 1]

y_true_def = df.defaulted_true.values
rejmask = df.rejected.values == 1
res = {
    "risk": {
        "naive_gbm_full_pop_auc": roc_auc_score(y_true_def, gbm_def),
        "rmtnet_full_pop_auc": roc_auc_score(y_true_def, p_def),
        "naive_gbm_rejected_only_auc": roc_auc_score(y_true_def[rejmask], gbm_def[rejmask]),
        "rmtnet_rejected_only_auc": roc_auc_score(y_true_def[rejmask], p_def[rejmask]),
        "note": "Evaluated against defaulted_true - the oracle only a synthetic dataset provides.",
    },
    "risk_calibration_on_rejected": {
        "actual_default_rate": float(y_true_def[rejmask].mean()),
        "naive_gbm_predicted_rate": float(gbm_def[rejmask].mean()),
        "rmtnet_predicted_rate": float(p_def[rejmask].mean()),
        "note": "Naive model, trained only on approved loans, cannot see how risky rejected applicants truly are.",
    },
    "intent": {
        "naive_gbm_entire_space_auc": roc_auc_score(df.converted, gbm_cvr),
        "esmm_entire_space_auc": roc_auc_score(df.converted, (pcvr_ctr := esmm(Xi))[2].detach().squeeze().numpy()),
        "note": "pCTCVR evaluated over ALL impressions (the space the model is actually used on).",
    },
}
json.dump(res, open("models/experiment_results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
