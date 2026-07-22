"""
Counterfactual loan-readiness coach: for a borderline/declined applicant,
grid-search the SMALLEST realistic change that brings default risk under the
approval threshold. Turns a rejection into a concrete improvement plan
(retention + financial inclusion story).
"""
import numpy as np, pandas as pd, joblib

THRESHOLD = 0.30  # p_default below this -> approvable

# (feature, actionable direction, step, max_steps, human label formatter)
_ACTIONS = [
    ("dti_ratio", -0.05, 8, lambda base, new, income:
        f"Reduce monthly obligations by ₹{(base-new)*income:,.0f} "
        f"(DTI {base:.0%} → {new:.0%})"),
    ("bounce_count_6m", -1, 3, lambda base, new, income:
        f"Maintain {int(base-new)} fewer payment bounces over the next 6 months"),
    ("avg_monthly_balance", +0.15, 6, lambda base, new, income:
        f"Raise average balance from ₹{base:,.0f} to ₹{new:,.0f}"),
]


def coach(risk_row: dict, p_default_now: float):
    """Returns list of minimal single-lever plans that flip the decision."""
    if p_default_now < THRESHOLD:
        return []
    bundle = joblib.load("models/risk_model.pkl")
    model, feats = bundle["model"], bundle["features"]
    income = risk_row.get("monthly_income", 1)
    plans = []
    for feat, step, max_steps, fmt in _ACTIONS:
        base = risk_row[feat]
        for k in range(1, max_steps + 1):
            new_val = base + step * k * (base if feat == "avg_monthly_balance" else 1)
            if feat == "avg_monthly_balance":
                new_val = base * (1 + 0.15 * k)
            new_val = max(new_val, 0)
            row = dict(risk_row)
            row[feat] = new_val
            if feat == "dti_ratio":  # keep obligations consistent
                row["total_obligations"] = new_val * income
            p = float(model.predict_proba(pd.DataFrame([row])[feats])[0, 1])
            if p < THRESHOLD:
                plans.append({"plan": fmt(base, new_val, income),
                              "new_risk": p, "old_risk": p_default_now})
                break
    return sorted(plans, key=lambda x: x["new_risk"])
