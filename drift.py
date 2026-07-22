"""
Live drift monitor: two-sample Kolmogorov-Smirnov test comparing the features
of applicants scored in this session against the training distribution.
In production this signals (a) economic regime change or (b) fraud rings
probing the model with crafted inputs -> both require model review.
"""
import pandas as pd
from scipy.stats import ks_2samp

MONITORED = ["dti_ratio", "income_stability_score", "bounce_count_6m", "monthly_income"]


def drift_report(session_rows: list, train_csv="data/synthetic_applicants.csv"):
    """session_rows: list of dicts of risk features from this session."""
    if len(session_rows) < 3:
        return None  # not enough live samples yet
    train = pd.read_csv(train_csv)
    live = pd.DataFrame(session_rows)
    out = []
    for f in MONITORED:
        if f not in live:
            continue
        stat, p = ks_2samp(train[f], live[f])
        out.append({"feature": f, "ks_stat": round(stat, 3), "p_value": round(p, 4),
                    "status": "🚨 DRIFT" if p < 0.05 else "✅ stable",
                    "train_mean": round(float(train[f].mean()), 3),
                    "live_mean": round(float(live[f].mean()), 3)})
    return pd.DataFrame(out)


def anomaly_flags(metrics: dict) -> list:
    """BSA-level fraud heuristics from the C-PRISM paper."""
    flags = []
    if metrics.get("income_stability_score", 1) < 0.3:
        flags.append("Erratic income pattern — possible amount infusion / deposit cycling")
    if metrics.get("dti_ratio", 0) > 1.0:
        flags.append("Obligations exceed verified income — undisclosed debt likely")
    if metrics.get("bounce_count_6m", 0) >= 3:
        flags.append("Repeated payment bounces — chronic liquidity stress")
    return flags
