"""
Explainable underwriting: SHAP on the risk model + LLM-generated adverse action
notice (RBI-style plain-language explanation). Uses OPENAI_API_KEY from .env if
present; falls back to a deterministic template so the demo never breaks.
"""
import os, shap, numpy as np, joblib
from dotenv import load_dotenv
load_dotenv()

_FRIENDLY = {
    "dti_ratio": "debt-to-income ratio",
    "bounce_count_6m": "payment bounces in the last 6 months",
    "income_stability_score": "income stability",
    "avg_monthly_balance": "average monthly balance",
    "monthly_income": "verified monthly income",
    "total_obligations": "existing monthly obligations",
}

_explainer = None
def get_explainer():
    global _explainer
    if _explainer is None:
        bundle = joblib.load("models/risk_model.pkl")
        _explainer = shap.TreeExplainer(bundle["model"])
    return _explainer


def shap_factors(risk_row_df, top_k=3):
    """Returns [(feature, shap_value, feature_value)] sorted by |impact|."""
    sv = get_explainer().shap_values(risk_row_df)[0]
    order = np.argsort(-np.abs(sv))[:top_k]
    cols = risk_row_df.columns
    return [(cols[i], float(sv[i]), float(risk_row_df.iloc[0, i])) for i in order]


def _template_notice(decision, factors, p_default):
    lines = [f"Decision: {decision} (estimated default risk {p_default:.0%}).",
             "Primary factors in this assessment:"]
    for feat, sv, val in factors:
        direction = "increased" if sv > 0 else "reduced"
        name = _FRIENDLY.get(feat, feat)
        if feat == "dti_ratio":
            val_str = f"{val:.0%} of income committed to obligations"
        elif feat in ("monthly_income", "avg_monthly_balance", "total_obligations"):
            val_str = f"₹{val:,.0f}"
        else:
            val_str = f"{val:.2f}"
        lines.append(f"- Your {name} ({val_str}) {direction} the assessed risk.")
    lines.append("This assessment used consented Account Aggregator data only. "
                 "You may request a review or re-apply after addressing the factors above.")
    return "\n".join(lines)


def adverse_action_notice(decision, factors, p_default):
    """LLM notice if OPENAI_API_KEY is set; template otherwise."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return _template_notice(decision, factors, p_default), "template"
    try:
        from openai import OpenAI
        client = OpenAI()
        facts = "; ".join(f"{_FRIENDLY.get(f, f)}={v:.3f} (SHAP impact {s:+.3f})"
                          for f, s, v in factors)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=250,
            messages=[{"role": "system", "content":
                "You write RBI-compliant adverse action notices for an Indian retail bank. "
                "Plain language, empathetic, factual, no legal jargon, under 120 words. "
                "Never promise approval. Amounts are INR."},
                {"role": "user", "content":
                f"Decision: {decision}. Estimated default probability: {p_default:.0%}. "
                f"Top model factors: {facts}. Write the notice to the applicant."}])
        return resp.choices[0].message.content, "llm"
    except Exception as e:
        return _template_notice(decision, factors, p_default) + f"\n\n(LLM unavailable: {e})", "template"
