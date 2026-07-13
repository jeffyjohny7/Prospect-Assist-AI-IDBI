# C-PRISM — Working Prototype
### Contextual Predictive Repayment & Intent Scoring Model
IDBI Innovate 2026 · Prospect Assist AI Bank Challenge

## Run it
```bash
pip install streamlit pandas numpy scikit-learn cryptography joblib plotly
python3 train_models.py      # trains the real ML models (~5 sec, already done, re-run if you want)
streamlit run app.py
```

## What's real vs. simulated (be upfront about this with judges)

| Layer | Status |
|---|---|
| X25519 Diffie-Hellman key exchange | **Real** — runs live, genuine cryptographic handshake via the `cryptography` library |
| Bank Statement Analyzer (income, DTI, obligations, bounce detection) | **Real** — pandas computation over transaction-level data, nothing hardcoded |
| Intent model (stand-in for ESMM) | **Real** — GradientBoostingClassifier trained on 6,000 synthetic applicants, Test AUC ≈ 0.75 |
| Risk model (stand-in for RMT-Net) | **Real** — GradientBoostingClassifier, Test AUC ≈ 0.69 |
| CMAB routing (epsilon-greedy bandit) | **Real** — learns online from simulated rewards as you run the demo |
| Account Aggregator connection | **Simulated** — synthetic bank statements stand in for a live Setu/Finvu/OneMoney sandbox. Live AA integration requires FIU certification through Sahamati, which is a multi-day onboarding process, out of scope for this build window. |
| AI Voice Agent | Not implemented — described in the architecture doc only |



## Files
- `app.py` — Streamlit UI, orchestrates the full pipeline
- `crypto_layer.py` — real X25519 DHE handshake + XOR demo encryption using the derived key
- `bsa.py` — real pandas-based bank statement analysis
- `train_models.py` — generates synthetic applicant population + trains real sklearn models
- `cmab.py` — real epsilon-greedy contextual bandit
- `models/` — saved trained models (.pkl)
- `data/synthetic_applicants.csv` — the training data used

## Production roadmap (from the full whitepaper)
Live AA integration via Setu/Sahamati → RMT-Net / ESMM multi-task networks trained on
real approved/rejected loan history → AWS Lambda + API Gateway + KMS deployment →
AI Voice Agent for lead qualification.
