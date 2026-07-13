# C-PRISM — Working Prototype
### Contextual Predictive Repayment & Intent Scoring Model
IDBI Innovate 2026 · Prospect Assist AI Bank Challenge

**Team:** [Jeffy / FinMIND ]
**Repo:** https://github.com/jeffyjohny7/Prospect-Assist-AI-IDBI

---

## For judges — quickest way to see it run

```bash
pip install -r requirements.txt
python train_models.py      # trains the real ML models (~5 sec — already done, but you can re-run it)
streamlit run app.py
```

This opens a local Streamlit app in your browser walking through the full applicant funnel — behavioral intent scoring, simulated bank data fetch, and live risk/intent gauges.

---

## What's real vs. simulated (we're upfront about this)

| Layer | Status |
|---|---|
| X25519 Diffie-Hellman key exchange | **Real** — runs live, genuine cryptographic handshake via the `cryptography` library |
| Bank Statement Analyzer (income, DTI, obligations, bounce detection) | **Real** — pandas computation over transaction-level data, nothing hardcoded |
| Intent model (stand-in for ESMM) | **Real** — GradientBoostingClassifier trained on 6,000 synthetic applicants, Test AUC ≈ 0.75 |
| Risk model (stand-in for RMT-Net) | **Real** — GradientBoostingClassifier, Test AUC ≈ 0.69 |
| CMAB routing (epsilon-greedy bandit) | **Real** — learns online from simulated rewards as you run the demo |
| Account Aggregator connection | **Simulated** — synthetic bank statements stand in for a live Setu/Finvu/OneMoney sandbox. Live AA integration requires FIU certification through Sahamati, a multi-day onboarding process, out of scope for this build window. |
| AI Voice Agent | Not implemented — described in the architecture doc only |

## A note on scope

The Account Aggregator step uses simulated bank data instead of a live Setu/Finvu 
integration, since FIU certification through Sahamati takes several days and was 
out of scope for this build window. Every other layer — the cryptographic handshake, 
bank statement analysis, both trained ML models, and the bandit routing — runs for 
real, live, when you run the app.

## Files

- `app.py` — Streamlit UI, orchestrates the full pipeline
- `crypto_layer.py` — real X25519 DHE handshake + XOR demo encryption using the derived key
- `bsa.py` — real pandas-based bank statement analysis
- `train_models.py` — generates synthetic applicant population + trains real sklearn models
- `cmab.py` — real epsilon-greedy contextual bandit
- `models/` — saved trained models (`.pkl`)
- `data/synthetic_applicants.csv` — the training data used

## Production roadmap (from the full whitepaper)

Live AA integration via Setu/Sahamati → RMT-Net / ESMM multi-task networks trained on real approved/rejected loan history → AWS Lambda + API Gateway + KMS deployment → AI Voice Agent for lead qualification.
