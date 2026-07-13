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

## How it works — pipeline walkthrough

The app runs the applicant through four layers in sequence, each visible live in the UI:

**1. Applicant Funnel**
Captures acquisition channel, device tier, time on landing page, requested loan amount, self-declared income, and hour of application — the behavioral signals used by the intent model.

**2. Secure Data Acquisition Layer**
A live X25519 Diffie-Hellman key exchange runs between simulated FIU (the lender) and FIP (the bank) roles — each side generates an ephemeral keypair, both independently derive a matching shared secret, and a symmetric session key is derived via HKDF-SHA256. The (simulated) bank statement payload is then encrypted with that session key and decrypted on arrival — mirroring how a real RBI Account Aggregator exchange protects data in transit.

**3. Cash-Flow Underwriting Layer (Bank Statement Analyzer)**
Once decrypted, six months of transaction data are analyzed with pandas to compute verified monthly income, monthly obligations, DTI (debt-to-income) ratio, and an income stability score — plus a spend breakdown by category (salary, rent, EMI, discretionary, SIP). This replaces the applicant's self-declared numbers from Step 1 with verified figures.

**4. Algorithmic Decision Layer**
The verified financial features feed the risk model, while the funnel/behavioral features feed the intent model — both real, trained `GradientBoostingClassifier`s — producing live Intent/Conversion and Default/Risk probability gauges.

**5. Dynamic CMAB Routing Decision**
An epsilon-greedy contextual multi-armed bandit takes the intent score, risk score, and DTI ratio as context, and picks a routing action — e.g., straight-through approval, collateralized loan offer, request more documents, or AI voice assist. The "CMAB Learning State" table shows how many times each arm has been pulled and its average reward, updating live as more applicants run through the demo — this is genuine online learning, not a lookup table. The "Session History" log records every applicant processed in the session for transparency.

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
