# C-PRISM — Working Prototype

### Contextual Predictive Repayment & Intent Scoring Model

**IDBI Innovate 2026** · Prospect Assist AI Bank Challenge

**Team:** Jeffy / FinMIND  
**Repository:** [github.com/jeffyjohny7/Prospect-Assist-AI-IDBI](https://github.com/jeffyjohny7/Prospect-Assist-AI-IDBI)

---

## ── Quickstart Guide for Judges

To get the interactive prototype running locally in under a minute, execute the following commands in your terminal:

```bash
# 1. Install required dependencies
pip install -r requirements.txt

# 2. Train the real machine learning estimators (~5 seconds)
python train_models.py

# 3. Launch the interactive Streamlit funnel portal
streamlit run app.py
```

This opens a local Streamlit dashboard in your web browser illustrating the complete borrower acquisition loop: behavioral intent scoring, simulated bank statement fetching, live cryptographic handshakes, cash-flow underwriting, and live-updating bandit routing.

---

## ── System Pipeline Architecture

The platform runs applicants through successive pipeline modules. You can view the comprehensive flow below:

<p align="center">
  <a href="Dynamic%20User%20Acquisition-2026-07-13-164520_2.jpg" target="_blank">
    <img src="Dynamic%20User%20Acquisition-2026-07-13-164520_2.jpg" alt="C-PRISM Pipeline Flow Diagram" width="100%" style="max-width:700px; border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);"/>
  </a>
  <br/>
  <em>Click on the diagram to open and inspect it in high-resolution full-screen mode.</em>
</p>

---

## ── Engineering Rigor: What's Real vs. Simulated

We believe in complete transparency. Here is a granular breakdown of what runs natively inside our source code vs. what has been simulated for this prototype build window:

| Pipeline Layer | Implementation Status | Technical Stack |
| :--- | :--- | :--- |
| **X25519 Diffie-Hellman Handshake** | **100% Real & Live** | Dynamic cryptography via `cryptography.hazmat` library |
| **Bank Statement Analyzer (BSA)** | **100% Real & Live** | Pandas parsing of transaction-level ledger data (running DTI, regularity, and bounce detection) |
| **Intent Underwriting Classifier** | **100% Real & Live** | `GradientBoostingClassifier` trained on 6,000 synthetic applicants (Test AUC $\approx 0.75$) |
| **Risk Underwriting Classifier** | **100% Real & Live** | `GradientBoostingClassifier` trained on synthetic historical portfolio outcomes (Test AUC $\approx 0.69$) |
| **Dynamic Multi-Armed Bandit** | **100% Real & Live** | Online epsilon-greedy contextual reinforcement learning updating live via portal feedback loops |
| **Account Aggregator (AA) Fetch** | **Simulated Sandbox** | Synthetic statements act as the secure data payload. Production AA integration requires formal FIU/FIP sandbox certification via Sahamati |
| **AI Voice Telecalling Agent** | **Documented Spec Only** | Described in the production routing architecture; out of scope for this hackathon window |

---

## ── Step-by-Step Pipeline Walkthrough

The platform runs applicants through five successive modules, all of which are orchestrated and visible in real-time within the Streamlit UI:

### 1. Contextual Acquisition & Funnel
Captures user behavioral signals at the digital point-of-sale: acquisition channel (UTMs), device tier/metadata, time spent on the landing page, requested loan value, self-declared monthly salary, and temporal application parameters. These parameters form the core features evaluated by the conversion intent model.

### 2. Cryptographic Handshake (`crypto_layer.py`)
To emulate the data-in-transit security standards of the RBI Account Aggregator framework, we execute a live **X25519 Diffie-Hellman Ephemeral (DHE)** key exchange between the simulated Financial Information User (FIU - lender) and the Financial Information Provider (FIP - bank):

* The FIU generates ephemeral private key $d_{\text{FIU}}$ and public key $Q_{\text{FIU}}$.
* The FIP generates ephemeral private key $d_{\text{FIP}}$ and public key $Q_{\text{FIP}}$.
* Both parties independently compute the identical shared coordinate secret ($SS$) via:

$$
SS = \text{X25519}(d_{\text{FIU}}, Q_{\text{FIP}}) = \text{X25519}(d_{\text{FIP}}, Q_{\text{FIU}})
$$

* A symmetric session key $K$ is derived via a Hash-based Key Derivation Function:

$$
K = \text{HKDF-SHA256}(SS)
$$

* The dynamic bank statement JSON payload is symmetrically encrypted using $K$ during transport and cleanly decrypted inside the local sandbox execution layer.

### 3. Bank Statement Analyzer (`bsa.py`)
Once decrypted, six months of raw ledger data are analyzed in real-time using vector operations in `pandas`. It extracts:
* Verified Average Monthly Balance (AMB)
* Debt-to-Income (DTI) ratio
* Salary occurrence regularity and income stability scores
* Debit bounce detection indices and EMIs
* Categorized spending aggregates (Rent, EMIs, Discretionary, Savings)

### 4. Dual-Track Underwriting Engine (`train_models.py`)
Decoupled machine learning estimators assess the user across two different tracks:
1. **Risk Track (`risk_model.pkl`):** Assesses credit default probability ($P_{\text{default}}$) using verified financial features from the bank statements.
2. **Intent Track (`intent_model.pkl`):** Assesses application closure and conversion propensity ($P_{\text{conversion}}$) using behavioral metadata collected during acquisition.

### 5. Contextual Bandit Routing (`cmab.py`)
An online epsilon-greedy Contextual Multi-Armed Bandit (MAB) processes the multi-dimensional output vector (risk, intent, DTI) to select the optimal routing outcome. Arms include:
* **Instant Automated Disbursal** (for high intent, low risk)
* **Collateralized Loan Offer**
* **Request More Documents**
* **Interactive AI Voice Verification** (to resolve borderline statement discrepancies)

You can view the "MAB Learning State" table live in the app to see pull-counts, average rewards, and exploration parameters update on-the-fly. The "Session History" log records every applicant processed in the session for transparency.

---

## ── Repository Directory Map

```text
cprism/
│
├── app.py                      # Streamlit UI & End-to-End Pipeline Controller
├── cmab.py                     # Epsilon-Greedy Contextual Multi-Armed Bandit
├── crypto_layer.py             # X25519 Diffie-Hellman Ephemeral + HKDF-SHA256
├── bsa.py                      # Pandas-powered Bank Statement Analyzer (BSA)
├── train_models.py             # Population generation & Model training pipeline
│
├── data/
│   └── synthetic_applicants.csv # 6,000 baseline tabular applicant profiles
│
└── models/
    ├── risk_model.pkl          # Trained Risk-Estimator State
    └── intent_model.pkl        # Trained Intent-Estimator State
```

---

## ── Scope Boundary Note

Real-world Account Aggregator implementations require registering as a Financial Information User (FIU) and going through a multi-day onboarding process with licensed providers (like Setu, Finvu, or OneMoney) to obtain a live Sahamati production certificate. Because this requires institutional credentials, we have emulated the AA response payloads using highly structured synthetic historical statements. Every other cryptographic, analytic, and algorithmic model in this repository is 100% active and functional.

---

## ── Production Roadmap

1. **Production Account Aggregator Integration:** Bridge standard API contracts from the simulated sandbox to live Setu/Sahamati production endpoints.
2. **Deep Learning Sequence Modeling:** Upgrade the tabular classifiers to deep recurrent architectures (such as RMT-Net and ESMM multi-task models) trained directly on your institutional transaction-level ledger data.
3. **Serverless Cloud Infrastructure:** Deploy the modules onto AWS Lambda, API Gateway, and AWS Key Management Service (KMS) for high-concurrency capability.
4. **Speech-to-Text Dynamic Synthesis:** Hook up live VAPI or Twilio streams to the AI voice routing pipeline for borderline applicants.
