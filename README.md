# C-PRISM — Prospect Assist AI (Team FinMIND · IDBI Innovate 2026)

**Contextual Predictive Repayment & Intent Scoring Model**

A dual-engine underwriting prototype for the IDBI Innovate 2026 · Prospect Assist AI Bank Challenge. C-PRISM runs a naive **production baseline** (GradientBoosting) side-by-side with a **bias-corrected PyTorch engine** (ESMM + RMT-Net), wrapped in a live borrower acquisition loop with cryptographic consent, cash-flow underwriting, explainability, counterfactual coaching, drift monitoring, and adaptive routing.

Repository: `github.com/jeffyjohny7/Prospect-Assist-AI-IDBI`

---

## Quick start

Clone the repo, then from the project root:

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic funnel data + train the GBM baselines (~5s)
python train_models.py

# 4. Train the bias-corrected PyTorch engines (ESMM + RMT-Net) and
#    run the controlled bias experiment
python train_torch.py

# 5. Launch the interactive Streamlit portal
streamlit run app.py
```

This opens a local dashboard illustrating the complete borrower acquisition loop: behavioral intent scoring, simulated bank statement fetching, live cryptographic handshakes, cash-flow underwriting, dual-engine risk assessment, SHAP explainability, counterfactual coaching, drift monitoring, and live-updating bandit routing.

Optional: drop an `OPENAI_API_KEY` into a `.env` file to enable LLM-generated adverse action notices. Without it, the system falls back to a deterministic RBI-style template — nothing breaks.

---

## What makes this different

### 1. The controlled bias experiment

Every lending model in production faces two silent biases that quietly poison decisions:

- **Sample Selection Bias (SSB):** CVR / intent models train only on clicked users but score everyone.
- **Missing-Not-At-Random (MNAR) rejects:** Risk models train on approved loans, but the riskiest applicants were rejected — so their true default outcomes are never observed.

Because our data is synthetic, we retain the oracle label (`defaulted_true`) that real banks never have, and measure the damage directly:

| Model | Metric | Score |
|---|---|---|
| Naive GBM on rejected applicants | Predicted default rate | 7% |
| Ground truth on rejected applicants | Actual default rate | **29%** |
| Naive GBM | Full-population AUC | 0.59 |
| **RMT-Net** (masked default loss + monotonic reject-aware gating) | Full-population AUC | **0.73** |
| Naive GBM | Entire-space intent AUC | 0.73 |
| **ESMM** (pCTCVR = pCTR × pCVR + CTnoCVR anti-clickbait head) | Entire-space intent AUC | **0.82** |

Deploying the naive baseline to expand New-to-Credit lending silently approves the riskiest cohort. The bias-corrected engines recover honest signal on the full population.

### 2. Explainable decisions

SHAP factor attribution on every score, plus an RBI-style plain-language adverse action notice generated per applicant (LLM-drafted via OpenAI when a key is present, deterministic template otherwise). Auditable AI is a regulatory requirement, not a feature.

### 3. Counterfactual loan-readiness coach

Declined applicants receive the **smallest realistic change** that flips the decision — e.g., *"reduce monthly obligations by ₹X → approvable"*. Rejections convert into future pipeline: retention and financial inclusion in one move.

### 4. Live drift & fraud monitor

KS-test of session applicants against the training distribution, plus BSA-level anomaly heuristics (deposit cycling, hidden debt patterns, salary irregularity).

### 5. Adaptive routing via contextual bandit

An online ε-greedy Contextual Multi-Armed Bandit consumes the (risk, intent, DTI) vector and learns live from portal feedback, updating pull-counts and average rewards on-the-fly.

---

## Architecture


```
AA consent (real X25519 DHE handshake) → BSA (pandas cash-flow features)
  → dual engines: GBM baseline ∥ ESMM + RMT-Net (PyTorch, drives routing)
  → SHAP explainability + adverse action notice
  → CMAB router (ε-greedy, learns live) → drift & fraud monitor
```


### Step-by-step pipeline

**1. Contextual acquisition & funnel.** Captures behavioral signals at the digital point-of-sale: acquisition channel (UTMs), device tier/metadata, dwell time, requested loan value, self-declared salary, and temporal application parameters. These features feed the intent model.

**2. Cryptographic handshake (`crypto_layer.py`).** Emulates the RBI Account Aggregator data-in-transit standard with a live X25519 Diffie-Hellman Ephemeral (DHE) key exchange between the Financial Information User (lender) and Financial Information Provider (bank):

- FIU generates ephemeral keypair $(d_{\text{FIU}}, Q_{\text{FIU}})$; FIP generates $(d_{\text{FIP}}, Q_{\text{FIP}})$.
- Both independently compute the shared secret $SS = \text{X25519}(d_{\text{FIU}}, Q_{\text{FIP}}) = \text{X25519}(d_{\text{FIP}}, Q_{\text{FIU}})$.
- A symmetric session key is derived: $K = \text{HKDF-SHA256}(SS)$.
- The bank statement JSON payload is symmetrically encrypted with $K$ in transit and decrypted inside the local sandbox.

**3. Bank Statement Analyzer (`bsa.py`).** Six months of decrypted ledger data are analyzed via vectorized pandas operations to extract: Average Monthly Balance, Debt-to-Income ratio, salary regularity and income stability scores, debit bounce indices, EMI obligations, and categorized spending aggregates (rent, EMIs, discretionary, savings).

**4. Dual-track underwriting engine.** Two parallel estimators score every applicant:

- **Risk track** — probability of default $P_{\text{default}}$, from verified financial features. Naive GBM baseline (`train_models.py`) runs alongside the bias-corrected RMT-Net (`torch_models.py`, `train_torch.py`), which uses a masked default loss and a monotonic reject-aware gating network to recover signal on the unobserved rejected population.
- **Intent track** — probability of application closure and conversion $P_{\text{conversion}}$, from behavioral metadata. Naive GBM baseline runs alongside ESMM, which enforces the entire-space chain rule $\text{pCTCVR} = \text{pCTR} \times \text{pCVR}$ and adds a CTnoCVR anti-clickbait head to correct sample selection bias.

**5. Explainability & counterfactual coaching (`explain.py`, `counterfactual.py`).** SHAP attributions decompose each score into per-feature contributions; declined applicants receive an adverse action notice and the minimum feasible feature change that would flip the decision.

**6. Contextual bandit routing (`cmab.py`).** An ε-greedy Contextual MAB selects the optimal action from:

- Instant automated disbursal (high intent, low risk)
- Collateralized loan offer
- Request more documents
- Interactive AI voice verification (borderline statement discrepancies)

The live "MAB Learning State" table in the portal shows pull-counts, average rewards, and exploration parameters updating in real time.

**7. Drift & fraud monitor (`drift.py`).** KS-test compares session applicants to the training distribution and flags BSA-level anomalies indicating potential fraud (deposit cycling, hidden debt patterns).

---

## What's real vs. simulated

Full transparency on the implementation status of every layer:

| Pipeline layer | Status | Stack |
|---|---|---|
| X25519 Diffie-Hellman handshake | **100% real & live** | `cryptography.hazmat` |
| Bank Statement Analyzer | **100% real & live** | pandas vectorized parsing |
| Naive GBM baselines (risk + intent) | **100% real & live** | scikit-learn `GradientBoostingClassifier` on 6,000 synthetic applicants |
| ESMM intent engine | **100% real & live** | PyTorch multi-task network with pCTCVR chain rule |
| RMT-Net risk engine | **100% real & live** | PyTorch with masked default loss + monotonic reject-aware gating |
| SHAP explainability | **100% real & live** | `shap` library on both engines |
| Counterfactual coach | **100% real & live** | Minimal decision-flipping change search |
| Drift & fraud monitor | **100% real & live** | KS-test + BSA anomaly heuristics |
| Contextual Multi-Armed Bandit | **100% real & live** | Online ε-greedy, learns from portal feedback |
| Adverse action notice | **100% real & live** | OpenAI LLM if API key present, deterministic template otherwise |
| Account Aggregator fetch | Simulated sandbox | Synthetic statements as secure payload; production requires Sahamati/FIU certification |
| AI voice telecalling agent | Documented spec | In production routing architecture; out of scope for this build window |

---

## Repository map


```
cprism/
│
├── app.py                       # Streamlit UI & end-to-end pipeline controller (3 tabs)
├── crypto_layer.py              # X25519 Diffie-Hellman Ephemeral + HKDF-SHA256
├── bsa.py                       # Pandas-powered Bank Statement Analyzer
├── train_models.py              # Synthetic funnel data (chain labels + reject mask)
│                                #   + naive GBM baselines
├── torch_models.py              # ESMM & RMT-Net architectures
├── train_torch.py               # Trains torch engines + runs the controlled bias experiment
├── explain.py                   # SHAP + LLM/template adverse action notice
├── counterfactual.py            # Minimal decision-flipping change search
├── drift.py                     # KS drift test + fraud heuristics
├── cmab.py                      # ε-greedy contextual bandit router
│
├── data/
│   └── synthetic_applicants.csv # 6,000 baseline tabular applicant profiles
│
└── models/
    ├── risk_model.pkl           # Trained naive GBM risk estimator
    ├── intent_model.pkl         # Trained naive GBM intent estimator
    ├── esmm.pt                  # Trained ESMM entire-space intent engine
    └── rmtnet.pt                # Trained RMT-Net reject-aware risk engine
```

---

## Scope boundary

Real-world Account Aggregator integration requires registration as a Financial Information User and multi-day onboarding with a licensed provider (Setu, Finvu, or OneMoney) to obtain a live Sahamati production certificate. Because this needs institutional credentials, AA response payloads are emulated using highly structured synthetic historical statements. Every cryptographic, analytic, and algorithmic component is fully active and functional — the boundary is purely at the AA network edge.

---

## Production roadmap

- **Live AA integration:** Bridge existing API contracts from the sandbox to production Setu / Sahamati endpoints.
- **Deeper sequence modeling:** Extend RMT-Net with transaction-level recurrent architectures trained directly on institutional ledger data.
- **Serverless deployment:** Ship modules onto AWS Lambda + API Gateway + AWS KMS for high-concurrency handling of session traffic.
- **Voice channel activation:** Wire the AI voice verification route to live VAPI or Twilio streams for real-time borderline applicant resolution.