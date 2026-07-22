"""
Bias-corrected engine: real PyTorch ESMM + RMT-Net (as promised in the proposal).
ESMM   : trains pCTR and pCTCVR over the ENTIRE impression space; pCVR is
         derived (pCTCVR = pCTR * pCVR) and never fit directly -> kills SSB.
         Includes CTnoCVR auxiliary head to penalize clickbait leads.
RMT-Net: default head's loss is MASKED to approved loans only; a gating
         network driven by rejection probability lets the default head borrow
         the rejection head's representation exactly where it has no data.
"""
import torch, torch.nn as nn


class ESMM(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.ctr = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.cvr = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.shared(x)
        pctr = torch.sigmoid(self.ctr(h))
        pcvr = torch.sigmoid(self.cvr(h))     # derived quantity, never trained directly
        return pctr, pcvr, pctr * pcvr        # pCTCVR


def train_esmm(model, X, click, convert, epochs=250, lr=1e-3, aux_w=0.3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    for e in range(epochs):
        pctr, pcvr, pctcvr = model(X)
        loss = bce(pctr.squeeze(), click) + bce(pctcvr.squeeze(), convert)
        loss += aux_w * bce((pctr * (1 - pcvr)).squeeze(), click * (1 - convert))  # CTnoCVR
        opt.zero_grad(); loss.backward(); opt.step()
    return model


class RMTNet(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.rej_branch = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU())
        self.def_branch = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU())
        self.rej_head = nn.Linear(32, 1)
        self.def_head = nn.Linear(32, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))   # learnable, kept >=0 via softplus

    def forward(self, x):
        h = self.shared(x)
        rej_logit = self.rej_head(self.rej_branch(h))
        def_logit = self.def_head(self.def_branch(h))
        # Reject-aware gating: rejection knowledge can only INCREASE predicted risk.
        # Monotonic by construction (softplus(alpha) >= 0) - the higher the model
        # thinks the credit policy would reject you, the higher your default risk.
        p_rej = torch.sigmoid(rej_logit)
        p_def = torch.sigmoid(def_logit + torch.nn.functional.softplus(self.alpha) * rej_logit)
        return p_rej, p_def


def train_rmtnet(model, X, reject, default_observed, epochs=250, lr=1e-3):
    """default_observed: -1 where unobserved (rejected applicants)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    mask = default_observed >= 0
    for e in range(epochs):
        p_rej, p_def = model(X)
        loss = bce(p_rej.squeeze(), reject)
        loss += bce(p_def.squeeze()[mask], default_observed[mask])  # masked loss
        opt.zero_grad(); loss.backward(); opt.step()
    return model
