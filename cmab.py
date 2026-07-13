"""
Contextual Multi-Armed Bandit (CMAB) router - REAL epsilon-greedy implementation
with softmax-normalized action values, as described in the C-PRISM architecture.
Arms = the possible UI/routing actions. Reward is simulated for demo purposes
(1 = conversion, 0 = abandonment) and the bandit updates its value estimates
online, live in the Streamlit session, so you can watch it learn.
"""
import numpy as np

ARMS = [
    "STRAIGHT_THROUGH_APPROVAL",
    "AI_VOICE_ASSIST",
    "COLLATERALIZED_OFFER",
    "REQUEST_MORE_DOCS",
]


class EpsilonGreedyCMAB:
    def __init__(self, arms=ARMS, epsilon=0.15, seed=42):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = {a: 0 for a in arms}
        self.values = {a: 0.0 for a in arms}
        self.rng = np.random.default_rng(seed)

    def select_arm(self, context: dict) -> str:
        """context: dict with intent_probability, default_probability, dti_ratio etc.
        Uses context to bias the *rule-based prior* only when exploring is not
        chosen; exploitation uses the learned value estimates."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.arms)

        # Exploitation: softmax over learned values, nudged by live context
        intent = context.get("intent_probability", 0.5)
        risk = context.get("default_probability", 0.5)

        priors = {
            "STRAIGHT_THROUGH_APPROVAL": (intent > 0.7 and risk < 0.25),
            "AI_VOICE_ASSIST": (0.35 < intent <= 0.7),
            "COLLATERALIZED_OFFER": (risk >= 0.25 and intent > 0.5),
            "REQUEST_MORE_DOCS": (risk >= 0.4 or intent <= 0.35),
        }

        combined = {}
        for a in self.arms:
            base = self.values[a]
            nudge = 1.0 if priors.get(a) else 0.0
            combined[a] = base + nudge

        vals = np.array([combined[a] for a in self.arms])
        exp_vals = np.exp(vals - vals.max())
        probs = exp_vals / exp_vals.sum()
        return self.rng.choice(self.arms, p=probs)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n

    def snapshot(self):
        return {a: {"pulls": self.counts[a], "avg_reward": round(self.values[a], 3)} for a in self.arms}
