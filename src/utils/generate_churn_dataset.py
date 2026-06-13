"""Génère un dataset synthétique de churn SaaS.

Features (réalistes pour un SaaS B2B) :
- tenure_days       : ancienneté du compte
- nb_logins_30j     : connexions le mois dernier
- nb_features_used  : nombre de features distinctes utilisées
- plan              : Free / Pro / Enterprise
- support_tickets_90j
- mrr_eur           : revenu mensuel
- has_integration   : Slack/Zapier/etc connecté

Target : churned (booléen, churné dans les 30 jours)
"""
import numpy as np
import pandas as pd

def generate(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["Free", "Pro", "Enterprise"], n, p=[0.6, 0.3, 0.1])
    tenure = rng.integers(1, 1500, n)
    logins = rng.poisson(lam=np.where(plan == "Free", 3, 15), size=n)
    features_used = rng.poisson(lam=np.where(plan == "Free", 2, 8), size=n)
    tickets = rng.poisson(lam=0.5, size=n)
    mrr = np.where(plan == "Free", 0,
          np.where(plan == "Pro", rng.normal(49, 10, n),
                                  rng.normal(499, 100, n)))
    has_integration = rng.random(n) < np.where(plan == "Free", 0.1, 0.6)

    logit = (
        -0.005 * tenure
        - 0.1   * logins
        - 0.15  * features_used
        + 0.2   * tickets
        - 0.001 * mrr
        - 0.5   * has_integration.astype(int)
        + 1.0
    )
    proba = 1 / (1 + np.exp(-logit))
    churned = rng.random(n) < proba

    return pd.DataFrame({
        "tenure_days": tenure,
        "nb_logins_30j": logins,
        "nb_features_used": features_used,
        "plan": plan,
        "support_tickets_90j": tickets,
        "mrr_eur": mrr.round(2),
        "has_integration": has_integration,
        "churned": churned,
    })

if __name__ == "__main__":
    df = generate()
    df.to_parquet("data/churn.parquet")
    print(f"Wrote {len(df)} rows to data/churn.parquet")
    print(f"Churn rate: {df['churned'].mean():.1%}")