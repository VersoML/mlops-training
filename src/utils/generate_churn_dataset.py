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
import argparse
import os

import numpy as np
import pandas as pd


def generate(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["Free", "Pro", "Enterprise"], n, p=[0.6, 0.3, 0.1])
    tenure = rng.integers(1, 1500, n)
    logins = rng.poisson(lam=np.where(plan == "Free", 3, 15), size=n)
    features_used = rng.poisson(lam=np.where(plan == "Free", 2, 8), size=n)

    # Segment d'insatisfaction indépendant du plan : un moteur de churn qui touche
    # aussi les clients payants, pour que "revenue_at_risk" soit un KPI utile et ne
    # se concentre pas uniquement sur le tier gratuit (mrr = 0).
    dissatisfied = rng.random(n) < 0.10
    tickets = rng.poisson(lam=np.where(dissatisfied, 5, 0.4), size=n)

    mrr = np.where(plan == "Free", 0,
          np.where(plan == "Pro", rng.normal(49, 10, n),
                                  rng.normal(499, 100, n)))
    has_integration = rng.random(n) < np.where(plan == "Free", 0.1, 0.6)

    logit = (
        -0.003 * tenure
        - 0.06  * logins
        - 0.08  * features_used
        + 0.55  * tickets
        - 0.0006 * mrr
        - 0.3   * has_integration.astype(int)
        - 0.3
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


def apply_drift(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Décalage de covariables (mêmes labels `churned`) pour le batch courant du module 4.

    Reproduit un changement de population : une partie des comptes Free passe en Pro,
    le MRR monte, l'engagement et les intégrations augmentent. Les features bougent,
    la cible non : c'est le cas d'usage canonique de detection de data drift.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    free = out["plan"].to_numpy() == "Free"
    upgraded = free & (rng.random(len(out)) < 0.6)
    out.loc[upgraded, "plan"] = "Pro"
    out.loc[upgraded, "mrr_eur"] = rng.normal(49, 10, int(upgraded.sum())).round(2).clip(0)

    paid = out["mrr_eur"].to_numpy() > 0
    out.loc[paid, "mrr_eur"] = (out.loc[paid, "mrr_eur"] * 1.3).round(2)

    out["nb_logins_30j"] = out["nb_logins_30j"] + rng.poisson(0.6, len(out))

    add_integration = (~out["has_integration"].to_numpy()) & (rng.random(len(out)) < 0.15)
    out.loc[add_integration, "has_integration"] = True

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    reference = generate()
    current = apply_drift(reference)
    reference.to_parquet(os.path.join(args.out_dir, "churn.parquet"))
    current.to_parquet(os.path.join(args.out_dir, "churn_drifted.parquet"))

    print(f"Wrote {len(reference)} rows to {args.out_dir}/churn.parquet")
    print(f"Wrote {len(current)} rows to {args.out_dir}/churn_drifted.parquet")
    print(f"Churn rate: {reference['churned'].mean():.1%}")
