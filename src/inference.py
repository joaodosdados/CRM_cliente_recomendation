"""
Inferencia em tempo real para os dois cenarios do projeto.

Cenario 1 - Match Vendedor x Cliente:
  Dado um vendedor, calcula para cada conta da base um score de probabilidade
  de fechamento (Won) considerando o perfil do vendedor + perfil firmografico
  da conta. Retorna o ranking de clientes recomendados para aquele vendedor.

Cenario 2 - Ranking da Urna (Lead Scoring):
  Para o snapshot atual das contas (urna), calcula o valor futuro esperado de
  cada conta e classifica em faixas de temperatura (quente/morno/frio) para
  priorizacao do time de vendas.

Os artefatos sao carregados uma unica vez (treino offline) e usados aqui apenas
para pontuacao -> scoring em tempo real, sem re-treino.
"""

import json
import os

import joblib
import pandas as pd

SELLER_CATEGORICAL = ["sector", "regional_office", "manager"]
SELLER_NUMERIC = [
    "company_age", "revenue", "employees", "is_subsidiary", "is_domestic",
    "agent_win_rate", "agent_avg_deal_value", "agent_total_deals", "agent_sector_winrate",
]

LEAD_CATEGORICAL = ["sector"]
LEAD_NUMERIC = [
    "company_age", "revenue", "employees", "is_subsidiary", "is_domestic",
    "total_opps_hist", "win_rate_hist", "total_value_won_hist",
    "avg_deal_value_hist", "recency_days_hist",
]


def load_artifacts(base_dir: str) -> dict:
    models_dir = os.path.join(base_dir, "models")
    processed_dir = os.path.join(base_dir, "data", "processed")

    with open(os.path.join(processed_dir, "global_stats.json")) as f:
        global_stats = json.load(f)

    return {
        "seller_match_model": joblib.load(os.path.join(models_dir, "seller_match_model.joblib")),
        "lead_value_model": joblib.load(os.path.join(models_dir, "lead_value_model.joblib")),
        "accounts_features": pd.read_csv(os.path.join(processed_dir, "accounts_features.csv")),
        "seller_profiles": pd.read_csv(os.path.join(processed_dir, "seller_profiles.csv")),
        "seller_sector_winrate": pd.read_csv(os.path.join(processed_dir, "seller_sector_winrate.csv")),
        "urna_features": pd.read_csv(os.path.join(processed_dir, "urna_features.csv")),
        "global_stats": global_stats,
    }


def recommend_clients_for_seller(artifacts: dict, sales_agent: str, top_n: int = None) -> pd.DataFrame:
    """Cenario 1: ranking de clientes recomendados para um vendedor especifico."""
    profiles = artifacts["seller_profiles"]
    profile_row = profiles[profiles["sales_agent"] == sales_agent]
    if profile_row.empty:
        raise ValueError(f"Vendedor '{sales_agent}' nao encontrado em seller_profiles")
    profile = profile_row.iloc[0]

    accounts = artifacts["accounts_features"].copy()

    sector_wr = artifacts["seller_sector_winrate"]
    agent_sector = sector_wr[sector_wr["sales_agent"] == sales_agent][["sector", "agent_sector_winrate"]]
    accounts = accounts.merge(agent_sector, on="sector", how="left")
    accounts["agent_sector_winrate"] = accounts["agent_sector_winrate"].fillna(profile["agent_win_rate"])

    accounts["regional_office"] = profile["regional_office"]
    accounts["manager"] = profile["manager"]
    accounts["agent_win_rate"] = profile["agent_win_rate"]
    accounts["agent_avg_deal_value"] = profile["agent_avg_deal_value"]
    accounts["agent_total_deals"] = profile["agent_total_deals"]

    X = accounts[SELLER_CATEGORICAL + SELLER_NUMERIC]
    accounts["match_score"] = artifacts["seller_match_model"].predict_proba(X)[:, 1]

    accounts = accounts.sort_values("match_score", ascending=False).reset_index(drop=True)
    accounts["rank"] = accounts.index + 1

    if top_n:
        accounts = accounts.head(top_n)

    cols = ["rank", "account", "sector", "match_score", "revenue", "employees",
            "company_age", "is_subsidiary", "is_domestic"]
    return accounts[cols]


def rank_lead_pool(artifacts: dict, top_n: int = None) -> pd.DataFrame:
    """Cenario 2: ranking da urna de clientes por valor futuro esperado."""
    urna = artifacts["urna_features"].copy()

    X = urna[LEAD_CATEGORICAL + LEAD_NUMERIC]
    urna["lead_score"] = artifacts["lead_value_model"].predict(X)
    urna["lead_score"] = urna["lead_score"].clip(lower=0)

    urna = urna.sort_values("lead_score", ascending=False).reset_index(drop=True)
    urna["rank"] = urna.index + 1

    q1, q2 = urna["lead_score"].quantile([1 / 3, 2 / 3])
    urna["temperatura"] = pd.cut(
        urna["lead_score"], bins=[-float("inf"), q1, q2, float("inf")],
        labels=["frio", "morno", "quente"],
    )

    if top_n:
        urna = urna.head(top_n)

    cols = ["rank", "account", "sector", "lead_score", "temperatura", "revenue",
            "employees", "win_rate_hist", "total_value_won_hist", "recency_days_hist"]
    return urna[cols]


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts = load_artifacts(base)

    print("=== Cenario 1: recomendacao para vendedor ===")
    sample_agent = artifacts["seller_profiles"]["sales_agent"].iloc[0]
    print("Vendedor:", sample_agent)
    print(recommend_clients_for_seller(artifacts, sample_agent, top_n=5))

    print("\n=== Cenario 2: ranking da urna ===")
    print(rank_lead_pool(artifacts, top_n=5))
