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

CURRENT_AS_OF = "2018-01-01"

SELLER_CATEGORICAL = [
    "sector", "customer_segment", "regional_office", "manager",
    "seller_specialty", "seller_focus_sector",
]
SELLER_NUMERIC = [
    "company_age", "revenue", "employees", "is_subsidiary", "is_domestic",
    "agent_win_rate", "agent_avg_deal_value", "agent_total_deals", "agent_sector_winrate",
]

LEAD_CATEGORICAL = ["sector", "customer_segment"]
LEAD_NUMERIC = [
    "company_age", "revenue", "employees", "is_subsidiary", "is_domestic",
    "total_opps_hist", "win_rate_hist", "total_value_won_hist",
    "avg_deal_value_hist", "recency_days_hist",
]

SPECIALTY_CATEGORIES = {
    "EPI e seguranca": {"EPI"},
    "ferramentas e abrasivos": {"Ferramentas", "Abrasivos"},
    "solda e metalmecanica": {"Solda", "Abrasivos", "EPI"},
    "lubrificantes e graxas": {"Lubrificantes"},
    "combustiveis e frota": {"Combustiveis", "Lubrificantes"},
    "manutencao industrial": {"Ferramentas", "Lubrificantes", "EPI"},
    "campo e saneamento": {"EPI", "Lubrificantes"},
    "agroindustria": {"EPI", "Lubrificantes"},
}

CATEGORY_OFFERS = {
    "EPI": "renovar kit de EPI e contrato de reposicao",
    "Ferramentas": "ofertar ferramentas de manutencao e reposicao",
    "Abrasivos": "repor abrasivos por consumo recorrente",
    "Solda": "avaliar pacote de solda e consumiveis",
    "Lubrificantes": "ofertar lubrificantes, graxas e plano preventivo",
    "Combustiveis": "negociar abastecimento de frota e recorrencia",
    "Sem historico": "mapear demanda inicial por categoria industrial",
}


def _validate_artifacts(artifacts: dict) -> None:
    seller_model = artifacts["seller_match_model"]
    lead_model = artifacts["lead_value_model"]

    if not hasattr(seller_model, "predict_proba"):
        raise TypeError(
            "Artefato invalido: seller_match_model.joblib precisa ser um classificador "
            "com predict_proba. Reexecute src/train_seller_match.py."
        )

    if not hasattr(lead_model, "predict"):
        raise TypeError(
            "Artefato invalido: lead_value_model.joblib precisa ser um regressor "
            "com predict. Reexecute src/train_lead_scoring.py."
        )

    lead_estimator = getattr(lead_model, "named_steps", {}).get("model")
    if lead_estimator is not None and hasattr(lead_estimator, "predict_proba"):
        raise TypeError(
            "Artefato invalido: lead_value_model.joblib parece ser um classificador, "
            "mas o ranking da urna espera um modelo de regressao. "
            "Reexecute src/train_lead_scoring.py."
        )


def load_artifacts(base_dir: str) -> dict:
    models_dir = os.path.join(base_dir, "models")
    processed_dir = os.path.join(base_dir, "data", "processed")

    with open(os.path.join(processed_dir, "global_stats.json")) as f:
        global_stats = json.load(f)

    artifacts = {
        "seller_match_model": joblib.load(os.path.join(models_dir, "seller_match_model.joblib")),
        "lead_value_model": joblib.load(os.path.join(models_dir, "lead_value_model.joblib")),
        "accounts_features": pd.read_csv(os.path.join(processed_dir, "accounts_features.csv")),
        "seller_profiles": pd.read_csv(os.path.join(processed_dir, "seller_profiles.csv")),
        "seller_sector_winrate": pd.read_csv(os.path.join(processed_dir, "seller_sector_winrate.csv")),
        "opportunity_table": pd.read_csv(os.path.join(processed_dir, "opportunity_table.csv")),
        "urna_features": pd.read_csv(os.path.join(processed_dir, "urna_features.csv")),
        "global_stats": global_stats,
    }
    _validate_artifacts(artifacts)
    return artifacts


def _minmax(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(0.5, index=series.index)
    return (series - min_value) / (max_value - min_value)


def _recency_factor(recency_days: pd.Series) -> pd.Series:
    """Prioriza contas recentes sem punir agressivamente contas dormentes."""
    capped = recency_days.fillna(365).clip(lower=0, upper=365)
    return 0.75 + 0.50 * (1 - capped / 365)


def _effort_index(recency_days: pd.Series, total_opps: pd.Series, win_rate: pd.Series) -> pd.Series:
    """Estima esforco comercial: menor quando ha relacao recente, historico e conversao."""
    recency_penalty = recency_days.fillna(365).clip(lower=0, upper=365) / 365
    history_relief = _minmax(total_opps.fillna(0))
    conversion_relief = win_rate.fillna(0).clip(0, 1)
    effort = 1.15 + 0.85 * recency_penalty - 0.35 * history_relief - 0.25 * conversion_relief
    return effort.clip(lower=0.55, upper=2.0)


def _effort_label(effort: float) -> str:
    if effort <= 0.85:
        return "baixo"
    if effort <= 1.20:
        return "medio"
    return "alto"


def _next_offer(category: str) -> str:
    return CATEGORY_OFFERS.get(category, "validar necessidade ativa e mix de compra")


def _account_category_profile(opportunity_table: pd.DataFrame) -> pd.DataFrame:
    opp = opportunity_table.copy()
    if "series" not in opp.columns:
        opp["series"] = "Outros"

    won = opp[opp["target_won"] == 1].copy()
    if won.empty:
        return pd.DataFrame(columns=["account", "top_category", "top_category_share", "category_value_won"])

    by_category = (
        won.groupby(["account", "series"])["close_value"]
        .sum()
        .rename("category_value_won")
        .reset_index()
    )
    totals = by_category.groupby("account")["category_value_won"].sum().rename("total_category_value")
    by_category = by_category.merge(totals, on="account", how="left")
    by_category["top_category_share"] = by_category["category_value_won"] / by_category["total_category_value"]

    top = (
        by_category.sort_values(["account", "category_value_won"], ascending=[True, False])
        .drop_duplicates("account")
        .rename(columns={"series": "top_category"})
    )
    return top[["account", "top_category", "top_category_share", "category_value_won"]]


def _seller_category_factor(profile: pd.Series, categories: pd.Series) -> pd.Series:
    specialty = profile["seller_specialty"]
    preferred = SPECIALTY_CATEGORIES.get(specialty, set())
    return categories.map(lambda category: 1.18 if category in preferred else 0.96)


def _reason_match(row: pd.Series) -> str:
    reasons = []
    if row["sector_affinity"] > 1:
        reasons.append("setor foco")
    if row["category_affinity"] > 1:
        reasons.append(f"categoria {row['top_category']}")
    if row["recency_factor"] >= 1.05:
        reasons.append("contato recente")
    if row["value_factor"] >= 1.15:
        reasons.append("alto potencial")
    return ", ".join(reasons[:3]) if reasons else "fit estatistico consistente"


def _reason_lead(row: pd.Series) -> str:
    reasons = []
    if row["lead_score"] >= row["lead_score_p67"]:
        reasons.append("alto valor esperado")
    if row["win_rate_hist"] >= 0.65:
        reasons.append("bom historico de conversao")
    if row["recency_factor"] >= 1.05:
        reasons.append("relacao recente")
    if row["total_value_won_hist"] > 0:
        reasons.append("comprador recorrente")
    return ", ".join(reasons[:3]) if reasons else "potencial moderado"


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
    accounts["seller_specialty"] = profile["seller_specialty"]
    accounts["seller_focus_sector"] = profile["seller_focus_sector"]
    accounts["agent_win_rate"] = profile["agent_win_rate"]
    accounts["agent_avg_deal_value"] = profile["agent_avg_deal_value"]
    accounts["agent_total_deals"] = profile["agent_total_deals"]

    X = accounts[SELLER_CATEGORICAL + SELLER_NUMERIC]
    accounts["match_score"] = artifacts["seller_match_model"].predict_proba(X)[:, 1]

    lead_pool = artifacts["urna_features"].copy()
    lead_pool["lead_score"] = artifacts["lead_value_model"].predict(lead_pool[LEAD_CATEGORICAL + LEAD_NUMERIC])
    lead_pool["lead_score"] = lead_pool["lead_score"].clip(lower=0)
    lead_pool["cliente_base_score"] = (
        lead_pool["lead_score"]
        * _recency_factor(lead_pool["recency_days_hist"])
        * (0.85 + 0.50 * lead_pool["win_rate_hist"].fillna(0).clip(0, 1))
    )
    lead_pool["score_cliente"] = 100 * _minmax(lead_pool["cliente_base_score"])
    lead_cols = [
        "account", "lead_score", "score_cliente", "win_rate_hist",
        "total_value_won_hist", "recency_days_hist",
    ]
    accounts = accounts.merge(lead_pool[lead_cols], on="account", how="left")

    category_profile = _account_category_profile(artifacts["opportunity_table"])
    accounts = accounts.merge(category_profile, on="account", how="left")
    accounts["top_category"] = accounts["top_category"].fillna("Sem historico")
    accounts["top_category_share"] = accounts["top_category_share"].fillna(0)

    accounts["value_factor"] = 0.75 + 0.75 * _minmax(accounts["lead_score"].fillna(0))
    accounts["sector_affinity"] = accounts["sector"].eq(profile["seller_focus_sector"]).map({True: 1.18, False: 0.97})
    accounts["category_affinity"] = _seller_category_factor(profile, accounts["top_category"])
    accounts["recency_factor"] = _recency_factor(accounts["recency_days_hist"])
    accounts["valor_esperado"] = accounts["match_score"] * accounts["lead_score"].fillna(0)
    accounts["esforco_comercial"] = _effort_index(
        accounts["recency_days_hist"],
        accounts["agent_total_deals"],
        accounts["win_rate_hist"],
    )
    accounts["score_custo_beneficio"] = accounts["valor_esperado"] / accounts["esforco_comercial"]
    accounts["score_custo_beneficio"] = 100 * _minmax(accounts["score_custo_beneficio"])
    accounts["nivel_esforco"] = accounts["esforco_comercial"].map(_effort_label)
    accounts["proxima_oferta"] = accounts["top_category"].map(_next_offer)
    accounts["score_comercial"] = (
        accounts["match_score"]
        * accounts["value_factor"]
        * accounts["sector_affinity"]
        * accounts["category_affinity"]
        * accounts["recency_factor"]
        * (0.85 + 0.30 * accounts["score_custo_beneficio"] / 100)
    )
    accounts["score_comercial"] = 100 * _minmax(accounts["score_comercial"])
    accounts["motivo"] = accounts.apply(_reason_match, axis=1)

    accounts = accounts.sort_values("score_comercial", ascending=False).reset_index(drop=True)
    accounts["rank"] = accounts.index + 1

    if top_n:
        accounts = accounts.head(top_n)

    cols = [
        "rank", "account", "sector", "customer_segment", "score_comercial", "match_score",
        "lead_score", "score_cliente", "valor_esperado", "score_custo_beneficio", "nivel_esforco",
        "top_category", "proxima_oferta", "motivo", "revenue", "employees", "company_age",
        "is_subsidiary", "is_domestic", "win_rate_hist", "total_value_won_hist", "recency_days_hist",
    ]
    return accounts[cols]


def rank_lead_pool(artifacts: dict, top_n: int = None) -> pd.DataFrame:
    """Cenario 2: ranking da urna de clientes por valor futuro esperado."""
    urna = artifacts["urna_features"].copy()

    X = urna[LEAD_CATEGORICAL + LEAD_NUMERIC]
    urna["lead_score"] = artifacts["lead_value_model"].predict(X)
    urna["lead_score"] = urna["lead_score"].clip(lower=0)

    category_profile = _account_category_profile(artifacts["opportunity_table"])
    urna = urna.merge(category_profile, on="account", how="left")
    urna["top_category"] = urna["top_category"].fillna("Sem historico")
    urna["top_category_share"] = urna["top_category_share"].fillna(0)

    urna["recency_factor"] = _recency_factor(urna["recency_days_hist"])
    urna["value_factor"] = 0.75 + 0.75 * _minmax(urna["lead_score"])
    urna["winrate_factor"] = 0.85 + 0.50 * urna["win_rate_hist"].fillna(0).clip(0, 1)
    urna["cliente_base_score"] = urna["lead_score"] * urna["recency_factor"] * urna["winrate_factor"]
    urna["score_cliente"] = 100 * _minmax(urna["cliente_base_score"])
    urna["esforco_comercial"] = _effort_index(
        urna["recency_days_hist"],
        urna["total_opps_hist"],
        urna["win_rate_hist"],
    )
    urna["score_custo_beneficio"] = urna["lead_score"] / urna["esforco_comercial"]
    urna["score_custo_beneficio"] = 100 * _minmax(urna["score_custo_beneficio"])
    urna["nivel_esforco"] = urna["esforco_comercial"].map(_effort_label)
    urna["proxima_oferta"] = urna["top_category"].map(_next_offer)
    urna["score_prioridade"] = (
        urna["lead_score"]
        * urna["recency_factor"]
        * urna["winrate_factor"]
        * (0.85 + 0.30 * urna["score_custo_beneficio"] / 100)
    )
    urna["score_prioridade"] = 100 * _minmax(urna["score_prioridade"])

    urna = urna.sort_values("score_prioridade", ascending=False).reset_index(drop=True)
    urna["rank"] = urna.index + 1

    q1, q2 = urna["lead_score"].quantile([1 / 3, 2 / 3])
    p67 = urna["lead_score"].quantile(2 / 3)
    urna["lead_score_p67"] = p67
    urna["temperatura"] = pd.cut(
        urna["lead_score"], bins=[-float("inf"), q1, q2, float("inf")],
        labels=["frio", "morno", "quente"],
    )
    urna["motivo"] = urna.apply(_reason_lead, axis=1)

    if top_n:
        urna = urna.head(top_n)

    cols = [
        "rank", "account", "sector", "customer_segment", "score_prioridade", "lead_score",
        "score_cliente", "score_custo_beneficio", "nivel_esforco", "temperatura", "top_category",
        "proxima_oferta", "motivo", "revenue", "employees", "win_rate_hist",
        "total_value_won_hist", "recency_days_hist",
    ]
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
