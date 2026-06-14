"""
Preparacao de dados: le as tabelas brutas do CRM (accounts, products, sales_pipeline,
sales_teams), constroi a feature store de contas (firmograficas, validas inclusive
para leads novos sem historico) e a tabela de oportunidades usada para treino dos
modelos. Tambem calcula os perfis agregados de cada vendedor e a taxa de conversao
historica por (vendedor, setor), com suavizacao bayesiana.

Saidas em data/processed/:
  - accounts_features.csv     -> 1 linha por conta (cliente)
  - opportunity_table.csv      -> 1 linha por oportunidade fechada (Won/Lost)
  - seller_profiles.csv         -> 1 linha por vendedor
  - seller_sector_winrate.csv   -> taxa de conversao suavizada por (vendedor, setor)
  - global_stats.json           -> estatisticas globais usadas como fallback
"""

import json
import os

import pandas as pd

REFERENCE_YEAR = 2018  # ano usado para calcular "idade da empresa" (dado historico ate 2017)
SMOOTHING_K = 5         # forca da suavizacao bayesiana para taxa de conversao por setor/vendedor

TRAIN_CUTOFF = "2017-07-01"   # divide o historico em janela de treino (passado) e alvo (futuro)
CURRENT_AS_OF = "2018-01-01"  # data de referencia "hoje" para o snapshot atual da urna


def load_raw(data_dir: str) -> dict:
    return {
        "accounts": pd.read_csv(os.path.join(data_dir, "accounts.csv")),
        "products": pd.read_csv(os.path.join(data_dir, "products.csv")),
        "pipeline": pd.read_csv(os.path.join(data_dir, "sales_pipeline.csv")),
        "teams": pd.read_csv(os.path.join(data_dir, "sales_teams.csv")),
    }


def build_accounts_features(accounts: pd.DataFrame) -> pd.DataFrame:
    """Features firmograficas por conta. Disponiveis mesmo para um lead novo
    (nao depende de historico de compra)."""
    df = accounts.copy()
    df["sector"] = df["sector"].str.strip().str.lower()
    df["company_age"] = REFERENCE_YEAR - df["year_established"]
    df["is_subsidiary"] = df["subsidiary_of"].notna().astype(int)
    df["is_domestic"] = (df["office_location"] == "United States").astype(int)

    cols = [
        "account", "sector", "company_age", "revenue", "employees",
        "is_subsidiary", "is_domestic",
    ]
    return df[cols]


def build_opportunity_table(pipeline: pd.DataFrame, accounts_features: pd.DataFrame,
                             teams: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por oportunidade encerrada (Won/Lost), com features da conta e
    do vendedor responsavel. Oportunidades em aberto (Engaging/Prospecting) ou
    sem conta associada sao descartadas do treino."""
    df = pipeline[pipeline["deal_stage"].isin(["Won", "Lost"])].copy()
    df = df.dropna(subset=["account"])
    df["target_won"] = (df["deal_stage"] == "Won").astype(int)

    df = df.merge(accounts_features, on="account", how="left")
    df = df.merge(teams, on="sales_agent", how="left")

    return df


def build_account_rfm(opportunity_table: pd.DataFrame, as_of_date: str,
                       window_start: str = None) -> pd.DataFrame:
    """Agrega Recencia/Frequencia/Valor por conta usando oportunidades encerradas
    com engage_date dentro de [window_start, as_of_date)."""
    df = opportunity_table.copy()
    df["engage_date"] = pd.to_datetime(df["engage_date"])

    if window_start is not None:
        df = df[df["engage_date"] >= pd.Timestamp(window_start)]
    df = df[df["engage_date"] < pd.Timestamp(as_of_date)]

    freq = df.groupby("account").agg(
        total_opps=("target_won", "size"),
        win_rate=("target_won", "mean"),
    )
    value = df[df["target_won"] == 1].groupby("account")["close_value"].agg(
        total_value_won="sum", avg_deal_value="mean"
    )
    last_engage = df.groupby("account")["engage_date"].max().rename("last_engage_date")

    out = freq.join(value, how="left").join(last_engage, how="left")
    out["total_value_won"] = out["total_value_won"].fillna(0)
    out["avg_deal_value"] = out["avg_deal_value"].fillna(0)
    out["recency_days"] = (pd.Timestamp(as_of_date) - out["last_engage_date"]).dt.days
    return out.drop(columns=["last_engage_date"]).reset_index()


def build_lead_scoring_table(opportunity_table: pd.DataFrame, accounts_features: pd.DataFrame,
                              train_cutoff: str = TRAIN_CUTOFF) -> pd.DataFrame:
    """Tabela de treino do Cenario 2: features RFM + firmograficas calculadas ANTES
    do corte temporal, alvo = receita gerada (Won) DEPOIS do corte (potencial futuro)."""
    rfm = build_account_rfm(opportunity_table, as_of_date=train_cutoff)
    rfm = rfm.rename(columns={c: f"{c}_hist" for c in rfm.columns if c != "account"})

    future = opportunity_table.copy()
    future["engage_date"] = pd.to_datetime(future["engage_date"])
    future = future[future["engage_date"] >= pd.Timestamp(train_cutoff)]
    target = future[future["target_won"] == 1].groupby("account")["close_value"].sum().rename("future_value")

    table = accounts_features.merge(rfm, on="account", how="left").merge(target, on="account", how="left")
    hist_cols = [c for c in table.columns if c.endswith("_hist")]
    table[hist_cols] = table[hist_cols].fillna(0)
    table["future_value"] = table["future_value"].fillna(0)
    return table


def build_urna_features(opportunity_table: pd.DataFrame, accounts_features: pd.DataFrame,
                         as_of_date: str = CURRENT_AS_OF) -> pd.DataFrame:
    """Snapshot atual ('urna') com o mesmo schema de features usado no treino do
    Cenario 2, calculado a partir de todo o historico disponivel."""
    rfm = build_account_rfm(opportunity_table, as_of_date=as_of_date)
    rfm = rfm.rename(columns={c: f"{c}_hist" for c in rfm.columns if c != "account"})

    table = accounts_features.merge(rfm, on="account", how="left")
    hist_cols = [c for c in table.columns if c.endswith("_hist")]
    table[hist_cols] = table[hist_cols].fillna(0)
    return table



def build_seller_profiles(opportunity_table: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Perfil agregado de cada vendedor: regiao, gestor e historico de performance."""
    won_value = opportunity_table.loc[opportunity_table["target_won"] == 1].groupby("sales_agent")["close_value"].mean()

    agg = opportunity_table.groupby("sales_agent").agg(
        agent_total_deals=("target_won", "size"),
        agent_win_rate=("target_won", "mean"),
    ).reset_index()
    agg["agent_avg_deal_value"] = agg["sales_agent"].map(won_value).fillna(0)

    profiles = teams.merge(agg, on="sales_agent", how="left")
    profiles[["agent_total_deals", "agent_win_rate", "agent_avg_deal_value"]] = (
        profiles[["agent_total_deals", "agent_win_rate", "agent_avg_deal_value"]].fillna(0)
    )
    return profiles


def build_seller_sector_winrate(opportunity_table: pd.DataFrame, seller_profiles: pd.DataFrame,
                                 k: int = SMOOTHING_K) -> pd.DataFrame:
    """Taxa de conversao por (vendedor, setor), suavizada em direcao a taxa
    historica geral do vendedor quando ha poucas amostras naquele setor."""
    grp = opportunity_table.groupby(["sales_agent", "sector"]).agg(
        sector_deals=("target_won", "size"),
        sector_wins=("target_won", "sum"),
    ).reset_index()

    grp = grp.merge(seller_profiles[["sales_agent", "agent_win_rate"]], on="sales_agent", how="left")
    grp["agent_sector_winrate"] = (
        (grp["sector_wins"] + k * grp["agent_win_rate"]) / (grp["sector_deals"] + k)
    )
    return grp[["sales_agent", "sector", "agent_sector_winrate"]]


def run(data_dir: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    raw = load_raw(data_dir)

    accounts_features = build_accounts_features(raw["accounts"])
    opportunity_table = build_opportunity_table(raw["pipeline"], accounts_features, raw["teams"])
    seller_profiles = build_seller_profiles(opportunity_table, raw["teams"])
    seller_sector_winrate = build_seller_sector_winrate(opportunity_table, seller_profiles)
    lead_scoring_table = build_lead_scoring_table(opportunity_table, accounts_features)
    urna_features = build_urna_features(opportunity_table, accounts_features)

    global_stats = {
        "global_win_rate": float(opportunity_table["target_won"].mean()),
        "global_avg_deal_value": float(opportunity_table.loc[opportunity_table["target_won"] == 1, "close_value"].mean()),
    }

    accounts_features.to_csv(os.path.join(output_dir, "accounts_features.csv"), index=False)
    opportunity_table.to_csv(os.path.join(output_dir, "opportunity_table.csv"), index=False)
    seller_profiles.to_csv(os.path.join(output_dir, "seller_profiles.csv"), index=False)
    seller_sector_winrate.to_csv(os.path.join(output_dir, "seller_sector_winrate.csv"), index=False)
    lead_scoring_table.to_csv(os.path.join(output_dir, "lead_scoring_table.csv"), index=False)
    urna_features.to_csv(os.path.join(output_dir, "urna_features.csv"), index=False)
    with open(os.path.join(output_dir, "global_stats.json"), "w") as f:
        json.dump(global_stats, f, indent=2)

    return {
        "accounts_features": accounts_features,
        "opportunity_table": opportunity_table,
        "seller_profiles": seller_profiles,
        "seller_sector_winrate": seller_sector_winrate,
        "lead_scoring_table": lead_scoring_table,
        "urna_features": urna_features,
        "global_stats": global_stats,
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = run(os.path.join(base, "data"), os.path.join(base, "data", "processed"))
    print("accounts_features:", result["accounts_features"].shape)
    print("opportunity_table:", result["opportunity_table"].shape)
    print("seller_profiles:", result["seller_profiles"].shape)
    print("seller_sector_winrate:", result["seller_sector_winrate"].shape)
    print("lead_scoring_table:", result["lead_scoring_table"].shape)
    print("urna_features:", result["urna_features"].shape)
    print("global_stats:", result["global_stats"])
