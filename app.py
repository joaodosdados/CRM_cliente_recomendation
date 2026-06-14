"""
Dashboard Streamlit para priorizacao comercial B2B industrial.

A tela foi desenhada para uso operacional: o vendedor escolhe a carteira,
visualiza uma lista curta de contas priorizadas e recebe contexto suficiente
para decidir a proxima ligacao sem depender de rolagem longa.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from inference import load_artifacts, rank_lead_pool, recommend_clients_for_seller  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="CRM Industrial - Cockpit Comercial", layout="wide")


@st.cache_resource
def get_artifacts(cache_version: str):
    return load_artifacts(BASE_DIR)


def money(value: float) -> str:
    return f"$ {value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def action_for_match(score: float) -> str:
    if score >= 70:
        return "Ligar hoje"
    if score >= 55:
        return "Priorizar na semana"
    return "Nutrir antes da abordagem"


def action_for_lead(row: pd.Series) -> str:
    if row["score_prioridade"] >= 70:
        return "Abrir contato comercial"
    if row["score_prioridade"] >= 45:
        return "Validar necessidade ativa"
    return "Manter em cadencia leve"


def account_history(artifacts: dict, account: str) -> dict:
    opp = artifacts.get("opportunity_table")
    if opp is None:
        opp_path = os.path.join(BASE_DIR, "data", "processed", "opportunity_table.csv")
        opp = pd.read_csv(opp_path)
    account_opp = opp[opp["account"] == account].copy()

    if account_opp.empty:
        return {
            "total_opps": 0,
            "won_rate": 0,
            "won_value": 0,
            "last_engage": "-",
            "top_products": pd.DataFrame(columns=["product", "deals", "value"]),
        }

    account_opp["engage_date"] = pd.to_datetime(account_opp["engage_date"])
    won = account_opp[account_opp["target_won"] == 1]
    product_mix = (
        account_opp.groupby(["series", "product"], dropna=False)
        .agg(deals=("opportunity_id", "count"), value=("close_value", "sum"))
        .sort_values(["deals", "value"], ascending=False)
        .head(4)
        .reset_index()
    )

    return {
        "total_opps": len(account_opp),
        "won_rate": float(account_opp["target_won"].mean()),
        "won_value": float(won["close_value"].sum()),
        "last_engage": account_opp["engage_date"].max().date().isoformat(),
        "top_products": product_mix,
    }


def render_account_panel(row: pd.Series, artifacts: dict, mode: str) -> None:
    history = account_history(artifacts, row["account"])

    st.markdown(f"### {row['account']}")
    st.caption(f"{row['sector']} | {row['customer_segment']}")

    if mode == "match":
        score = row["score_comercial"]
        st.metric("Score comercial", f"{score:.0f}/100")
        st.success(action_for_match(score))
        st.caption(
            f"Fit estatistico: {pct(row['match_score'])} | "
            f"Score cliente: {row['score_cliente']:.0f}/100 | "
            f"Potencial: {money(row['lead_score'])} | "
            f"Custo-beneficio: {row['score_custo_beneficio']:.0f}/100"
        )
    else:
        st.metric("Score de prioridade", f"{row['score_prioridade']:.0f}/100")
        st.success(action_for_lead(row))
        st.caption(
            f"Potencial estimado: {money(row['lead_score'])} | "
            f"Score cliente: {row['score_cliente']:.0f}/100 | "
            f"Temperatura: {row['temperatura']} | "
            f"Custo-beneficio: {row['score_custo_beneficio']:.0f}/100"
        )

    st.info(f"Motivo: {row['motivo']}")
    st.caption(f"Categoria dominante: {row['top_category']} | Esforco: {row['nivel_esforco']}")
    st.markdown(f"**Gancho comercial:** {row['proxima_oferta']}")

    k1, k2, k3 = st.columns(3)
    k1.metric("Receita cliente", money(row["revenue"]))
    k2.metric("Funcionarios", f"{int(row['employees']):,}")
    k3.metric("Idade", f"{int(row['company_age'])} anos" if "company_age" in row else "-")

    h1, h2, h3 = st.columns(3)
    h1.metric("Oportunidades", int(history["total_opps"]))
    h2.metric("Win rate hist.", pct(history["won_rate"]))
    h3.metric("Valor ganho", money(history["won_value"]))

    st.caption(f"Ultimo contato registrado: {history['last_engage']}")

    product_mix = history["top_products"]
    if not product_mix.empty:
        product_mix["value"] = product_mix["value"].map(money)
        st.dataframe(product_mix, use_container_width=True, hide_index=True, height=145)


def compact_match_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[
        "rank", "account", "sector", "score_comercial",
        "score_cliente", "score_custo_beneficio", "nivel_esforco", "motivo",
    ]].copy()
    out["score_comercial"] = out["score_comercial"].map(lambda value: f"{value:.0f}")
    out["score_cliente"] = out["score_cliente"].map(lambda value: f"{value:.0f}")
    out["score_custo_beneficio"] = out["score_custo_beneficio"].map(lambda value: f"{value:.0f}")
    out.columns = ["#", "Cliente", "Setor", "Score", "Score cliente", "C/B", "Esforco", "Motivo"]
    return out


def compact_lead_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[
        "rank", "account", "sector", "score_prioridade",
        "score_cliente", "score_custo_beneficio", "nivel_esforco", "lead_score", "motivo",
    ]].copy()
    out["score_prioridade"] = out["score_prioridade"].map(lambda value: f"{value:.0f}")
    out["score_cliente"] = out["score_cliente"].map(lambda value: f"{value:.0f}")
    out["score_custo_beneficio"] = out["score_custo_beneficio"].map(lambda value: f"{value:.0f}")
    out["lead_score"] = out["lead_score"].map(money)
    out.columns = ["#", "Cliente", "Setor", "Score", "Score cliente", "C/B", "Esforco", "Potencial", "Motivo"]
    return out


artifacts = get_artifacts("commercial_score_v2")
profiles = artifacts["seller_profiles"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9dee7;
        padding: 8px 10px;
        border-radius: 6px;
    }
    div[data-testid="stMetric"] label {font-size: 0.78rem;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {font-size: 1.25rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    h1, h2, h3 {letter-spacing: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cockpit Comercial Industrial")

tab_match, tab_pool = st.tabs(["Carteira por vendedor", "Urna de clientes"])

with tab_match:
    c1, c2, c3, c4 = st.columns([2.2, 1.6, 1.4, 1])
    agent = c1.selectbox("Vendedor", sorted(profiles["sales_agent"].unique()))
    profile = profiles[profiles["sales_agent"] == agent].iloc[0]

    raw_ranking = recommend_clients_for_seller(artifacts, agent, top_n=None)
    sectors = ["Todos"] + sorted(raw_ranking["sector"].unique())
    selected_sector = c2.selectbox("Setor", sectors)
    min_score = c3.slider("Score minimo", 0, 100, 45, 1)
    limit = c4.number_input("Top", min_value=5, max_value=25, value=12, step=1)

    ranking = raw_ranking[raw_ranking["score_comercial"] >= min_score]
    if selected_sector != "Todos":
        ranking = ranking[ranking["sector"] == selected_sector]
    ranking = ranking.head(int(limit))

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Especialidade", profile["seller_specialty"])
    p2.metric("Setor foco", profile["seller_focus_sector"])
    p3.metric("Win rate", pct(profile["agent_win_rate"]))
    p4.metric("Ticket medio", money(profile["agent_avg_deal_value"]))

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Prioridade de ligacao")
        if ranking.empty:
            st.warning("Nenhum cliente atende aos filtros atuais.")
        else:
            selected_account = st.selectbox(
                "Cliente em analise",
                ranking["account"].tolist(),
                label_visibility="collapsed",
            )
            st.dataframe(
                compact_match_table(ranking),
                use_container_width=True,
                hide_index=True,
                height=340,
            )
    with right:
        if not ranking.empty:
            selected_row = ranking[ranking["account"] == selected_account].iloc[0]
            render_account_panel(selected_row, artifacts, mode="match")

with tab_pool:
    full_ranking = rank_lead_pool(artifacts)

    c1, c2, c3, c4 = st.columns([1.8, 1.5, 1.4, 1])
    pool_sectors = ["Todos"] + sorted(full_ranking["sector"].unique())
    pool_sector = c1.selectbox("Setor da urna", pool_sectors)
    temperatures = c2.multiselect("Temperatura", ["quente", "morno", "frio"], default=["quente", "morno"])
    min_value = c3.number_input("Score minimo", min_value=0, max_value=100, value=35, step=5)
    pool_limit = c4.number_input("Top clientes", min_value=5, max_value=30, value=15, step=1)

    filtered = full_ranking[full_ranking["temperatura"].isin(temperatures)]
    filtered = filtered[filtered["score_prioridade"] >= min_value]
    if pool_sector != "Todos":
        filtered = filtered[filtered["sector"] == pool_sector]
    filtered = filtered.head(int(pool_limit))

    counts = full_ranking["temperatura"].value_counts().reindex(["quente", "morno", "frio"]).fillna(0)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes quentes", int(counts["quente"]))
    k2.metric("Clientes mornos", int(counts["morno"]))
    k3.metric("Clientes frios", int(counts["frio"]))
    k4.metric("Potencial top", money(float(filtered["lead_score"].sum())) if not filtered.empty else "$ 0")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Urna priorizada")
        if filtered.empty:
            st.warning("Nenhum cliente atende aos filtros atuais.")
        else:
            selected_pool_account = st.selectbox(
                "Cliente da urna em analise",
                filtered["account"].tolist(),
                label_visibility="collapsed",
            )
            st.dataframe(
                compact_lead_table(filtered),
                use_container_width=True,
                hide_index=True,
                height=315,
            )

        funnel_df = counts.reset_index()
        funnel_df.columns = ["temperatura", "quantidade"]
        fig = px.bar(
            funnel_df,
            x="quantidade",
            y="temperatura",
            orientation="h",
            color="temperatura",
            color_discrete_map={"quente": "#b42318", "morno": "#b54708", "frio": "#175cd3"},
            height=130,
        )
        fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if not filtered.empty:
            selected_pool_row = filtered[filtered["account"] == selected_pool_account].iloc[0]
            render_account_panel(selected_pool_row, artifacts, mode="lead")
