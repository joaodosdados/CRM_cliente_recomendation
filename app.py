"""
Interface Streamlit para o sistema de recomendacao de clientes para vendedores.

Cenario 1 - Match Vendedor x Cliente:
  Seleciona um vendedor e exibe o ranking de clientes recomendados para o
  perfil dele, com base no modelo de probabilidade de fechamento (Won).

Cenario 2 - Ranking da Urna:
  Exibe o ranking completo da urna de clientes por valor futuro esperado,
  classificado em quente / morno / frio, para priorizacao do time de vendas.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from inference import load_artifacts, rank_lead_pool, recommend_clients_for_seller  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="CRM Lead Scoring - MVP", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts(BASE_DIR)


artifacts = get_artifacts()

st.title("CRM Lead Scoring & Recomendacao de Clientes")

tab1, tab2 = st.tabs(["Match Vendedor x Cliente", "Ranking da Urna"])

# ---------------------------------------------------------------------------
# Cenario 1
# ---------------------------------------------------------------------------
with tab1:
    st.header("Clientes recomendados por vendedor")

    profiles = artifacts["seller_profiles"]
    agent = st.selectbox("Vendedor", sorted(profiles["sales_agent"].unique()))

    profile = profiles[profiles["sales_agent"] == agent].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Escritorio regional", profile["regional_office"])
    col2.metric("Gestor", profile["manager"])
    col3.metric("Win rate historico", f"{profile['agent_win_rate']:.1%}")
    col4.metric("Ticket medio (Won)", f"$ {profile['agent_avg_deal_value']:,.0f}")

    top_n = st.slider("Quantidade de clientes no ranking", 5, 85, 10, key="top_n_seller")

    ranking = recommend_clients_for_seller(artifacts, agent, top_n=top_n)

    st.subheader("Ranking de clientes recomendados")
    st.dataframe(
        ranking.style.format({
            "match_score": "{:.1%}",
            "revenue": "$ {:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        ranking.sort_values("match_score"),
        x="match_score",
        y="account",
        orientation="h",
        color="sector",
        title=f"Top {top_n} clientes recomendados para {agent}",
        labels={"match_score": "Probabilidade de fechamento (Won)", "account": "Cliente"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(400, 25 * top_n))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Cenario 2
# ---------------------------------------------------------------------------
with tab2:
    st.header("Ranking da urna de clientes")

    full_ranking = rank_lead_pool(artifacts)

    sectors = sorted(full_ranking["sector"].unique())
    selected_sectors = st.multiselect("Filtrar por setor", sectors, default=sectors)

    filtered = full_ranking[full_ranking["sector"].isin(selected_sectors)]

    col1, col2, col3 = st.columns(3)
    counts = filtered["temperatura"].value_counts()
    col1.metric("Quentes", int(counts.get("quente", 0)))
    col2.metric("Mornos", int(counts.get("morno", 0)))
    col3.metric("Frios", int(counts.get("frio", 0)))

    funnel_df = (
        filtered["temperatura"]
        .value_counts()
        .reindex(["quente", "morno", "frio"])
        .reset_index()
    )
    funnel_df.columns = ["temperatura", "quantidade"]

    fig_funnel = px.funnel(
        funnel_df,
        x="quantidade",
        y="temperatura",
        title="Funil de priorizacao da urna",
        color="temperatura",
        color_discrete_map={"quente": "#d62728", "morno": "#ff7f0e", "frio": "#1f77b4"},
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.subheader("Ranking completo")
    st.dataframe(
        filtered.style.format({
            "lead_score": "$ {:,.0f}",
            "revenue": "$ {:,.0f}",
            "win_rate_hist": "{:.1%}",
            "total_value_won_hist": "$ {:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
