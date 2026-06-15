"""
Converte a base CRM generica do MVP para um cenario B2B industrial.

O script preserva os CSVs originais em data/original_maven/ na primeira execucao
e reescreve os CSVs de data/ com setores, segmentos, produtos e especialidades
de vendedores mais proximos de distribuidores de EPI, ferramentas e insumos
industriais.
"""

import hashlib
import os
import shutil

import pandas as pd


INDUSTRIAL_SECTORS = [
    "mineracao",
    "construcao de vias",
    "metalurgia",
    "energia",
    "agroindustria",
    "logistica",
    "saneamento",
    "manufatura pesada",
]

SEGMENTS_BY_SECTOR = {
    "mineracao": "operacao extrativa",
    "construcao de vias": "obras e infraestrutura",
    "metalurgia": "producao metalmecanica",
    "energia": "operacao e manutencao",
    "agroindustria": "processamento industrial",
    "logistica": "frota e armazem",
    "saneamento": "campo e manutencao",
    "manufatura pesada": "chao de fabrica",
}

PRODUCTS = pd.DataFrame([
    {"product": "Capacete de seguranca classe B", "series": "EPI", "sales_price": 45},
    {"product": "Bota de seguranca CA", "series": "EPI", "sales_price": 120},
    {"product": "Luva nitrilica industrial", "series": "EPI", "sales_price": 18},
    {"product": "Oculos de protecao ampla visao", "series": "EPI", "sales_price": 35},
    {"product": "Furadeira de impacto industrial", "series": "Ferramentas", "sales_price": 890},
    {"product": "Disco abrasivo corte metal", "series": "Abrasivos", "sales_price": 14},
    {"product": "Eletrodo solda revestido", "series": "Solda", "sales_price": 32},
    {"product": "Graxa alta temperatura", "series": "Lubrificantes", "sales_price": 55},
    {"product": "Oleo lubrificante hidraulico", "series": "Lubrificantes", "sales_price": 280},
    {"product": "Diesel S10 para frota", "series": "Combustiveis", "sales_price": 6},
])

PRODUCT_AFFINITY = {
    "mineracao": [
        "Capacete de seguranca classe B", "Bota de seguranca CA",
        "Graxa alta temperatura", "Oleo lubrificante hidraulico",
        "Diesel S10 para frota",
    ],
    "construcao de vias": [
        "Capacete de seguranca classe B", "Bota de seguranca CA",
        "Furadeira de impacto industrial", "Disco abrasivo corte metal",
        "Diesel S10 para frota",
    ],
    "metalurgia": [
        "Luva nitrilica industrial", "Oculos de protecao ampla visao",
        "Disco abrasivo corte metal", "Eletrodo solda revestido",
        "Graxa alta temperatura",
    ],
    "energia": [
        "Capacete de seguranca classe B", "Luva nitrilica industrial",
        "Oleo lubrificante hidraulico", "Furadeira de impacto industrial",
    ],
    "agroindustria": [
        "Bota de seguranca CA", "Luva nitrilica industrial",
        "Oleo lubrificante hidraulico", "Graxa alta temperatura",
    ],
    "logistica": [
        "Diesel S10 para frota", "Oleo lubrificante hidraulico",
        "Bota de seguranca CA", "Furadeira de impacto industrial",
    ],
    "saneamento": [
        "Luva nitrilica industrial", "Bota de seguranca CA",
        "Oculos de protecao ampla visao", "Graxa alta temperatura",
    ],
    "manufatura pesada": [
        "Disco abrasivo corte metal", "Eletrodo solda revestido",
        "Graxa alta temperatura", "Oculos de protecao ampla visao",
    ],
}

SELLER_SPECIALTIES = [
    ("EPI e seguranca", "mineracao"),
    ("ferramentas e abrasivos", "construcao de vias"),
    ("solda e metalmecanica", "metalurgia"),
    ("lubrificantes e graxas", "manufatura pesada"),
    ("combustiveis e frota", "logistica"),
    ("manutencao industrial", "energia"),
    ("campo e saneamento", "saneamento"),
    ("agroindustria", "agroindustria"),
]


def stable_index(value: str, modulo: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def backup_raw_files(data_dir: str) -> None:
    backup_dir = os.path.join(data_dir, "original_maven")
    os.makedirs(backup_dir, exist_ok=True)

    for filename in ["accounts.csv", "products.csv", "sales_pipeline.csv", "sales_teams.csv"]:
        src = os.path.join(data_dir, filename)
        dst = os.path.join(backup_dir, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def build_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    accounts = accounts.copy()

    sector_by_account = {}
    segment_by_account = {}
    for account in accounts["account"]:
        sector = INDUSTRIAL_SECTORS[stable_index(account, len(INDUSTRIAL_SECTORS))]
        sector_by_account[account] = sector
        segment_by_account[account] = SEGMENTS_BY_SECTOR[sector]

    accounts["sector"] = accounts["account"].map(sector_by_account)
    accounts["customer_segment"] = accounts["account"].map(segment_by_account)
    return accounts


def build_products() -> pd.DataFrame:
    return PRODUCTS.copy()


def build_pipeline(pipeline: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    pipeline = pipeline.copy()
    account_sector = accounts.set_index("account")["sector"].to_dict()

    def choose_product(row: pd.Series) -> str:
        sector = account_sector.get(row["account"])
        options = PRODUCT_AFFINITY.get(sector, PRODUCTS["product"].tolist())
        key = f"{row['opportunity_id']}:{row['sales_agent']}:{row['account']}"
        return options[stable_index(key, len(options))]

    has_account = pipeline["account"].notna()
    pipeline.loc[has_account, "product"] = pipeline.loc[has_account].apply(choose_product, axis=1)
    pipeline.loc[~has_account, "product"] = pipeline.loc[~has_account, "product"].map(
        lambda value: PRODUCTS["product"].iloc[stable_index(str(value), len(PRODUCTS))]
    )
    return pipeline


def build_teams(teams: pd.DataFrame) -> pd.DataFrame:
    teams = teams.copy()

    specialties = []
    focus_sectors = []
    for agent in teams["sales_agent"]:
        specialty, sector = SELLER_SPECIALTIES[stable_index(agent, len(SELLER_SPECIALTIES))]
        specialties.append(specialty)
        focus_sectors.append(sector)

    teams["seller_specialty"] = specialties
    teams["seller_focus_sector"] = focus_sectors
    return teams


def run(base_dir: str) -> None:
    data_dir = os.path.join(base_dir, "data")
    backup_raw_files(data_dir)

    accounts = pd.read_csv(os.path.join(data_dir, "original_maven", "accounts.csv"))
    pipeline = pd.read_csv(os.path.join(data_dir, "original_maven", "sales_pipeline.csv"))
    teams = pd.read_csv(os.path.join(data_dir, "original_maven", "sales_teams.csv"))

    industrial_accounts = build_accounts(accounts)
    industrial_products = build_products()
    industrial_pipeline = build_pipeline(pipeline, industrial_accounts)
    industrial_teams = build_teams(teams)

    industrial_accounts.to_csv(os.path.join(data_dir, "accounts.csv"), index=False)
    industrial_products.to_csv(os.path.join(data_dir, "products.csv"), index=False)
    industrial_pipeline.to_csv(os.path.join(data_dir, "sales_pipeline.csv"), index=False)
    industrial_teams.to_csv(os.path.join(data_dir, "sales_teams.csv"), index=False)


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(base)
