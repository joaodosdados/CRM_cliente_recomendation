"""
Cenario 1 - Match Vendedor x Cliente.

Treina um classificador que estima a probabilidade de fechamento (Won) de uma
oportunidade dado o par (perfil do vendedor, perfil firmografico do cliente).
Em producao, fixando o vendedor e variando o cliente, o modelo gera um score
de "fit" por cliente -> ranking de recomendacao para aquele vendedor.

Saida: models/seller_match_model.joblib
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["sector", "regional_office", "manager"]
NUMERIC = [
    "company_age", "revenue", "employees", "is_subsidiary", "is_domestic",
    "agent_win_rate", "agent_avg_deal_value", "agent_total_deals", "agent_sector_winrate",
]
TARGET = "target_won"


def build_training_table(processed_dir: str) -> pd.DataFrame:
    opp = pd.read_csv(os.path.join(processed_dir, "opportunity_table.csv"))
    profiles = pd.read_csv(os.path.join(processed_dir, "seller_profiles.csv"))
    sector_wr = pd.read_csv(os.path.join(processed_dir, "seller_sector_winrate.csv"))

    df = opp.merge(
        profiles[["sales_agent", "agent_win_rate", "agent_avg_deal_value", "agent_total_deals"]],
        on="sales_agent", how="left",
    )
    df = df.merge(sector_wr, on=["sales_agent", "sector"], how="left")
    return df


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    model = LogisticRegression(max_iter=2000)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def run(processed_dir: str, models_dir: str) -> None:
    df = build_training_table(processed_dir)

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cv_auc = cross_val_score(build_pipeline(), X, y, cv=5, scoring="roc_auc")

    print("AUC (holdout):", round(roc_auc_score(y_test, proba), 4))
    print("AUC (5-fold CV): %.4f +/- %.4f" % (cv_auc.mean(), cv_auc.std()))
    print(classification_report(y_test, pred, target_names=["Lost", "Won"]))

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, "seller_match_model.joblib"))
    print("Modelo salvo em", os.path.join(models_dir, "seller_match_model.joblib"))


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(os.path.join(base, "data", "processed"), os.path.join(base, "models"))
