"""
Cenario 2 - Ranking da Urna (Lead Scoring).

Treina um modelo de regressao que estima o "valor futuro esperado" de uma conta
(receita gerada em fechamentos Won no periodo seguinte) a partir de:
  - features firmograficas (setor, porte, idade da empresa, etc.)
  - features RFM do historico recente (frequencia de oportunidades, taxa de
    conversao, valor total ja ganho, ticket medio, recencia do ultimo contato)

A tabela de treino e construida com um corte temporal: as features RFM sao
calculadas ANTES do corte e o alvo (future_value) e a receita gerada DEPOIS do
corte -> validacao honesta de que o historico passado prediz o potencial futuro.

Em producao, aplica-se o mesmo modelo sobre o snapshot atual da urna
(data/processed/urna_features.csv) para gerar o lead_score de cada conta.

Saida: models/lead_value_model.joblib
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["sector"]
NUMERIC = [
    "company_age", "revenue", "employees", "is_subsidiary", "is_domestic",
    "total_opps_hist", "win_rate_hist", "total_value_won_hist",
    "avg_deal_value_hist", "recency_days_hist",
]
TARGET = "future_value"


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    model = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=42)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def run(processed_dir: str, models_dir: str) -> None:
    df = pd.read_csv(os.path.join(processed_dir, "lead_scoring_table.csv"))

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print("R2 (holdout):", round(r2_score(y_test, pred), 3))
    print("MAE (holdout):", round(mean_absolute_error(y_test, pred), 2))

    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv_r2 = cross_val_score(build_pipeline(), X, y, cv=rkf, scoring="r2")
    print("R2 (5-fold CV x10): %.3f +/- %.3f" % (cv_r2.mean(), cv_r2.std()))

    # refit com todos os dados disponiveis para uso em producao
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    importances = final_pipeline.named_steps["model"].feature_importances_
    feature_names = (
        list(final_pipeline.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
        + NUMERIC
    )
    top = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:8]
    print("\nTop features:")
    for name, imp in top:
        print("  %-25s %.3f" % (name, imp))

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(final_pipeline, os.path.join(models_dir, "lead_value_model.joblib"))
    print("\nModelo salvo em", os.path.join(models_dir, "lead_value_model.joblib"))


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(os.path.join(base, "data", "processed"), os.path.join(base, "models"))
