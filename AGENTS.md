# Contexto para agentes e LLMs

Este repositorio e um MVP de Data Science para priorizacao comercial em vendas
B2B industriais. Use este arquivo para recuperar contexto rapidamente antes de
fazer alteracoes.

## Objetivo do produto

Construir um cockpit para vendedor decidir:

- quais clientes abordar primeiro na carteira;
- quais clientes da urna sao mais promissores;
- qual motivo justifica a prioridade;
- qual gancho comercial usar na ligacao.

O dominio simulado e venda industrial: EPI, ferramentas, abrasivos, solda,
lubrificantes, graxas e combustiveis.

## Fluxo de dados

1. `src/industrialize_dataset.py`
   - preserva a base original em `data/original_maven/`;
   - reescreve os CSVs ativos em `data/` com dominio industrial.

2. `src/data_prep.py`
   - gera feature store em `data/processed/`;
   - monta `opportunity_table.csv`, `seller_profiles.csv`,
     `seller_sector_winrate.csv`, `lead_scoring_table.csv` e `urna_features.csv`.

3. `src/train_seller_match.py`
   - treina `models/seller_match_model.joblib`;
   - modelo de classificacao para probabilidade de ganho.

4. `src/train_lead_scoring.py`
   - treina `models/lead_value_model.joblib`;
   - modelo de regressao para valor futuro esperado.

5. `src/inference.py`
   - carrega dados/modelos;
   - calcula `match_score`, `lead_score`, `score_comercial`,
     `score_cliente`, `score_custo_beneficio`, `score_prioridade`,
     `nivel_esforco`, `motivo` e `proxima_oferta`.

6. `app.py`
   - dashboard Streamlit operacional.

## Comandos principais

```bash
./.venv/bin/python src/industrialize_dataset.py
./.venv/bin/python src/data_prep.py
./.venv/bin/python src/train_seller_match.py
./.venv/bin/python src/train_lead_scoring.py
./.venv/bin/python src/inference.py
./.venv/bin/streamlit run app.py --server.port 8620
```

## Regras importantes

- Nao versionar `.venv/`, `__pycache__/` ou `*.pyc`.
- Se alterar schema de `data/processed/`, atualize `src/inference.py` e `app.py`.
- Se alterar features de treino, regenere `data/processed/` e `models/`.
- O app usa `st.cache_resource`; quando mudar estrutura de artefatos, atualize o
  argumento de versao em `get_artifacts(...)`.
- Dados industrializados sao simulados; deixar isso claro em documentacao.

## Conceitos do score

- `match_score`: probabilidade estatistica de ganho.
- `lead_score`: valor futuro esperado.
- `score_cliente`: qualidade do cliente independente do vendedor.
- `score_custo_beneficio`: retorno esperado ajustado por esforco.
- `score_comercial`: ranking final para vendedor.
- `score_prioridade`: ranking final da urna.
- `motivo`: explicacao curta.
- `proxima_oferta`: sugestao de abordagem comercial.

## Checks antes de finalizar

```bash
./.venv/bin/python -m py_compile app.py src/*.py
./.venv/bin/python src/inference.py
```
