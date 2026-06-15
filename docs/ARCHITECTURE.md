# Arquitetura do MVP

## Visao geral

O projeto segue um fluxo simples e reproduzivel:

```text
dados brutos/simulados
  -> ETL e feature engineering
  -> treino offline
  -> artefatos versionados
  -> inferencia em tempo de uso
  -> dashboard Streamlit
```

O objetivo e priorizar clientes para venda B2B industrial, mantendo o codigo
curto o suficiente para ser entendido em exercicio de Data Science.

## Tabelas de entrada

Arquivos ativos em `data/`:

- `accounts.csv`: uma linha por cliente.
- `products.csv`: catalogo industrial com produto, categoria e preco.
- `sales_pipeline.csv`: oportunidades de venda.
- `sales_teams.csv`: vendedores, gestores, regiao, especialidade e setor foco.

Arquivos originais em `data/original_maven/` preservam a base publica antes da
camada de simulacao industrial.

## Feature store

Gerada por `src/data_prep.py` em `data/processed/`.

### `accounts_features.csv`

Features firmograficas e segmento industrial:

- `sector`
- `customer_segment`
- `company_age`
- `revenue`
- `employees`
- `is_subsidiary`
- `is_domestic`

### `opportunity_table.csv`

Base de oportunidades fechadas (`Won` e `Lost`) com:

- dados da oportunidade;
- produto e categoria (`series`);
- target `target_won`;
- features do cliente;
- perfil do vendedor.

### `seller_profiles.csv`

Agregados por vendedor:

- total de deals;
- win rate historico;
- ticket medio ganho;
- especialidade;
- setor foco.

### `seller_sector_winrate.csv`

Win rate vendedor-setor com suavizacao bayesiana.

### `lead_scoring_table.csv`

Tabela de treino temporal para estimar valor futuro:

- features historicas antes de `2017-07-01`;
- alvo `future_value` depois de `2017-07-01`.

### `urna_features.csv`

Snapshot atual da urna usando todo historico disponivel ate `2018-01-01`.

## Modelos

### `seller_match_model.joblib`

Pipeline scikit-learn:

- `ColumnTransformer`
- `OneHotEncoder` para categoricas
- `StandardScaler` para numericas
- `LogisticRegression`

Saida usada: `predict_proba(X)[:, 1]`.

### `lead_value_model.joblib`

Pipeline scikit-learn:

- `ColumnTransformer`
- `OneHotEncoder` para categoricas
- passthrough para numericas
- `RandomForestRegressor`

Saida usada: `predict(X)`.

## Camada de inferencia comercial

`src/inference.py` transforma outputs de modelo em sinais comerciais:

- `match_score`: probabilidade de fechamento;
- `lead_score`: valor futuro esperado;
- `score_cliente`: atratividade do cliente;
- `score_custo_beneficio`: retorno ajustado por esforco;
- `score_comercial`: ranking final da carteira;
- `score_prioridade`: ranking final da urna;
- `nivel_esforco`: baixo, medio, alto;
- `motivo`: explicacao de ate tres fatores;
- `proxima_oferta`: gancho comercial sugerido.

Essa camada mistura modelos e heuristicas de negocio. Em producao, as
heuristicas deveriam ser calibradas com dados reais.

## Dashboard

`app.py` e uma interface Streamlit orientada a decisao:

- aba "Carteira por vendedor";
- aba "Urna de clientes";
- filtros compactos;
- ranking em tabela;
- painel do cliente selecionado;
- historico, score, custo-beneficio, esforco e abordagem sugerida.

## Dependencias principais

- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- plotly

## Pontos de extensao

- `src/scoring_rules.py`: futuro modulo para remover heuristicas de
  `src/inference.py`.
- `src/product_recommendation.py`: recomendacao de proximo produto.
- `src/churn.py`: risco de churn ou reativacao.
- `tests/`: testes unitarios para schema e inferencia.
