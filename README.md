# CRM Lead Scoring & Recomendacao de Clientes (MVP)

MVP de Data Science que simula dois cenarios de priorizacao de clientes em
um CRM de vendas B2B:

1. **Match Vendedor x Cliente** - dado o perfil de um vendedor, ranqueia os
   clientes existentes pela probabilidade de fechamento (Won) com aquele
   vendedor.
2. **Ranking da Urna** - dado um conjunto ("urna") de clientes, ranqueia
   pelo valor futuro esperado e classifica em quente / morno / frio, para o
   time de vendas priorizar onde investir tempo.

## Dataset

[CRM Sales Opportunities](https://github.com/ikebude/CRM-Sales-Analysis)
(Maven Analytics), simulando a empresa ficticia MavenTech (vendas B2B de
hardware). Contem:

- `accounts.csv` - 85 contas/clientes (setor, faturamento, funcionarios, etc.)
- `sales_pipeline.csv` - ~8.800 oportunidades de venda (Won/Lost/Engaging/Prospecting)
- `sales_teams.csv` - 35 vendedores, gestores e escritorios regionais
- `products.csv` - catalogo de produtos

## Arquitetura

```
project/
├── data/
│   ├── accounts.csv, products.csv, sales_pipeline.csv, sales_teams.csv
│   └── processed/            # gerado por data_prep.py
├── src/
│   ├── data_prep.py          # ETL e engenharia de features
│   ├── train_seller_match.py # treino do modelo do Cenario 1
│   ├── train_lead_scoring.py # treino do modelo do Cenario 2
│   └── inference.py          # scoring em tempo real (sem re-treino)
├── models/                    # modelos treinados (.joblib)
├── app.py                     # interface Streamlit
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt

# 1. ETL e engenharia de features
python3 src/data_prep.py

# 2. Treino dos modelos
python3 src/train_seller_match.py
python3 src/train_lead_scoring.py

# 3. Interface
streamlit run app.py
```

Os modelos treinados ficam salvos em `models/` e os datasets processados em
`data/processed/`. O `app.py` carrega esses artefatos uma unica vez
(`st.cache_resource`) e faz apenas inferencia - nao ha re-treino em tempo de
uso.

## Cenario 1 - Match Vendedor x Cliente

### Features

- **Do cliente (firmograficas):** setor, idade da empresa, faturamento,
  numero de funcionarios, se e subsidiaria, se e domestico (US).
- **Do vendedor:** escritorio regional, gestor, win rate historico, ticket
  medio em deals ganhos, volume total de deals.
- **Cruzamento vendedor x setor:** taxa de conversao do vendedor naquele
  setor especifico, com suavizacao bayesiana (peso k=5 em direcao ao win
  rate geral do vendedor) para lidar com combinacoes com poucos dados.

### Modelo

Regressao Logistica com pre-processamento (`OneHotEncoder` para variaveis
categoricas + `StandardScaler` para numericas), escolhida apos comparacao
com Random Forest e Gradient Boosting.

### Metricas

- AUC (holdout): **0.6155**
- AUC (5-fold CV): **0.6083 ± 0.0156**

### Output

Para um vendedor selecionado, a tabela de `accounts_features` recebe as
colunas de perfil daquele vendedor, o modelo gera `match_score`
(probabilidade de Won) para cada cliente, e o ranking e ordenado por esse
score.

## Cenario 2 - Ranking da Urna (Lead Scoring)

### Desenho do split temporal

Para gerar sinal preditivo real (e nao apenas correlacionar firmograficos
estaticos com resultado), as oportunidades sao divididas por
`engage_date`:

- **Janela historica** (`engage_date < 2017-07-01`): usada para calcular
  features RFM por conta - `total_opps_hist`, `win_rate_hist`,
  `total_value_won_hist`, `avg_deal_value_hist`, `recency_days_hist`.
- **Janela futura** (`engage_date >= 2017-07-01`): soma de `close_value` dos
  deals Won, usada como alvo `future_value`.

Essa divisao temporal faz sentido com o caso de uso real: hoje, so temos o
historico do cliente; o modelo aprende a relacao entre "como o cliente se
comportou no passado" e "quanto valor ele gerou depois". A correlacao entre
`total_value_won` na janela historica e `future_value` e de **0.53**,
confirmando que ha sinal real nessa relacao.

Para a urna em producao (`urna_features.csv`), as mesmas features RFM sao
recalculadas usando todo o historico disponivel, com data de referencia
`2018-01-01`.

### Modelo

Random Forest Regressor (`n_estimators=300`, `max_depth=3`), com
pre-processamento (`OneHotEncoder` para setor + passthrough para
numericas), escolhido apos comparacao com Ridge, Gradient Boosting e Extra
Trees (com e sem log-transform do alvo), validado com Repeated K-Fold
(10x5-fold).

### Metricas

- R2 (5-fold CV x10): **0.167 ± 0.286**
- R2 (holdout unico): -0.484 (instavel dado n=85 contas; CV repetido e a
  metrica mais confiavel aqui)
- MAE (holdout): 17.591,90
- Features mais importantes: `revenue` (0.360), `total_opps_hist` (0.261),
  `employees` (0.089), `total_value_won_hist` (0.085), `company_age`
  (0.077), `win_rate_hist` (0.070)

### Output

O modelo prediz `lead_score` (valor futuro esperado) para cada conta da
urna. As contas sao divididas em tercis e classificadas como **quente**,
**morno** ou **frio**, permitindo ao vendedor focar nos clientes com maior
potencial.

## Limitacoes conhecidas (transparencia do MVP)

- O dataset e **fictício** e tem sinal preditivo limitado por design:
  features firmograficas isoladas (setor, faturamento, funcionarios) tem
  correlacao fraca com o resultado (win rate varia entre 0.61-0.66 vs media
  global de 63,15% em quase todos os cortes).
- Com apenas 85 contas, metricas de holdout unico (especialmente R2 do
  Cenario 2) sao instaveis; a validacao cruzada repetida e mais
  representativa.
- O foco do MVP e a **arquitetura do pipeline** (ETL -> features -> treino
  -> artefatos -> inferencia em tempo real -> interface), que e
  diretamente reaproveitavel com um dataset real de CRM com mais volume e
  sinal mais forte.

## Reuso com dados reais

Para adaptar a um CRM real, basta:

1. Mapear as colunas de origem para o schema esperado por
   `build_accounts_features`, `build_opportunity_table`,
   `build_seller_profiles` em `src/data_prep.py`.
2. Re-executar o pipeline (`data_prep.py` -> `train_*.py`).
3. O `inference.py` e o `app.py` nao precisam de alteracao, desde que o
   schema das tabelas processadas seja mantido.
"# CRM_cliente_recomendation" 
