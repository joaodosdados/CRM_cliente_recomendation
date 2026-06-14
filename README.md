# CRM Industrial - Lead Scoring e Recomendacao Comercial

MVP de Data Science para priorizacao de clientes em vendas B2B industriais.
O projeto simula um distribuidor de produtos industriais, como EPI,
ferramentas, abrasivos, solda, lubrificantes, graxas e combustiveis.

O objetivo e ajudar o vendedor a responder duas perguntas operacionais:

1. **Qual cliente devo abordar primeiro na minha carteira?**
2. **Quais clientes da urna valem o esforco comercial agora?**

O projeto usa Python, scikit-learn e Streamlit. O treino e offline; a interface
carrega artefatos treinados e faz scoring em tempo de uso.

## Cenarios

### 1. Carteira por vendedor

Para um vendedor selecionado, o sistema ranqueia clientes considerando:

- probabilidade de fechamento (`match_score`);
- potencial estimado de receita (`lead_score`);
- afinidade vendedor x setor;
- afinidade vendedor x categoria de produto;
- recencia do relacionamento;
- score de custo-beneficio;
- esforco comercial estimado.

O ranking final usa `score_comercial`, em escala 0-100.

### 2. Urna de clientes

Para uma lista geral de clientes, o sistema ranqueia oportunidades por:

- valor futuro esperado;
- score do cliente independente do vendedor;
- custo-beneficio;
- historico de conversao;
- recencia;
- temperatura comercial: `quente`, `morno`, `frio`.

O ranking da urna usa `score_prioridade`, em escala 0-100.

## Dataset

A base original vem do dataset publico **CRM Sales Opportunities**, usado como
estrutura B2B com contas, vendedores, produtos e oportunidades.

Como nao ha uma base publica completa com vendas industriais, vendedores,
clientes, historico de compra e categorias como EPI/ferramentas/lubrificantes,
o projeto aplica uma camada de simulacao industrial:

- setores: `mineracao`, `construcao de vias`, `metalurgia`, `energia`,
  `agroindustria`, `logistica`, `saneamento`, `manufatura pesada`;
- produtos: EPI, ferramentas, abrasivos, solda, lubrificantes e combustiveis;
- vendedores: especialidade e setor foco;
- clientes: segmento industrial;
- oportunidades: produtos reatribuidos por afinidade setor-categoria.

Os arquivos originais ficam em `data/original_maven/`.
Os arquivos ativos industrializados ficam em `data/`.

## Estrutura

```text
.
|-- app.py                         # Dashboard Streamlit
|-- data/
|   |-- original_maven/            # Backup da base publica original
|   |-- processed/                 # Feature store gerada pelo ETL
|   |-- accounts.csv               # Clientes industrializados
|   |-- products.csv               # Catalogo industrial
|   |-- sales_pipeline.csv         # Oportunidades industrializadas
|   `-- sales_teams.csv            # Vendedores com especialidade
|-- docs/
|   `-- ARCHITECTURE.md            # Decisoes tecnicas e fluxo
|-- models/
|   |-- lead_value_model.joblib
|   `-- seller_match_model.joblib
|-- src/
|   |-- industrialize_dataset.py   # Converte a base original para dominio industrial
|   |-- data_prep.py               # ETL e engenharia de features
|   |-- train_seller_match.py      # Modelo vendedor x cliente
|   |-- train_lead_scoring.py      # Modelo de valor futuro
|   `-- inference.py               # Scoring, ranking e explicabilidade
|-- AGENTS.md                      # Contexto rapido para LLMs/agentes
|-- README.md
`-- requirements.txt
```

## Como rodar

Crie/ative um ambiente Python e instale dependencias:

```bash
pip install -r requirements.txt
```

Execute o pipeline completo:

```bash
python src/industrialize_dataset.py
python src/data_prep.py
python src/train_seller_match.py
python src/train_lead_scoring.py
streamlit run app.py
```

Com a `.venv` local usada neste projeto:

```bash
./.venv/bin/python src/industrialize_dataset.py
./.venv/bin/python src/data_prep.py
./.venv/bin/python src/train_seller_match.py
./.venv/bin/python src/train_lead_scoring.py
./.venv/bin/streamlit run app.py --server.port 8620
```

## Artefatos gerados

`src/data_prep.py` gera:

- `data/processed/accounts_features.csv`
- `data/processed/opportunity_table.csv`
- `data/processed/seller_profiles.csv`
- `data/processed/seller_sector_winrate.csv`
- `data/processed/lead_scoring_table.csv`
- `data/processed/urna_features.csv`
- `data/processed/global_stats.json`

Os treinos geram:

- `models/seller_match_model.joblib`
- `models/lead_value_model.joblib`

## Modelos

### Match vendedor x cliente

- Algoritmo: `LogisticRegression`
- Target: `target_won`
- Features: setor, segmento, perfil do vendedor, historico do vendedor,
  win rate vendedor-setor e firmograficos do cliente.
- Saida: `match_score`, probabilidade estimada de ganho.

### Valor futuro do cliente

- Algoritmo: `RandomForestRegressor`
- Target: `future_value`
- Split temporal: historico antes de `2017-07-01` prediz receita ganha depois.
- Saida: `lead_score`, valor futuro esperado.

## Scores comerciais

O projeto nao usa apenas o output bruto dos modelos. A inferencia cria scores
mais proximos de decisao comercial:

```text
score_comercial =
    probabilidade_de_fechamento
    * potencial_estimado
    * afinidade_vendedor_setor
    * afinidade_vendedor_categoria
    * recencia
    * custo_beneficio
```

Tambem sao calculados:

- `score_cliente`: qualidade do cliente independente do vendedor;
- `score_custo_beneficio`: retorno esperado ajustado pelo esforco;
- `nivel_esforco`: baixo, medio ou alto;
- `motivo`: explicacao curta do ranking;
- `proxima_oferta`: gancho comercial sugerido.

## Dashboard

O `app.py` mostra um cockpit comercial compacto:

- filtros por vendedor, setor e score minimo;
- KPIs do vendedor;
- lista priorizada de clientes;
- painel do cliente selecionado;
- motivo da recomendacao;
- score comercial, score do cliente e custo-beneficio;
- historico de oportunidades;
- categoria dominante;
- gancho comercial para a ligacao.

## Validacao rapida

```bash
python -m py_compile app.py src/*.py
python src/inference.py
```

## Limitacoes

- A base industrial e simulada a partir de um dataset publico B2B generico.
- Os scores de custo-beneficio e esforco sao heuristicas de MVP, nao modelos
  calibrados com dados reais de tempo comercial.
- Com apenas 85 contas, metricas de regressao podem oscilar bastante.
- Para producao real, o ideal e incluir margem, estoque, SLA, carteira atual,
  visitas comerciais, churn e dados transacionais por categoria.

## Proximos passos recomendados

- Adicionar margem por categoria e ranquear por lucro esperado.
- Criar aba "Plano do Dia" com as melhores ligacoes.
- Modelar churn/reativacao de clientes inativos.
- Gerar recomendacao de proximo produto por cliente.
- Separar regras comerciais de `inference.py` em modulo dedicado.
