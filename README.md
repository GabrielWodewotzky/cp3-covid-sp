# 🦠 COVID-19 nos municípios do Estado de São Paulo — Dashboard (Checkpoint 3)

Disciplina **Data Analytics em SP**.

**Integrantes:** Gabriel Simioni (RM563475) · Guilherme Vega (RM562655) · Davi Xavier (RM563572) ·
Gabriel Pereira (RM563571) · Luiz Henrique (RM563795) · Felipe Ramalho (RM565073)

🔗 **Aplicação online:** https://SEU-APP.streamlit.app  <!-- TODO: preencher após o deploy -->

## Problema
A pandemia de COVID-19 afetou de forma muito desigual os municípios paulistas. O dashboard permite explorar
a evolução temporal, comparar municípios com indicadores normalizados pela população e testar um modelo
de previsão semanal de casos e óbitos.

## Fonte dos dados
Fundação SEADE (SES-SP, SIVEP-Gripe, SIMI-SP) — https://github.com/seade-R/dados-covid-sp
(série diária por município, 25/02/2020 a 18/11/2023). O app baixa o CSV automaticamente; se existir
`data/dados_covid_sp.csv`, usa a cópia local.

## O que o dashboard contém
- **KPIs:** casos e óbitos acumulados, incidência, mortalidade e letalidade.
- **Gráficos interativos (Plotly):** séries diárias com média móvel de 7 dias, ranking de municípios,
  dispersão população × letalidade, previsão × observado.
- **Gráfico estático (Matplotlib):** distribuição da população dos municípios.
- **Filtros:** município, período, população mínima, indicador e tamanho do ranking.
- **Análise preditiva:** Ridge / Random Forest / Gradient Boosting com split temporal, métricas
  (MAE, RMSE, R², WAPE) e comparação com baseline.

## Estrutura
```
app.py            # dashboard Streamlit
modelo.py         # módulo de previsão (scikit-learn)
requirements.txt  # dependências
```

## Executar localmente
```bash
git clone https://github.com/SEU-USUARIO/SEU-REPO.git
cd SEU-REPO
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Abra http://localhost:8501.

## Publicação (Streamlit Community Cloud)
1. Suba o projeto para um repositório público no GitHub.
2. Em https://share.streamlit.io, faça login com o GitHub e clique em **Create app**.
3. Selecione o repositório, a branch e o arquivo principal `app.py`, e clique em **Deploy**.
