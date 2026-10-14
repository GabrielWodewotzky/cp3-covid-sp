"""Dashboard - COVID-19 nos municípios do Estado de São Paulo (Checkpoint 3)
Disciplina: Data Analytics em SP | Fonte: Fundação SEADE (SES-SP, SIVEP-Gripe, SIMI-SP)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import modelo as ml

URL = "https://raw.githubusercontent.com/seade-R/dados-covid-sp/master/data/dados_covid_sp.csv"
LOCAL = os.path.join("data", "dados_covid_sp.csv")  # opcional: cópia local evita depender da internet
ESTADO = "Estado de São Paulo (todos os municípios)"

st.set_page_config(page_title="COVID-19 nos municípios de SP", page_icon="🦠", layout="wide")


# ------------------------------------------------------------------ dados
@st.cache_data(show_spinner="Carregando dados da Fundação SEADE...")
def carregar() -> pd.DataFrame:
    df = pd.read_csv(LOCAL if os.path.exists(LOCAL) else URL, sep=";", decimal=",")
    df["datahora"] = pd.to_datetime(df["datahora"])
    # Valores negativos são correções da própria série (ver CP1): truncamos em zero
    for c in ["casos", "obitos", "casos_novos", "obitos_novos"]:
        df[c] = df[c].clip(lower=0)
    df["nome_munic"] = df["nome_munic"].astype("category")
    # Código >= 9.000.000 = registros sem município identificado (o "646º nome" do CP1)
    df["ident"] = df["codigo_ibge"] < 9_000_000
    return df


@st.cache_data
def serie_escopo(_df: pd.DataFrame, municipio: str) -> pd.DataFrame:
    d = _df if municipio == ESTADO else _df[_df["nome_munic"] == municipio]
    s = d.groupby("datahora", observed=True)[["casos", "obitos", "casos_novos", "obitos_novos", "pop"]].sum()
    s["casos_mm7d"] = s["casos_novos"].rolling(7).mean()
    s["obitos_mm7d"] = s["obitos_novos"].rolling(7).mean()
    s["letalidade_pct"] = np.where(s["casos"] > 0, s["obitos"] / s["casos"] * 100, np.nan)
    return s


def snapshot(_df: pd.DataFrame, data: pd.Timestamp) -> pd.DataFrame:
    s = _df[(_df["datahora"] == data) & _df["ident"] & (_df["pop"] > 0)].copy()
    s["nome_munic"] = s["nome_munic"].astype(str)
    s["incidencia_100k"] = s["casos"] / s["pop"] * 100_000
    s["mortalidade_100k"] = s["obitos"] / s["pop"] * 100_000
    s["letalidade_pct"] = np.where(s["casos"] > 0, s["obitos"] / s["casos"] * 100, np.nan)
    return s


df = carregar()
DATA_MIN, DATA_MAX = df["datahora"].min().date(), df["datahora"].max().date()

# ------------------------------------------------------------------ filtros
st.sidebar.title("Filtros")
municipios = [ESTADO] + sorted(df.loc[df["ident"], "nome_munic"].astype(str).unique())
escopo = st.sidebar.selectbox("Município", municipios, help="Escolha o Estado inteiro ou um município.")
periodo = st.sidebar.date_input("Período", (DATA_MIN, DATA_MAX), min_value=DATA_MIN, max_value=DATA_MAX)
if not isinstance(periodo, (tuple, list)) or len(periodo) != 2:
    st.info("Selecione a data inicial e a final no filtro de período.")
    st.stop()
ini, fim = pd.Timestamp(periodo[0]), pd.Timestamp(periodo[1])

st.sidebar.markdown("---")
st.sidebar.caption("Filtros da aba **Comparação entre municípios**")
pop_min = st.sidebar.slider("População mínima", 0, 200_000, 50_000, 5_000)
indicador = st.sidebar.selectbox(
    "Indicador do ranking",
    ["mortalidade_100k", "incidencia_100k", "letalidade_pct"],
    format_func=lambda x: {"mortalidade_100k": "Mortalidade (óbitos/100 mil hab.)",
                           "incidencia_100k": "Incidência (casos/100 mil hab.)",
                           "letalidade_pct": "Letalidade (% óbitos/casos)"}[x],
)
top_n = st.sidebar.slider("Quantidade no ranking", 5, 30, 15)

serie = serie_escopo(df, escopo)
serie_per = serie.loc[ini:fim]
if serie_per.empty:
    st.warning("Sem dados no período selecionado.")
    st.stop()

# ------------------------------------------------------------------ cabeçalho
st.title("🦠 COVID-19 nos municípios do Estado de São Paulo")
st.caption("Data Analytics em SP · Checkpoint 3 · Grupo: Gabriel Simioni (RM563475), Guilherme Vega (RM562655), "
           "Davi Xavier (RM563572), Gabriel Pereira (RM563571), Luiz Henrique (RM563795), Felipe Ramalho (RM565073)")
st.markdown(f"Visualizando: **{escopo}** · de **{ini:%d/%m/%Y}** a **{fim:%d/%m/%Y}**")

tab_geral, tab_comp, tab_prev, tab_sobre = st.tabs(
    ["📈 Visão geral", "🗺️ Comparação entre municípios", "🔮 Previsão", "ℹ️ Fonte e metodologia"])

# ------------------------------------------------------------------ visão geral
with tab_geral:
    ult = serie_per.iloc[-1]
    casos_ac, obitos_ac, pop = ult["casos"], ult["obitos"], ult["pop"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Casos acumulados", f"{casos_ac:,.0f}".replace(",", "."))
    c2.metric("Óbitos acumulados", f"{obitos_ac:,.0f}".replace(",", "."))
    c3.metric("População", f"{pop:,.0f}".replace(",", "."))
    c4, c5, c6 = st.columns(3)
    c4.metric("Incidência (casos/100 mil hab.)", f"{casos_ac / pop * 1e5:,.0f}".replace(",", ".") if pop else "—")
    c5.metric("Mortalidade (óbitos/100 mil hab.)", f"{obitos_ac / pop * 1e5:,.0f}".replace(",", ".") if pop else "—")
    c6.metric("Letalidade", f"{obitos_ac / casos_ac * 100:.2f}%" if casos_ac else "—")
    st.caption(f"Indicadores acumulados até {serie_per.index[-1]:%d/%m/%Y}. "
               f"Casos novos no período: {serie_per['casos_novos'].sum():,.0f} · "
               f"Óbitos novos no período: {serie_per['obitos_novos'].sum():,.0f}".replace(",", "."))

    for col, mm, cor, nome in [("casos_novos", "casos_mm7d", "steelblue", "Casos novos"),
                               ("obitos_novos", "obitos_mm7d", "indianred", "Óbitos novos")]:
        fig = go.Figure()
        fig.add_bar(x=serie_per.index, y=serie_per[col], name=f"{nome} (diário)", marker_color=cor, opacity=0.3)
        fig.add_scatter(x=serie_per.index, y=serie_per[mm], name="Média móvel 7 dias", line=dict(color=cor, width=3))
        fig.update_layout(title=f"{nome} por dia — {escopo}", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
        st.plotly_chart(fig)

    fig = px.line(serie_per.reset_index(), x="datahora", y="letalidade_pct",
                  title="Letalidade acumulada (%) ao longo do tempo",
                  labels={"datahora": "Data", "letalidade_pct": "Letalidade (%)"})
    fig.update_traces(line_color="darkorange")
    st.plotly_chart(fig)

# ------------------------------------------------------------------ comparação
with tab_comp:
    snap = snapshot(df, fim)
    base = snap[snap["pop"] >= pop_min]
    st.markdown(f"Foto em **{fim:%d/%m/%Y}** · municípios com população ≥ **{pop_min:,}** "
                f"({len(base)} de {len(snap)})".replace(",", "."))
    nomes = {"mortalidade_100k": "Mortalidade (óbitos/100 mil hab.)",
             "incidencia_100k": "Incidência (casos/100 mil hab.)",
             "letalidade_pct": "Letalidade (%)"}
    if base.empty:
        st.warning("Nenhum município atende ao filtro de população.")
    else:
        top = base.nlargest(top_n, indicador).sort_values(indicador)
        fig = px.bar(top, x=indicador, y="nome_munic", orientation="h", color=indicador,
                     color_continuous_scale="Reds", hover_data=["pop", "casos", "obitos"],
                     labels={indicador: nomes[indicador], "nome_munic": ""},
                     title=f"Top {top_n} municípios — {nomes[indicador]}")
        fig.update_layout(coloraxis_showscale=False, height=max(400, 28 * top_n))
        st.plotly_chart(fig)

        st.subheader("População × letalidade")
        val = snap[snap["casos"] >= 100]
        pearson = val["pop"].corr(val["letalidade_pct"])
        spearman = val["pop"].corr(val["letalidade_pct"], method="spearman")
        st.caption(f"Municípios com ≥ 100 casos (n={len(val)}). Correlação de Pearson: {pearson:.3f} · "
                   f"Spearman: {spearman:.3f} — relação fraca: o tamanho do município sozinho não explica a letalidade.")
        fig = px.scatter(val, x="pop", y="letalidade_pct", log_x=True, size="casos", color="mortalidade_100k",
                         hover_name="nome_munic", color_continuous_scale="Viridis", size_max=40, opacity=0.7,
                         labels={"pop": "População (escala log)", "letalidade_pct": "Letalidade (%)",
                                 "mortalidade_100k": "Mortalidade/100 mil"})
        st.plotly_chart(fig)

        st.subheader("Distribuição da população dos municípios (gráfico estático)")
        fig_est, ax = plt.subplots(figsize=(9, 3.5))
        ax.hist(snap["pop"], bins=60, color="slateblue", edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel("População estimada")
        ax.set_ylabel("Nº de municípios (log)")
        ax.set_title("A maioria dos municípios é pequena: comparar valores absolutos distorce a análise")
        st.pyplot(fig_est)

        st.subheader("Tabela e download")
        tabela = base[["nome_munic", "pop", "casos", "obitos", "incidencia_100k",
                       "mortalidade_100k", "letalidade_pct"]].sort_values(indicador, ascending=False)
        st.dataframe(tabela.round(2), hide_index=True)
        st.download_button("⬇️ Baixar tabela (CSV)", tabela.to_csv(index=False).encode("utf-8"),
                           "municipios_covid_sp.csv", "text/csv")

# ------------------------------------------------------------------ previsão
with tab_prev:
    st.subheader("Previsão semanal de casos e óbitos novos")
    st.markdown(
        "Modelo de **regressão supervisionada** sobre a série semanal (atributos: valores das 4 semanas anteriores, "
        "média das últimas 4 semanas e tendência recente). A avaliação usa **split temporal** — as últimas semanas "
        "ficam de fora do treino — e é comparada a um **baseline ingênuo** (próxima semana = semana atual).")
    p1, p2, p3, p4 = st.columns(4)
    alvo = p1.selectbox("Variável a prever", ["casos_novos", "obitos_novos"],
                        format_func=lambda x: "Casos novos" if x == "casos_novos" else "Óbitos novos")
    nome_modelo = p2.selectbox("Modelo", list(ml.MODELOS))
    n_teste = p3.slider("Semanas de teste", 8, 26, 12)
    horizonte = p4.slider("Semanas à frente", 1, 8, 4)

    s_sem = ml.serie_semanal(serie[alvo])  # usa a série completa, independente do filtro de período
    try:
        res, m_mod, m_base, modelo_treinado = ml.avaliar(s_sem, nome_modelo, n_teste)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    st.markdown(f"**Escopo:** {escopo}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE (modelo)", f"{m_mod['MAE']:,.1f}", f"{m_mod['MAE'] - m_base['MAE']:+,.1f} vs baseline",
              delta_color="inverse")
    k2.metric("RMSE (modelo)", f"{m_mod['RMSE']:,.1f}", f"{m_mod['RMSE'] - m_base['RMSE']:+,.1f} vs baseline",
              delta_color="inverse")
    k3.metric("R² (teste)", f"{m_mod['R²']:.3f}")
    k4.metric("WAPE", f"{m_mod['WAPE (%)']:.1f}%", f"{m_mod['WAPE (%)'] - m_base['WAPE (%)']:+.1f} p.p. vs baseline",
              delta_color="inverse")
    if m_mod["MAE"] >= m_base["MAE"]:
        st.warning("Neste recorte o modelo **não superou** o baseline ingênuo — o resultado deve ser lido com cautela.")

    futuro = ml.prever_futuro(s_sem, nome_modelo, horizonte)
    hist = s_sem.iloc[-(n_teste + 40):]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="Observado", line=dict(color="gray"))
    fig.add_scatter(x=res.index, y=res["previsto"], name="Previsto (teste)", line=dict(color="crimson", width=3))
    fig.add_scatter(x=res.index, y=res["baseline"], name="Baseline", line=dict(color="steelblue", dash="dot"))
    fig.add_scatter(x=[s_sem.index[-1], *futuro.index], y=[s_sem.iloc[-1], *futuro.values],
                    name="Projeção", line=dict(color="darkorange", dash="dash", width=3))
    fig.update_layout(title="Observado × previsto (por semana)", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), yaxis_title="Casos/óbitos por semana")
    st.plotly_chart(fig)
    st.caption(f"A série da SEADE termina em {DATA_MAX:%d/%m/%Y}; a projeção é, portanto, um exercício "
               "demonstrativo para as semanas seguintes ao fim da base — não uma previsão oficial.")

    imp = ml.importancias(modelo_treinado)
    if imp is not None:
        st.plotly_chart(px.bar(imp, orientation="h", title="Importância dos atributos",
                               labels={"value": "Importância", "index": ""}).update_layout(showlegend=False))
    with st.expander("Ver tabela de teste"):
        st.dataframe(res.round(1))

# ------------------------------------------------------------------ fonte
with tab_sobre:
    st.markdown(f"""
### Fonte dos dados
**Fundação SEADE** (Sistema Estadual de Análise de Dados), com base em SES-SP, SIVEP-Gripe e SIMI-SP.
Repositório oficial: <https://github.com/seade-R/dados-covid-sp> · série diária por município,
{DATA_MIN:%d/%m/%Y} a {DATA_MAX:%d/%m/%Y}. Acesso em 19/08/2026.

### Indicadores (CP1)
| Indicador | Fórmula |
|---|---|
| Incidência | casos / população × 100.000 |
| Mortalidade | óbitos / população × 100.000 |
| Letalidade | óbitos / casos × 100 |
| Média móvel de 7 dias | suaviza o ruído diário e evidencia as ondas |

### Tratamentos
- Valores negativos (correções da própria série) foram truncados em zero.
- Registros com `codigo_ibge` ≥ 9.000.000 (sem município identificado) entram nos totais do Estado, mas ficam fora dos rankings municipais.
- Rankings usam população mínima configurável para evitar distorção em municípios muito pequenos.

### Modelo preditivo
Previsão semanal com Ridge, Random Forest ou Gradient Boosting (scikit-learn), atributos defasados,
split temporal e comparação com baseline. Veja `modelo.py`.

### Limitações
Subnotificação e mudanças na testagem ao longo do tempo afetam a letalidade; o modelo não considera vacinação,
variantes ou mobilidade; a série de dados terminou em nov/2023.
""")
