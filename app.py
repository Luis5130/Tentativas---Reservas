import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# ------------------------
# Carregar dados
# ------------------------
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

@st.cache_data(ttl=300)
def load_data(url):
    df = pd.read_csv(url)
    # Se a planilha vier com Mês/Ano e Tentativa de Reserva, renomeie para ds/y
    if "Mês/Ano" in df.columns and "Tentativa de Reserva" in df.columns:
        df = df.rename(columns={"Mês/Ano": "ds", "Tentativa de Reserva": "y"})
    # ds -> datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    # y -> numérica (inteiro)
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce").astype('Int64')
    return df

df = load_data(SHEET_CSV)

# Garantir que exista UF
if "UF" not in df.columns:
    st.error("Coluna UF não encontrada nos dados.")
    st.stop()

# ------------------------
# Funções utilitárias
# ------------------------
def mes_br_port(dt):
    # representa mês/ano em formato curto (jan/2025)
    month_names = {
        1: "jan", 2: "fev", 3: "mar", 4: "abr",
        5: "mai", 6: "jun", 7: "jul", 8: "ago",
        9: "set", 10: "out", 11: "nov", 12: "dez"
    }
    m = int(dt.month)
    y = dt.year
    return f"{month_names[m]}/{y}"

# ------------------------
# Título
# ------------------------
st.title("Tentativa de Reservas + Tendência")

# ------------------------
# Sidebar: UF + Período
# ------------------------
ufs = sorted(df["UF"].dropna().unique())
ufs_selected = st.sidebar.multiselect("Selecione os estados (UF)", ufs, default=ufs[:1])

start_date = st.sidebar.date_input("Data inicial", df["ds"].min())
end_date = st.sidebar.date_input("Data final", df["ds"].max())
horizon = st.sidebar.slider("Meses a projetar", 1, 24, 12)

# Filtrar dados por UF(s) e período
df_uf = df[(df["UF"].isin(ufs_selected)) & (df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] <= pd.to_datetime(end_date))]

# ------------------------
# Feriados nacionais + férias escolares
# ------------------------
feriados_nacionais = pd.DataFrame({
    'holiday': ['Confraternização', 'Carnaval', 'Paixão de Cristo', 'Tiradentes', 'Dia do Trabalho', 'Corpus Christi', 'Independência', 'Nossa Senhora Aparecida', 'Finados', 'Proclamação da República'],
    'ds': pd.to_datetime(['2023-01-01','2023-02-20','2023-04-07','2023-04-21','2023-05-01', '2023-06-08','2023-09-07','2023-10-12','2023-11-02','2023-11-15']),
    'lower_window': 0,
    'upper_window': 1
})
ferias_escolares = pd.DataFrame({
    'holiday': ['Férias Escolares', 'Férias Escolares'],
    'ds': pd.to_datetime(['2023-07-01', '2023-12-01']),
    'lower_window': [0, 0],
    'upper_window': [30, 61]
})
feriados = pd.concat([feriados_nacionais, ferias_escolares])

# ------------------------
# Projeção por UF (pré-calc na inicialização) + cache
# ------------------------
def compute_projection_all(all_uf, horizon, feriados):
    proj = {}
    monthly = {}
    for uf in all_uf:
        df_u = df[(df["UF"] == uf)][["ds","y"]].copy()
        if df_u.empty:
            proj[uf] = 0.0
            monthly[uf] = pd.DataFrame(columns=['ds','yhat'])
            continue
        df_u["y"] = df_u["y"].astype(float)
        model = Prophet(holidays=feriados)
        model.fit(df_u)
        last_date = df_u["ds"].max()
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] > last_date]
        yhat_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum() if not forecast_future.empty else 0.0
        proj[uf] = float(yhat_2025)
        monthly[uf] = forecast_future[forecast_future["ds"].dt.year == 2025][["ds","yhat"]].copy()
    return proj, monthly

# Projeção total por UF (pré-calc) com cache
all_ufs = sorted(df["UF"].dropna().unique())
if "proj_2025_by_all" not in st.session_state:
    st.session_state["proj_2025_by_all"], st.session_state["monthly_2025_by_uf_all"] = compute_projection_all(all_ufs, horizon, feriados)
proj_2025_by_all = st.session_state.get("proj_2025_by_all", {})
monthly_2025_by_uf_all = st.session_state.get("monthly_2025_by_uf_all", {})

# ------------------------
# Explicação do Modelo
# ------------------------
with st.expander("ℹ️ Como funciona a projeção"):
    st.markdown("""
    O modelo de projeção utiliza o **Facebook Prophet**, uma ferramenta de previsão de séries temporais desenvolvida para lidar com dados que apresentam:
    - **Tendência** (crescimento ou queda ao longo do tempo)  
    - **Sazonalidade anual/mensal/semanal**  
    - **Impacto de feriados e eventos especiais** (no caso, feriados nacionais e férias escolares foram incluídos)  

    O Prophet é um modelo **aditivo**, no qual a série é decomposta em:
    - Tendência  
    - Sazonalidade  
    - Efeitos de feriados  

    Ele é robusto para dados faltantes, mudanças de tendência e funciona bem em cenários de negócio.
    """)

# ------------------------
# Histórico por UF + Projeção por UF
# ------------------------
st.subheader("Histórico e Projeção por UF (selecionadas)")
for uf in ufs_selected:
    df_prophet = df[(df["UF"] == uf)][["ds","y"]].copy().sort_values("ds")
    if df_prophet.empty:
        continue

    # Garantir que y seja inteiro
    df_prophet["y"] = df_prophet["y"].astype(int)

    # Construir modelo e previsão
    model = Prophet(holidays=feriados)
    model.fit(df_prophet.rename(columns={"ds":"ds","y":"y"}))
    last_date = df_prophet["ds"].max()
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > last_date]

    # Reforçar inteiros no forecast
    if not forecast_future.empty:
        forecast_future["yhat"] = forecast_future["yhat"].round().astype(int)
        forecast_future["yhat_lower"] = forecast_future["yhat_lower"].round().astype(int)
        forecast_future["yhat_upper"] = forecast_future["yhat_upper"].round().astype(int)

    # Gráfico único com histórico + projeção (2 traces) + banda
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=df_prophet["ds"],
        y=df_prophet["y"],
        mode="lines",
        name="Histórico",
        hovertemplate="Data: %{x|%b/%Y}<br>Reservas: %{y:.0f}"
    ))

    # Projeção 2025
    if not forecast_future.empty:
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat"],
            mode="lines",
            name="Projeção 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Projeção 2025: %{y:.0f}"
        ))
        # Banda de incerteza ( Superior e Inferior )
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_upper"],
            mode="lines",
            line=dict(dash="dot", color="gray"),
            name="Intervalo Superior 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Superior: %{y:.0f}"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_lower"],
            mode="lines",
            line=dict(dash="dot", color="gray"),
            name="Intervalo Inferior 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Inferior: %{y:.0f}",
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.15)"
        ))

    fig.update_layout(
        title=f"Histórico + Projeção - {uf}",
        xaxis_title="Data",
        yaxis_title="Reservas",
        yaxis=dict(tickformat="d"),  # inteiros no eixo
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela de Projeção 2025 (se houver)
    if not forecast_future.empty:
        forecast_table = forecast_future[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        forecast_table["Mês/Ano"] = forecast_table["ds"].apply(lambda d: mes_br_port(pd.Timestamp(d)))
        forecast_table.rename(columns={
            "yhat": "Previsão 2025",
            "yhat_lower": "Intervalo Inferior 2025",
            "yhat_upper": "Intervalo Superior 2025"
        }, inplace=True)
        forecast_table["Mês/Ano"] = forecast_table["ds"].apply(lambda d: mes_br_port(d))
        st.dataframe(forecast_table[["Mês/Ano","Previsão 2025","Intervalo Inferior 2025","Intervalo Superior 2025"]])

        st.download_button(
            label="Exportar Projeção 2025 (CSV)",
            data=forecast_table[["Mês/Ano","Previsão 2025","Intervalo Inferior 2025","Intervalo Superior 2025"]].to_csv(index=False).encode('utf-8'),
            file_name=f"projecao_2025_{uf}.csv",
            mime="text/csv"
        )

    # Resumo da UF (opcional)
    total_2023_uf = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2023)]['y'].sum())
    total_2024_uf = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2024)]['y'].sum())
    proj_uf_2025 = int(proj_2025_by_all.get(uf, 0.0))
    st.markdown(f"Resumo da UF {uf}:")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(label="2023 (Executado)", value=str(total_2023_uf))
    with colB:
        st.metric(label="2024 (Executado)", value=str(total_2024_uf))
    with colC:
        st.metric(label="Projeção 2025 (UF)", value=str(proj_uf_2025))

# ------------------------
# Ranking Geral de UFs
# ------------------------
st.subheader("📊 Ranking Geral de UFs")

ranking_data = []
for uf in all_ufs:
    total_2023 = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2023)]['y'].sum())
    total_2024 = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2024)]['y'].sum())
    total_2025_real = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2025)]['y'].sum())
    proj_2025 = int(proj_2025_by_all.get(uf, 0.0))

    total_2025 = total_2025_real + proj_2025  # realizado até agora + projetado

    diff_25_24 = total_2025 - total_2024
    diff_25_23 = total_2025 - total_2023

    ranking_data.append({
        "UF": uf,
        "2023 (Executado)": total_2023,
        "2024 (Executado)": total_2024,
        "2025 (Realizado + Projetado)": total_2025,
        "Δ 2025-2024": diff_25_24,
        "Δ 2025-2023": diff_25_23
    })

df_ranking = pd.DataFrame(ranking_data)

# Ordenar pelo maior crescimento em relação a 2024
df_ranking = df_ranking.sort_values(by="Δ 2025-2024", ascending=False)

st.dataframe(df_ranking)

# Gráfico de barras com as diferenças
fig_rank = go.Figure()
fig_rank.add_trace(go.Bar(
    x=df_ranking["UF"],
    y=df_ranking["Δ 2025-2024"],
    name="2025 - 2024",
    marker_color="steelblue"
))
fig_rank.add_trace(go.Bar(
    x=df_ranking["UF"],
    y=df_ranking["Δ 2025-2023"],
    name="2025 - 2023",
    marker_color="darkorange"
))
fig_rank.update_layout(
    barmode="group",
    title="Comparativo de Crescimento por UF",
    xaxis_title="UF",
    yaxis_title="Diferença de Reservas"
)
st.plotly_chart(fig_rank, use_container_width=True)
