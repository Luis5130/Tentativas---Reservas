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
    # Espera: colunas xUF, ds (data), y (inteiro)
    if "Mês/Ano" in df.columns and "Tentativa de Reserva" in df.columns:
        df = df.rename(columns={"Mês/Ano": "ds", "Tentativa de Reserva": "y"})
    # ds -> datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    # y -> numeric inteiro (sem parsing BR)
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce").astype('Int64')
    return df

df = load_data(SHEET_CSV)

# Garantir que exista UF
if "UF" not in df.columns:
    st.error("Coluna UF não encontrada nos dados.")
    st.stop()

# ------------------------
# Título
# ------------------------
st.title("Tentativa de Reservas + Tendência (sem parsing BR)")

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
# (Mantém apenas se quiser; pode comentar)
# ------------------------
feriados_nacionais = pd.DataFrame({
    'holiday': ['Confraternização', 'Carnaval', 'Paixão de Cristo', 'Tiradentes', 'Dia do Trabalho', 'Corpus Christi', 'Independência', 'Nossa Senhora Aparecida', 'Finados', 'Proclamação da República'],
    'ds': pd.to_datetime(['2023-01-01','2023-02-20','2023-04-07','2023-04-21','2023-05-01','2023-06-08','2023-09-07','2023-10-12','2023-11-02','2023-11-15']),
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
        # Converter y para float para o Prophet
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
# Histórico por UF + Projeção por UF
# ------------------------
st.subheader("Histórico e Projeção por UF (selecionadas)")
for uf in ufs_selected:
    df_prophet = df[(df["UF"] == uf)][["ds","y"]].copy().sort_values("ds")
    if df_prophet.empty:
        continue

    # Garantir que y seja inteiro para exibir corretamente
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

    # Gráfico único com histórico + projeção (2 traces)
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
            hovertemplate="Data: %{x|%b/%Y}<br>Projeção 2025: %{yhat:.0f}"
        ))
        # Intervalos (inferior e superior)
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_lower"],
            mode="lines",
            line=dict(dash="dot", color="gray"),
            name="Intervalo Inferior 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Inferior: %{yhat_lower:.0f}"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_upper"],
            mode="lines",
            line=dict(dash="dot", color="gray"),
            name="Intervalo Superior 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Superior: %{yhat_upper:.0f}"
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
        forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")
        forecast_table.rename(columns={
            "yhat": "Previsão 2025",
            "yhat_lower": "Intervalo Inferior 2025",
            "yhat_upper": "Intervalo Superior 2025"
        }, inplace=True)
        forecast_table["Mês/Ano"] = forecast_table["ds"].apply(lambda d: d.strftime("%b/%Y"))
        st.dataframe(forecast_table[["Mês/Ano","Previsão 2025","Intervalo Inferior 2025","Intervalo Superior 2025"]])

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
