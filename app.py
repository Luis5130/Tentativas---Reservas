import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# URL do Google Sheets publicado como CSV
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

@st.cache_data(ttl=300)
def load_data(url):
    df = pd.read_csv(url)
    df["Mês/ Ano"] = pd.to_datetime(df["Mês/ Ano"], format="%Y-%m")
    return df

df = load_data(SHEET_CSV)

st.title("📊 Tendência de Reservas + Projeção")

# Sidebar: seleção de estado
ufs = sorted(df["UF"].unique())
uf = st.sidebar.selectbox("Selecione o estado", ufs, index=0)

# Sidebar: seleção do período
start_date = st.sidebar.date_input("Data inicial", df["Mês/ Ano"].min())
end_date = st.sidebar.date_input("Data final", df["Mês/ Ano"].max())

# Sidebar: meses a projetar
horizon = st.sidebar.slider("Meses a projetar", 1, 12, 6)

# Sidebar: exibir gráficos
show_hist = st.sidebar.checkbox("Exibir gráfico histórico", True)
show_forecast = st.sidebar.checkbox("Exibir gráfico de tendência", True)

# Filtrar dados
df_uf = df[(df["UF"] == uf) & (df["Mês/ Ano"] >= pd.to_datetime(start_date)) & (df["Mês/ Ano"] <= pd.to_datetime(end_date))]
df_prophet = df_uf[["Mês/ Ano", "Tentativa de Reserva"]].rename(columns={"Mês/ Ano": "ds", "Tentativa de Reserva": "y"})

# Modelo Prophet
model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=horizon, freq="MS")
forecast = model.predict(future)

# Última data com dado real
last_date = df_prophet["ds"].max()

# Previsão somente futuro
forecast_future = forecast[forecast["ds"] > last_date]

# ---- GRÁFICO 1: Histórico ----
if show_hist:
    st.subheader("📈 Histórico de Reservas")
    fig_hist = px.line(title=f"Histórico de Reservas - {uf}")
    fig_hist.add_scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Histórico")
    fig_hist.update_layout(
        xaxis_title="Data",
        yaxis_title="Tentativas de Reserva",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---- GRÁFICO 2: Tendência ----
if show_forecast:
    st.subheader("🔮 Tendência / Projeção de Reservas")
    fig_forecast = px.line(title=f"Projeção de Reservas - {uf}")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Previsão")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Inferior")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Superior")
    fig_forecast.update_layout(
        xaxis_title="Data",
        yaxis_title="Tentativas de Reserva",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# ---- Tabela somente meses futuros ----
if show_forecast:
    st.subheader("📊 Tabela de Projeção (meses futuros)")
    forecast_table = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")
    forecast_table.rename(columns={
        "yhat": "Previsão",
        "yhat_lower": "Intervalo Inferior",
        "yhat_upper": "Intervalo Superior"
    }, inplace=True)
    forecast_table = forecast_table[["Mês/Ano", "Previsão", "Intervalo Inferior", "Intervalo Superior"]]

    st.dataframe(forecast_table)

# 📈 Comparação de variação ano a ano
st.subheader("📊 Variação anual")
df_variacao = df_uf.copy()
df_variacao["Ano"] = df_variacao["Mês/ Ano"].dt.year
df_variacao["Mes"] = df_variacao["Mês/ Ano"].dt.month

# Pivot para comparação
pivot = df_variacao.pivot_table(index="Mes", columns="Ano", values="Tentativa de Reserva", aggfunc="sum")

# Adiciona comparações automáticas se anos existirem
anos = pivot.columns.tolist()
if 2023 in anos and 2024 in anos:
    pivot["2024 vs 2023 (%)"] = ((pivot[2024] - pivot[2023]) / pivot[2023]) * 100
if 2023 in anos and 2025 in anos:
    pivot["2025 vs 2023 (%)"] = ((pivot[2025] - pivot[2023]) / pivot[2023]) * 100

st.dataframe(pivot)
