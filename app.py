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

# Filtrar dados
df_uf = df[(df["UF"] == uf) & (df["Mês/ Ano"] >= pd.to_datetime(start_date)) & (df["Mês/ Ano"] <= pd.to_datetime(end_date))]
df_prophet = df_uf[["Mês/ Ano", "Tentativa de Reserva"]].rename(columns={"Mês/ Ano": "ds", "Tentativa de Reserva": "y"})

# Modelo Prophet
horizon = st.sidebar.slider("Meses a projetar", 1, 12, 6)
model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=horizon, freq="MS")
forecast = model.predict(future)

# Traduzir colunas
forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")
forecast_table.rename(columns={
    "yhat": "Previsão",
    "yhat_lower": "Intervalo Inferior",
    "yhat_upper": "Intervalo Superior"
}, inplace=True)
forecast_table = forecast_table[["Mês/Ano", "Previsão", "Intervalo Inferior", "Intervalo Superior"]]

st.subheader("🔮 Tabela de Projeção")
st.dataframe(forecast_table.tail(horizon))

# Gráfico
fig = px.line(title=f"Projeção de Reservas - {uf}")

# Linha histórica
fig.add_scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Histórico")

# Linha de previsão
fig.add_scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Previsão")

# Intervalos de confiança (inferior/superior)
fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", line=dict(dash="dot", color="gray"), name="Intervalo Inferior")
fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(dash="dot", color="gray"), name="Intervalo Superior")

fig.update_layout(
    xaxis_title="Data",
    yaxis_title="Tentativas de Reserva",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 📈 Comparação de variação ano a ano
st.subheader("📊 Variação anual")
df_variacao = df_uf.copy()
df_variacao["Ano"] = df_variacao["Mês/ Ano"].dt.year
df_variacao["Mes"] = df_variacao["Mês/ Ano"].dt.month

# Pivot para comparação
pivot = df_variacao.pivot_table(index="Mes", columns="Ano", values="Tentativa de Reserva", aggfunc="sum")
pivot["2025 vs 2023 (%)"] = ((pivot[2025] - pivot[2023])/pivot[2023])*100
pivot["2024 vs 2023 (%)"] = ((pivot[2024] - pivot[2023])/pivot[2023])*100

st.dataframe(pivot)
