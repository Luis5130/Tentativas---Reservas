import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ------------------------
# Carregar dados do Google Sheets
# ------------------------
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

@st.cache_data(ttl=300)
def load_data(url):
    df = pd.read_csv(url)
    df = df.rename(columns={"Mês/ Ano": "ds", "Tentativa de Reserva": "y"})
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df

df = load_data(SHEET_CSV)

st.title("📊 Tendência de Reservas + Projeção")

# ------------------------
# Sidebar
# ------------------------
ufs = sorted(df["UF"].unique())
uf = st.sidebar.selectbox("Selecione o estado", ufs, index=0)

start_date = st.sidebar.date_input("Data inicial", df["ds"].min())
end_date = st.sidebar.date_input("Data final", df["ds"].max())
horizon = st.sidebar.slider("Meses a projetar", 1, 12, 6)

# Filtrar dados por UF e período
df_uf = df[(df["UF"] == uf) & (df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] <= pd.to_datetime(end_date))]
df_prophet = df_uf[["ds", "y"]]

# ------------------------
# Modelo Prophet
# ------------------------
model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=horizon, freq="MS")
forecast = model.predict(future)

last_date = df_prophet["ds"].max()
forecast_future = forecast[forecast["ds"] > last_date]

# ------------------------
# Gráfico 1 - Histórico
# ------------------------
st.subheader("📈 Histórico de Reservas")
fig_hist = px.line(df_prophet, x="ds", y="y", title=f"Histórico - {uf}")
st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------
# Gráfico 2 - Tendência
# ------------------------
st.subheader("🔮 Tendência / Projeção (somente meses futuros)")
fig_forecast = px.line(title=f"Projeção de Reservas - {uf}")
fig_forecast.add_scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Histórico")
fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Previsão")
fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines",
                         line=dict(dash="dot", color="gray"), name="Intervalo Inferior")
fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines",
                         line=dict(dash="dot", color="gray"), name="Intervalo Superior")
st.plotly_chart(fig_forecast, use_container_width=True)

# ------------------------
# Tabela de Projeção com % vs anos anteriores
# ------------------------
st.subheader("📊 Tabela de Projeção")
forecast_table = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")

# Calcular variação percentual vs anos anteriores
anos = sorted(df_uf["ds"].dt.year.unique())
for ano in anos:
    df_ano = df_uf[df_uf["ds"].dt.year == ano].set_index(df_uf["ds"].dt.month)
    forecast_table[f"Vs {ano} (%)"] = forecast_table["ds"].dt.month.map(
        lambda m: ((forecast_table.set_index(forecast_table["ds"].dt.month)["yhat"][m] - df_ano["y"].get(m, None)) / df_ano["y"].get(m, None) * 100)
        if df_ano["y"].get(m, None) not in [None, 0] else None
    )

forecast_table.rename(columns={
    "yhat": "Previsão",
    "yhat_lower": "Intervalo Inferior",
    "yhat_upper": "Intervalo Superior"
}, inplace=True)

st.dataframe(forecast_table[["Mês/Ano", "Previsão", "Intervalo Inferior", "Intervalo Superior"] + [col for col in forecast_table.columns if "Vs" in col]])

# ------------------------
# Comparação Ano a Ano
# ------------------------
st.subheader("📊 Comparação Ano a Ano")
df_variacao = df_uf.copy()
df_variacao["Ano"] = df_variacao["ds"].dt.year
df_variacao["Mês"] = df_variacao["ds"].dt.month
fig_year = px.line(df_variacao, x="Mês", y="y", color="Ano", markers=True,
                   title=f"Comparação de Reservas por Ano - {uf}")
st.plotly_chart(fig_year, use_container_width=True)

# ------------------------
# Ranking de Maiores Quedas por UF
# ------------------------
st.subheader("📉 Ranking de Maiores Quedas por UF (2025 vs 2023)")
df_ranking = df.copy()
df_ranking["Ano"] = df_ranking["ds"].dt.year
ranking = df_ranking.pivot_table(index="UF", columns="Ano", values="y", aggfunc="sum").reset_index()

if 2023 in ranking.columns and 2025 in ranking.columns:
    ranking["Variação (%)"] = ((ranking[2025] - ranking[2023]) / ranking[2023]) * 100
    ranking = ranking.sort_values("Variação (%)")
    st.dataframe(ranking[["UF", 2023, 2025, "Variação (%)"]].head(10))

# ------------------------
# Explicativo
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo **Facebook Prophet**, que considera:
- **Tendência de longo prazo** (crescimento ou queda ao longo do tempo)  
- **Sazonalidade** (padrões anuais, mensais e semanais)  
- **Intervalo de confiança** (faixa de incerteza na previsão)  

Como 2023 foi o melhor ano e 2024/2025 mostraram queda,  
o modelo projeta essa desaceleração para os próximos meses.
""")
