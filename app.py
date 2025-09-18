import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# ------------------------
# Carregar dados do Google Sheets
# ------------------------
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

df = pd.read_csv(url)

# Verificar as colunas disponíveis
st.write("Colunas do CSV:", df.columns.tolist())

# Renomear corretamente para Prophet
# Supondo que sua planilha tenha colunas: "Data" e "Valor"
df = df.rename(columns={
    "Data": "ds",
    "Valor": "y"
})

# Converter colunas
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = pd.to_numeric(df["y"], errors="coerce")

# ------------------------
# Rodar Prophet
# ------------------------
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12, freq="M")
forecast = model.predict(future)

last_date = df["ds"].max()
forecast_future = forecast[forecast["ds"] > last_date]

# ------------------------
# Sidebar
# ------------------------
st.sidebar.header("⚙️ Opções de Visualização")
show_hist = st.sidebar.checkbox("Mostrar Histórico", value=True)
show_forecast = st.sidebar.checkbox("Mostrar Tendência / Projeção", value=True)
show_year = st.sidebar.checkbox("Mostrar Comparação Ano a Ano", value=True)
show_table = st.sidebar.checkbox("Mostrar Tabela de Projeção", value=True)

# ------------------------
# Gráfico 1 - Histórico
# ------------------------
if show_hist:
    st.subheader("📈 Histórico de Reservas")
    fig_hist = px.line(df, x="ds", y="y", title="Histórico de Reservas")
    st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------
# Gráfico 2 - Tendência
# ------------------------
if show_forecast:
    st.subheader("🔮 Tendência / Projeção de Reservas")
    fig_forecast = px.line(title="Projeção de Reservas")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"],
                             mode="lines", name="Previsão")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"],
                             mode="lines", line=dict(dash="dot", color="gray"),
                             name="Intervalo Inferior")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"],
                             mode="lines", line=dict(dash="dot", color="gray"),
                             name="Intervalo Superior")
    st.plotly_chart(fig_forecast, use_container_width=True)

# ------------------------
# Gráfico 3 - Comparação Ano a Ano
# ------------------------
if show_year:
    st.subheader("📊 Comparação Ano a Ano")
    df["Ano"] = df["ds"].dt.year
    df["Mês"] = df["ds"].dt.month
    fig_year = px.line(df, x="Mês", y="y", color="Ano",
                       title="Comparação de Reservas por Ano")
    st.plotly_chart(fig_year, use_container_width=True)

# ------------------------
# Tabela de projeção
# ------------------------
if show_table:
    st.subheader("📊 Tabela de Projeção (meses futuros)")
    forecast_table = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")
    forecast_table.rename(columns={
        "yhat": "Previsão",
        "yhat_lower": "Intervalo Inferior",
        "yhat_upper": "Intervalo Superior"
    }, inplace=True)
    st.dataframe(forecast_table[["Mês/Ano", "Previsão", "Intervalo Inferior", "Intervalo Superior"]])

# ------------------------
# Explicativo
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo **Facebook Prophet**, que considera:
- **Tendência de longo prazo** (crescimento ou queda ao longo do tempo).  
- **Sazonalidade** (padrões anuais, mensais e semanais).  
- **Intervalo de confiança** (faixa de incerteza na previsão).  

Como 2023 foi o melhor ano e 2024/2025 mostraram queda,  
o modelo projeta essa desaceleração para os meses seguintes.
""")
