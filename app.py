import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# ------------------------
# Carregar os dados
# ------------------------
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv")  # ajuste para seu arquivo
df["ds"] = pd.to_datetime(df["ds"])
df = df.rename(columns={"valor": "y"})

# ------------------------
# Rodar Prophet
# ------------------------
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12, freq="M")
forecast = model.predict(future)

# Última data real
last_date = df["ds"].max()
forecast_future = forecast[forecast["ds"] > last_date]

# ------------------------
# Sidebar com opções
# ------------------------
st.sidebar.header("⚙️ Opções de Visualização")
show_hist = st.sidebar.checkbox("Mostrar Histórico", value=True)
show_forecast = st.sidebar.checkbox("Mostrar Tendência / Projeção", value=True)
show_year = st.sidebar.checkbox("Mostrar Comparação Ano a Ano", value=True)
show_table = st.sidebar.checkbox("Mostrar Tabela de Projeção", value=True)

uf = "Brasil"  # ajuste conforme sua base

# ------------------------
# Gráfico 1 - Histórico
# ------------------------
if show_hist:
    st.subheader("📈 Histórico de Reservas")
    fig_hist = px.line(title=f"Histórico de Reservas - {uf}")
    fig_hist.add_scatter(x=df["ds"], y=df["y"], mode="lines+markers", name="Histórico")
    fig_hist.update_layout(
        xaxis_title="Data",
        yaxis_title="Tentativas de Reserva",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------
# Gráfico 2 - Tendência
# ------------------------
if show_forecast:
    st.subheader("🔮 Tendência / Projeção de Reservas")
    fig_forecast = px.line(title=f"Projeção de Reservas - {uf}")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"],
                             mode="lines", name="Previsão")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"],
                             mode="lines", line=dict(dash="dot", color="gray"),
                             name="Intervalo Inferior")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"],
                             mode="lines", line=dict(dash="dot", color="gray"),
                             name="Intervalo Superior")
    fig_forecast.update_layout(
        xaxis_title="Data",
        yaxis_title="Tentativas de Reserva",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# ------------------------
# Gráfico 3 - Comparação Ano a Ano
# ------------------------
if show_year:
    st.subheader("📊 Comparação Ano a Ano")
    df["Ano"] = df["ds"].dt.year
    fig_year = px.line(df, x="ds", y="y", color="Ano",
                       title="Comparação de Reservas por Ano")
    fig_year.update_layout(
        xaxis_title="Data",
        yaxis_title="Tentativas de Reserva",
        template="plotly_white",
        hovermode="x unified"
    )
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
    forecast_table = forecast_table[["Mês/Ano", "Previsão", "Intervalo Inferior", "Intervalo Superior"]]
    st.dataframe(forecast_table)

# ------------------------
# Explicativo da tendência
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo **Facebook Prophet**, que considera:
- **Tendência de longo prazo**: se o volume cresce ou cai ao longo do tempo.  
- **Sazonalidade**: repetições anuais, mensais ou semanais detectadas nos dados.  
- **Intervalo de confiança**: a faixa cinza mostra a incerteza natural da previsão.  

⚠️ Importante: o modelo aprende com os dados históricos.  
Ou seja, como **2023 foi o melhor ano** e **2024/2025 tiveram queda**,  
a tendência projetada reflete esse comportamento recente.
""")
