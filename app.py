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
show_combined = st.sidebar.checkbox("Exibir gráfico combinado (Histórico + Tendência)", True)

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

# ---- GRÁFICO 3: Combinado ----
if show_combined:
    st.subheader("📊 Histórico + Tendência")
    fig_combined = px.line(title=f"Histórico + Projeção de Reservas - {uf}")
    fig_combined.add_scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Histórico")
    fig_combined.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Previsão")
    fig_combined.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Inferior")
    fig_combined.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Superior")
    fig_combined.update_layout(
        xaxis_title="Data",
        yaxis_title="Tentativas de Reserva",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_combined, use_container_width=True)

# ---- Tabela somente meses futuros ----
if show_forecast:
    st.subheader("📊 Tabela de Projeção (meses futuros)")
    forecast_table = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")
    
    # Comparação % com último mês real
    last_real_value = df_prophet.loc[df_prophet["ds"] == last_date, "y"].values[0]
    forecast_table["Variação % vs Último Real"] = ((forecast_table["yhat"] - last_real_value) / last_real_value) * 100

    forecast_table.rename(columns={
        "yhat": "Previsão",
        "yhat_lower": "Intervalo Inferior",
        "yhat_upper": "Intervalo Superior"
    }, inplace=True)
    forecast_table = forecast_table[["Mês/Ano", "Previsão", "Intervalo Inferior", "Intervalo Superior", "Variação % vs Último Real"]]

    st.dataframe(forecast_table)

# 📈 Comparação de variação ano a ano
st.subheader("📊 Variação anual por mês")
df_variacao = df_uf.copy()
df_variacao["Ano"] = df_variacao["Mês/ Ano"].dt.year
df_variacao["Mes"] = df_variacao["Mês/ Ano"].dt.month

pivot = df_variacao.pivot_table(index="Mes", columns="Ano", values="Tentativa de Reserva", aggfunc="sum")

# Gráfico de barras comparando anos
fig_var = px.bar(pivot, barmode="group", title=f"Comparação Mensal - {uf}")
st.plotly_chart(fig_var, use_container_width=True)

st.dataframe(pivot)

# ---- TOP QUEDAS POR UF ----
st.subheader("📉 Top quedas por UF (ano contra ano)")
df_all = df.copy()
df_all["Ano"] = df_all["Mês/ Ano"].dt.year
df_all["Mes"] = df_all["Mês/ Ano"].dt.month

# Último ano e ano anterior
ano_max = df_all["Ano"].max()
ano_prev = ano_max - 1

df_last = df_all[df_all["Ano"] == ano_max].groupby("UF")["Tentativa de Reserva"].sum()
df_prev = df_all[df_all["Ano"] == ano_prev].groupby("UF")["Tentativa de Reserva"].sum()

df_comp = pd.DataFrame({"AnoAtual": df_last, "AnoAnterior": df_prev})
df_comp["Variação %"] = ((df_comp["AnoAtual"] - df_comp["AnoAnterior"]) / df_comp["AnoAnterior"]) * 100
df_comp = df_comp.sort_values("Variação %")

st.dataframe(df_comp.head(5))  # top 5 maiores quedas

fig_topquedas = px.bar(df_comp.head(5), x="Variação %", y=df_comp.head(5).index, orientation="h",
                       title="Top 5 UFs com maiores quedas (%)", color="Variação %")
st.plotly_chart(fig_topquedas, use_container_width=True)
