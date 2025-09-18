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

# Sidebar para escolher estado e horizonte
ufs = sorted(df["UF"].unique())
uf = st.sidebar.selectbox("Selecione o estado", ufs, index=0)
horizon = st.sidebar.slider("Meses a projetar", 1, 12, 6)

# Filtrar estado
df_uf = df[df["UF"] == uf][["Mês/ Ano", "Tentativa de Reserva"]]
df_uf = df_uf.rename(columns={"Mês/ Ano": "ds", "Tentativa de Reserva": "y"})

# Ajustar modelo Prophet
model = Prophet()
model.fit(df_uf)

# Criar datas futuras
future = model.make_future_dataframe(periods=horizon, freq="MS")
forecast = model.predict(future)

# Gráfico interativo
fig = px.line(forecast, x="ds", y="yhat", title=f"Projeção de Reservas - {uf}")
fig.add_scatter(x=df_uf["ds"], y=df_uf["y"], mode="markers+lines", name="Histórico")
fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
                line=dict(dash="dot", color="gray"), name="Intervalo inferior")
fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
                line=dict(dash="dot", color="gray"), name="Intervalo superior")

fig.update_layout(
    xaxis_title="Data",
    yaxis_title="Tentativas de Reserva",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔮 Tabela de Projeção")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))
