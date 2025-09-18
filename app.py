import streamlit as st
import pandas as pd
import plotly.express as px

SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/SEU_LINK/pub?output=csv"

df = pd.read_csv(SHEET_CSV)
df["Mês/ Ano"] = pd.to_datetime(df["Mês/ Ano"], format="%Y-%m")

st.title("📊 Tendência de Reservas")

# Seleção múltipla de estados
ufs = st.sidebar.multiselect("Selecione os estados", df["UF"].unique(), default=["SP","RJ","MG"])

df_filtrado = df[df["UF"].isin(ufs)]

fig = px.line(
    df_filtrado,
    x="Mês/ Ano",
    y="Tentativa de Reserva",
    color="UF",
    markers=True,
    title="Tendência de Reservas"
)

fig.update_xaxes(dtick="M3", tickformat="%b/%Y", tickangle=45)
fig.update_layout(
    xaxis_title="Período",
    yaxis_title="Tentativas de Reserva",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
