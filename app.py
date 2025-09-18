import streamlit as st
import pandas as pd
import plotly.express as px

SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

@st.cache_data(ttl=300)
def load_data(url):
    df = pd.read_csv(url)
    df["Mês/ Ano"] = pd.to_datetime(df["Mês/ Ano"], format="%Y-%m")
    return df

df = load_data(SHEET_CSV)

st.title("Tendência de Reservas")
uf = st.sidebar.selectbox("UF", sorted(df["UF"].unique()))
df_filtrado = df[df["UF"] == uf]

fig = px.line(df_filtrado, x="Mês/ Ano", y="Tentativa de Reserva", title=f"Tendência - {uf}")
st.plotly_chart(fig, use_container_width=True)

df_filtrado["Variação"] = df_filtrado["Tentativa de Reserva"].diff()
st.subheader("Maiores quedas")
st.dataframe(df_filtrado.sort_values("Variação").head(5))
