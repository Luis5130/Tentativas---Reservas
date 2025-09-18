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
selected_ufs = st.sidebar.multiselect("Selecione um ou mais estados", ufs, default=[ufs[0]])

start_date = st.sidebar.date_input("Data inicial", df["ds"].min())
end_date = st.sidebar.date_input("Data final", df["ds"].max())
horizon = st.sidebar.slider("Meses a projetar", 1, 12, 6)

# ------------------------
# Filtrar dados por UFs e período
# ------------------------
df_filtered = df[(df["UF"].isin(selected_ufs)) & 
                 (df["ds"] >= pd.to_datetime(start_date)) & 
                 (df["ds"] <= pd.to_datetime(end_date))]

# ------------------------
# Modelo Prophet e gráfico de tendência por UF
# ------------------------
for uf in selected_ufs:
    st.subheader(f"🔮 Tendência / Projeção - {uf}")
    df_uf = df_filtered[df_filtered["UF"] == uf][["ds", "y"]]
    
    model = Prophet()
    model.fit(df_uf)
    
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    
    last_date = df_uf["ds"].max()
    forecast_future = forecast[forecast["ds"] > last_date]
    
    fig_forecast = px.line(title=f"Projeção de Reservas - {uf}")
    fig_forecast.add_scatter(x=df_uf["ds"], y=df_uf["y"], mode="lines+markers", name="Histórico")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Previsão")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Inferior")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Superior")
    st.plotly_chart(fig_forecast, use_container_width=True)

# ------------------------
# Ranking de Maiores Quedas
# ------------------------
st.subheader("📉 Ranking de Maiores Quedas")

# Pivot table por UF e Ano
df_ranking = df_filtered.copy()
df_ranking["Ano"] = df_ranking["ds"].dt.year
ranking = df_ranking.pivot_table(index="UF", columns="Ano", values="y", aggfunc="sum").reset_index()

# Definir comparação dinâmica
anos = sorted([col for col in ranking.columns if isinstance(col, int)])
if anos:
    latest_year = max(anos)
    for ano in anos:
        if ano <= 2025:
            base_year = 2023
        else:
            base_year = 2025
        if base_year in ranking.columns and ano in ranking.columns:
            ranking["Perda Absoluta"] = ranking[base_year] - ranking[ano]
            ranking["Variação (%)"] = ((ranking[ano] - ranking[base_year]) / ranking[base_year]) * 100

# Ordenar por Perda Absoluta
ranking_sorted = ranking.sort_values("Perda Absoluta", ascending=False)
st.dataframe(ranking_sorted[["UF", base_year, latest_year, "Perda Absoluta", "Variação (%)"]].head(10))

# ------------------------
# Explicativo da Tendência
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando **Facebook Prophet**:
- **Tendência de longo prazo** (crescimento ou queda)  
- **Sazonalidade** (padrões anuais e mensais)  
- **Intervalo de confiança** (faixa de incerteza)  

📌 Comparação dinâmica:
- Anos 2023, 2024, 2025 → comparação sempre com 2023  
- Projeções 2026+ → comparação com 2025  

Isso ajuda a identificar os **maiores impactos absolutos**, mesmo em estados grandes.
""")
