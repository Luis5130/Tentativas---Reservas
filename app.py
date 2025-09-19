import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ------------------------
# Carregar dados
# ------------------------
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

@st.cache_data
def load_data():
    df = pd.read_csv(SHEET_CSV, sep=",")
    df["ds"] = pd.to_datetime(df["ds"])
    return df

df = load_data()

# ------------------------
# Seleção de UF
# ------------------------
ufs = df["UF"].unique()
uf_selected = st.selectbox("Selecione a UF", ufs)

df_uf = df[df["UF"] == uf_selected][["ds", "y"]]

# ------------------------
# Modelo Prophet
# ------------------------
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(df_uf.rename(columns={"ds": "ds", "y": "y"}))

future = model.make_future_dataframe(periods=12, freq="M")
forecast = model.predict(future)

# ------------------------
# Gráfico de previsão
# ------------------------
fig = px.line(forecast, x="ds", y="yhat", title=f"Projeção para {uf_selected}")
fig.add_scatter(x=df_uf["ds"], y=df_uf["y"], mode="markers", name="Histórico")
st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Cálculo de totais anuais
# ------------------------
df["ano"] = df["ds"].dt.year
totais = df.groupby(["UF", "ano"])["y"].sum().reset_index()

# Separar anos disponíveis
dados_pivot = totais.pivot(index="UF", columns="ano", values="y").fillna(0)
dados_pivot = dados_pivot.rename(columns={
    2023: "2023 (Executado)",
    2024: "2024 (Executado)",
    2025: "2025 (Realizado + Projetado)"
})

# Calcular deltas
dados_pivot["Δ 2025-2024"] = dados_pivot["2025 (Realizado + Projetado)"] - dados_pivot["2024 (Executado)"]
dados_pivot["Δ 2025-2023"] = dados_pivot["2025 (Realizado + Projetado)"] - dados_pivot["2023 (Executado)"]

# Ranking ordenado por Δ 2025-2024
ranking = dados_pivot.sort_values(by="Δ 2025-2024", ascending=True).reset_index()

# Inserir posição
ranking.insert(0, "Posição", range(1, len(ranking) + 1))

# ------------------------
# Exibir tabela
# ------------------------
st.markdown("## 📊 Ranking Geral de UFs")
st.dataframe(
    ranking.style.format(thousands=","),
    use_container_width=True
)

# ------------------------
# Explicação do modelo
# ------------------------
with st.expander("ℹ️ Como funciona a projeção"):
    st.markdown("""
    O modelo de projeção utiliza o **Facebook Prophet**, uma ferramenta de previsão de séries temporais desenvolvida para lidar com dados que apresentam:
    - **Tendência** (crescimento ou queda ao longo do tempo)  
    - **Sazonalidade anual/mensal/semanal**  
    - **Impacto de feriados e eventos especiais**  

    O Prophet é um modelo **aditivo**, no qual a série é decomposta em:
    - Tendência  
    - Sazonalidade  
    - Efeitos de feriados  

    Ele é robusto para dados faltantes, mudanças de tendência e funciona bem em cenários de negócio.
    """)
