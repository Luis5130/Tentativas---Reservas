import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# ------------------------
# Configuração inicial
# ------------------------
st.set_page_config(page_title="Tentativas de Reserva + Projeção", layout="wide")

SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

# ------------------------
# Carregar dados
# ------------------------
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_CSV)
    df = df.rename(columns={"Mês/Ano": "ds", "Tentativa de Reserva": "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("Int64")
    return df

df = load_data()

# ------------------------
# Função para formatar mês/ano
# ------------------------
def mes_br_port(dt):
    month_names = {
        1: "jan", 2: "fev", 3: "mar", 4: "abr",
        5: "mai", 6: "jun", 7: "jul", 8: "ago",
        9: "set", 10: "out", 11: "nov", 12: "dez"
    }
    return f"{month_names[dt.month]}/{dt.year}"

# ------------------------
# Título
# ------------------------
st.title("📊 Tentativas de Reserva + Tendência")

# Explicação do modelo
with st.expander("ℹ️ Como funciona o modelo de projeção"):
    st.markdown("""
    O modelo utilizado é o **Prophet**, desenvolvido pelo Facebook/Meta.

    Ele é um modelo **aditivo**, que decompõe a série temporal em três partes principais:
    - **Tendência**: crescimento ou queda ao longo do tempo.  
    - **Sazonalidade**: padrões recorrentes (mensais, anuais, etc).  
    - **Feriados/Eventos**: impacto de datas específicas como feriados nacionais e férias escolares.

    O Prophet é robusto a valores ausentes, variações bruscas e funciona bem mesmo com poucos dados.
    """)

# ------------------------
# Sidebar: filtros
# ------------------------
ufs = sorted(df["UF"].dropna().unique())
ufs_selected = st.sidebar.multiselect("Selecione os estados (UF)", ufs, default=ufs[:1])

start_date = st.sidebar.date_input("Data inicial", df["ds"].min())
end_date = st.sidebar.date_input("Data final", df["ds"].max())
horizon = st.sidebar.slider("Meses a projetar", 1, 24, 12)

df_filtered = df[(df["UF"].isin(ufs_selected)) &
                 (df["ds"] >= pd.to_datetime(start_date)) &
                 (df["ds"] <= pd.to_datetime(end_date))]

# ------------------------
# Feriados nacionais + férias escolares
# ------------------------
feriados_nacionais = pd.DataFrame({
    'holiday': ['Confraternização', 'Carnaval', 'Paixão de Cristo', 'Tiradentes', 'Dia do Trabalho', 'Corpus Christi',
                'Independência', 'Nossa Senhora Aparecida', 'Finados', 'Proclamação da República'],
    'ds': pd.to_datetime(['2023-01-01','2023-02-20','2023-04-07','2023-04-21','2023-05-01','2023-06-08',
                          '2023-09-07','2023-10-12','2023-11-02','2023-11-15']),
    'lower_window': 0,
    'upper_window': 1
})
ferias_escolares = pd.DataFrame({
    'holiday': ['Férias Escolares', 'Férias Escolares'],
    'ds': pd.to_datetime(['2023-07-01', '2023-12-01']),
    'lower_window': [0, 0],
    'upper_window': [30, 61]
})
feriados = pd.concat([feriados_nacionais, ferias_escolares])

# ------------------------
# Função: projeção por UF
# ------------------------
def compute_projection(df, horizon, feriados):
    results = {}
    for uf in df["UF"].unique():
        df_u = df[df["UF"] == uf][["ds","y"]].dropna().copy()
        if df_u.empty:
            continue
        df_u["y"] = df_u["y"].astype(float)
        model = Prophet(holidays=feriados)
        model.fit(df_u)
        last_date = df_u["ds"].max()
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] > last_date]

        # arredonda projeção
        forecast_future["yhat"] = forecast_future["yhat"].round().astype(int)
        results[uf] = forecast_future
    return results

projections = compute_projection(df, horizon, feriados)

# ------------------------
# Histórico + projeção por UF selecionada
# ------------------------
st.subheader("Histórico e Projeção por UF (selecionadas)")
for uf in ufs_selected:
    df_prophet = df[df["UF"] == uf][["ds","y"]].dropna().copy().sort_values("ds")

    if df_prophet.empty:
        continue

    forecast_future = projections.get(uf, pd.DataFrame())

    fig = go.Figure()
    # Histórico
    fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"],
                             mode="lines", name="Histórico",
                             hovertemplate="Data: %{x|%b/%Y}<br>Reservas: %{y:.0f}"))
    # Projeção
    if not forecast_future.empty:
        fig.add_trace(go.Scatter(x=forecast_future["ds"], y=forecast_future["yhat"],
                                 mode="lines", name="Projeção",
                                 hovertemplate="Data: %{x|%b/%Y}<br>Projeção: %{y:.0f}"))
    fig.update_layout(title=f"Histórico + Projeção - {uf}",
                      xaxis_title="Data", yaxis_title="Reservas",
                      yaxis=dict(tickformat="d"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Ranking geral por UF
# ------------------------
st.subheader("🏆 Ranking Geral de UFs")

ranking = []
for uf in df["UF"].unique():
    total_2023 = df[(df["UF"] == uf) & (df["ds"].dt.year == 2023)]["y"].sum()
    total_2024 = df[(df["UF"] == uf) & (df["ds"].dt.year == 2024)]["y"].sum()
    proj_2025 = projections.get(uf, pd.DataFrame())
    total_2025 = 0
    if not proj_2025.empty:
        total_2025 = proj_2025[proj_2025["ds"].dt.year == 2025]["yhat"].sum()

    ranking.append({
        "UF": uf,
        "2023": int(total_2023),
        "2024": int(total_2024),
        "2025 (Realizado+Projetado)": int(total_2025),
        "Δ 2025-2024": int(total_2025 - total_2024),
        "Δ 2025-2023": int(total_2025 - total_2023),
    })

df_ranking = pd.DataFrame(ranking)
df_ranking = df_ranking.sort_values(by="Δ 2025-2024", ascending=True).reset_index(drop=True)
df_ranking.insert(0, "Posição", range(1, len(df_ranking)+1))

st.dataframe(df_ranking.style.format(thousands=","))
