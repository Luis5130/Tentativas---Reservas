import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ------------------------
# Carregar dados
# ------------------------
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

@st.cache_data(ttl=300)
def load_data(url):
    df = pd.read_csv(url)
    df = df.rename(columns={"Mês/Ano": "ds", "Tentativa de Reserva": "y"})
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df

df = load_data(SHEET_CSV)
st.title("📊 Tendência de Reservas + Projeção")

# ------------------------
# Sidebar
# ------------------------
ufs = sorted(df["UF"].dropna().unique())
ufs_selected = st.sidebar.multiselect("Selecione os estados", ufs, default=ufs[:1])

start_date = st.sidebar.date_input("Data inicial", df["ds"].min())
end_date = st.sidebar.date_input("Data final", df["ds"].max())
horizon = st.sidebar.slider("Meses a projetar", 1, 24, 12)

# Filtrar dados por UF(s) e período
df_uf = df[(df["UF"].isin(ufs_selected)) & (df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] <= pd.to_datetime(end_date))]

# ------------------------
# Feriados nacionais + férias escolares
# ------------------------
feriados_nacionais = pd.DataFrame({
    'holiday': ['Confraternização', 'Carnaval', 'Paixão de Cristo', 'Tiradentes', 'Dia do Trabalho',
                'Corpus Christi', 'Independência', 'Nossa Senhora Aparecida', 'Finados', 'Proclamação da República'],
    'ds': pd.to_datetime(['2023-01-01','2023-02-20','2023-04-07','2023-04-21','2023-05-01',
                          '2023-06-08','2023-09-07','2023-10-12','2023-11-02','2023-11-15']),
    'lower_window': 0, 'upper_window': 1
})

ferias_escolares = pd.DataFrame({
    'holiday': ['Férias Escolares', 'Férias Escolares'],
    'ds': pd.to_datetime(['2023-07-01', '2023-12-01']),
    'lower_window': [0, 0],
    'upper_window': [30, 61]  # Julho: 31 dias, Dez/Jan: 62 dias
})

feriados = pd.concat([feriados_nacionais, ferias_escolares])

# ------------------------
# Calcular tração histórica por UF (para ranking)
# ------------------------
df_years = df_uf.copy()
df_years["Year"] = df_years["ds"].dt.year
pivot_years = df_years.pivot_table(index="UF", columns="Year", values="y", aggfunc="sum", fill_value=0).reset_index()

# Garantir colunas 2023, 2024, 2025 existindo
for yr in [2023, 2024, 2025]:
    if yr not in pivot_years.columns:
        pivot_years[yr] = 0

def get_year_value(uf, year):
    row = pivot_years[pivot_years["UF"] == uf]
    if not row.empty and year in row.columns:
        return int(row.iloc[0][year])
    return 0

# ------------------------
# Ranking de Maiores Quedas por UF (Base Real)
# (exclui explicitamente "2023" como coluna, utiliza 2023/2024/2025 para as diferenças)
# ------------------------
queda_2024_2023_real = []
queda_2025_2023_real = []
queda_2025_2024_real = []

for uf in ufs_selected:
    y2023 = get_year_value(uf, 2023)
    y2024 = get_year_value(uf, 2024)
    y2025 = get_year_value(uf, 2025)
    queda_2024_2023_real.append(max(0, y2023 - y2024))
    queda_2025_2023_real.append(max(0, y2023 - y2025))
    queda_2025_2024_real.append(max(0, y2024 - y2025))

ranking_real = pd.DataFrame({
    "UF": ufs_selected,
    "Queda 2024/2023 (Real)": queda_2024_2023_real,
    "Queda 2025/2023 (Real)": queda_2025_2023_real,
    "Queda 2025/2024 (Real)": queda_2025_2024_real
})
ranking_real["Max Queda (Real)"] = ranking_real[
    ["Queda 2024/2023 (Real)", "Queda 2025/2023 (Real)", "Queda 2025/2024 (Real)"]
].max(axis=1)
ranking_real_sorted = ranking_real.sort_values("Max Queda (Real)", ascending=False)

st.subheader("📉 Ranking de Maiores Quedas por UF (Base Real)")
st.dataframe(ranking_real_sorted[["UF","Max Queda (Real)","Queda 2024/2023 (Real)","Queda 2025/2023 (Real)","Queda 2025/2024 (Real)"]])

# ------------------------
# Projeção por UF (2025) + Gráficos
# ------------------------
proj_2025_by_uf = {}

st.subheader("🔮 Tendência / Projeção por UF (com Projeção 2025)")
for uf in ufs_selected:
    df_prophet = df_uf[df_uf["UF"] == uf][["ds","y"]].copy()
    if df_prophet.empty:
        continue

    model = Prophet(holidays=feriados)
    model.fit(df_prophet)
    last_date = df_prophet["ds"].max()
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > last_date]

    # Soma da projeção para 2025 (yhat)
    proj_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
    proj_2025_by_uf[uf] = proj_2025

    # Gráfico Histórico
    st.subheader(f"📈 Histórico - {uf}")
    fig_hist = px.line(df_prophet, x="ds", y="y", title=f"Histórico - {uf}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Gráfico de Projeção
    st.subheader(f"📊 Projeção - {uf}")
    fig_forecast = px.line(title=f"Projeção de Reservas - {uf}")
    fig_forecast.add_scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Histórico")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Previsão")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Inferior")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Superior")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Tabela de Projeção
    st.subheader(f"📊 Tabela de Projeção - {uf}")
    forecast_table = forecast_future[["ds","yhat","yhat_lower","yhat_upper"]].copy()
    forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")

    st.metric(label=f"Projeção 2025 (UF {uf})", value=f"{int(proj_2025_by_uf.get(uf,0)):,}")

    st.dataframe(forecast_table[["Mês/Ano","yhat","yhat_lower","yhat_upper"]])

# ------------------------
# Ranking de Maiores Quedas com Projeção (2025)
# ------------------------
# Preparar valores de 2023/2024 por UF (real) já calculados acima
def get_year_val_safe(uf, year):
    row = pivot_years[pivot_years["UF"] == uf]
    if not row.empty and year in row.columns:
        return int(row.iloc[0][year])
    return 0

# Montar listas para o ranking de projeção
queda_2024_2023_real_list = [max(0, get_year_val_safe(uf, 2023) - get_year_val_safe(uf, 2024)) for uf in ufs_selected]
queda_2025_2023_proj_list = [max(0, get_year_val_safe(uf, 2023) - proj_2025_by_uf.get(uf, 0.0)) for uf in ufs_selected]
queda_2025_2024_proj_list = [max(0, get_year_val_safe(uf, 2024) - proj_2025_by_uf.get(uf, 0.0)) for uf in ufs_selected]

ranking_proj = pd.DataFrame({
    "UF": ufs_selected,
    "Queda 2024/2023 (Real)": queda_2024_2023_real_list,
    "Queda 2025/2023 (Proj)": queda_2025_2023_proj_list,
    "Queda 2025/2024 (Proj)": queda_2025_2024_proj_list
})
ranking_proj["Max Queda (Proj)"] = ranking_proj[
    ["Queda 2024/2023 (Real)", "Queda 2025/2023 (Proj)", "Queda 2025/2024 (Proj)"]
].max(axis=1)
ranking_proj_sorted = ranking_proj.sort_values("Max Queda (Proj)", ascending=False)

st.subheader("📉 Ranking de Maiores Quedas por UF (Projeção 2025)")
st.dataframe(ranking_proj_sorted[["UF","Max Queda (Proj)","Queda 2024/2023 (Real)","Queda 2025/2023 (Proj)","Queda 2025/2024 (Proj)"]])

# ------------------------
# Projeção Nacional Consolidada
# ------------------------
# Projeção 2025 nacional: soma das projeções por UF
proj_nacional_2025 = sum(proj_2025_by_uf.values()) if proj_2025_by_uf else 0

st.subheader("🌎 Projeção Nacional Consolidada (2025)")
st.metric(label="Reservas projetadas em 2025 (nacional)", value=f"{int(proj_nacional_2025):,}")

# Observação
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo **Facebook Prophet**, que considera:
- Tendência de longo prazo
- Sazonalidade
- Feriados e férias escolares
- Intervalo de confiança
Os rankings mostram onde a queda absoluta é maior em cada UF, tanto com base nos dados reais até 2024 quanto com a projeção para 2025.
""")

# Explicação adicional (opcional)
st.markdown("""
Se preferir, posso adaptar para:
- apresentar apenas uma tabela consolidada com as três quedas por UF (sem separação Real/Proj)
- exibir as projeções com gráficos adicionais ou versões compactas
""")
