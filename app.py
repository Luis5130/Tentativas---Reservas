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
ufs = sorted(df["UF"].unique())
ufs_selected = st.sidebar.multiselect("Selecione os estados", ufs, default=ufs[:1])

start_date = st.sidebar.date_input("Data inicial", df["ds"].min())
end_date = st.sidebar.date_input("Data final", df["ds"].max())
horizon = st.sidebar.slider("Meses a projetar", 1, 24, 12)

# Filtrar dados por UF(s) e período
df_uf = df[df["UF"].isin(ufs_selected) & (df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] <= pd.to_datetime(end_date))]

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
# Loop por UF para gráficos e projeção
# ------------------------
st.subheader("🔮 Tendência / Projeção por UF")
for uf in ufs_selected:
    df_prophet = df_uf[df_uf["UF"] == uf][["ds", "y"]].copy()
    if df_prophet.empty:
        continue

    model = Prophet(holidays=feriados)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    last_date = df_prophet["ds"].max()
    forecast_future = forecast[forecast["ds"] > last_date]

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

    # Tabela de projeção com variação %
    st.subheader(f"📊 Tabela de Projeção - {uf}")
    forecast_table = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")

    anos = sorted(df_prophet["ds"].dt.year.unique())
    for ano in anos:
        df_ano = df_prophet[df_prophet["ds"].dt.year == ano].copy()
        df_ano["Mês"] = df_ano["ds"].dt.month
        df_ano.set_index("Mês", inplace=True)
        forecast_table[f"Vs {ano} (%)"] = forecast_table["ds"].dt.month.map(
            lambda m: ((forecast_table.set_index(forecast_table["ds"].dt.month)["yhat"].get(m,0) - df_ano["y"].get(m,0)) / df_ano["y"].get(m,1) * 100)
        )

    forecast_table.rename(columns={
        "yhat": "Previsão",
        "yhat_lower": "Intervalo Inferior",
        "yhat_upper": "Intervalo Superior"
    }, inplace=True)
    st.dataframe(forecast_table[["Mês/Ano","Previsão","Intervalo Inferior","Intervalo Superior"] + [col for col in forecast_table.columns if "Vs" in col]])

# ------------------------
# Ranking de Maiores Quedas por UF
# ------------------------
st.subheader("📉 Ranking de Maiores Quedas por UF")
df_ranking = df.copy()
df_ranking["Ano"] = df_ranking["ds"].dt.year

ranking = df_ranking.pivot_table(index="UF", columns="Ano", values="y", aggfunc="sum").reset_index()

# Comparação dinâmica
cols_perda = []
for ano in ranking.columns[1:]:  # Ignora UF
    if int(ano) > 2025:
        ranking[f"Perda Absoluta {ano}"] = ranking[2025] - ranking[ano]
    else:
        ranking[f"Perda Absoluta {ano}"] = ranking[2023] - ranking[ano]
    cols_perda.append(f"Perda Absoluta {ano}")

ranking[cols_perda] = ranking[cols_perda].apply(pd.to_numeric, errors='coerce').fillna(0)
ranking["Max Perda Absoluta"] = ranking[cols_perda].max(axis=1)
ranking_sorted = ranking.sort_values("Max Perda Absoluta", ascending=False)
st.dataframe(ranking_sorted[["UF","Max Perda Absoluta"] + cols_perda])

# ------------------------
# Tabela Nacional Consolidada
# ------------------------
st.subheader("🌎 Projeção Nacional Consolidada")
df_total = df.copy()
df_total_grouped = df_total.groupby("ds")["y"].sum().reset_index()

model_total = Prophet(holidays=feriados)
model_total.fit(df_total_grouped)
future_total = model_total.make_future_dataframe(periods=horizon, freq="MS")
forecast_total = model_total.predict(future_total)
forecast_total_future = forecast_total[forecast_total["ds"] > df_total_grouped["ds"].max()]

forecast_total_future["Mês/Ano"] = forecast_total_future["ds"].dt.strftime("%b/%Y")
forecast_total_future.rename(columns={"yhat":"Previsão","yhat_lower":"Intervalo Inferior","yhat_upper":"Intervalo Superior"}, inplace=True)
st.dataframe(forecast_total_future[["Mês/Ano","Previsão","Intervalo Inferior","Intervalo Superior"]])

# ------------------------
# Explicativo
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo **Facebook Prophet**, que considera:
- **Tendência de longo prazo** (crescimento ou queda ao longo do tempo)  
- **Sazonalidade** (padrões anuais, mensais e semanais)  
- **Feriados e férias escolares** (como Natal, Ano Novo e períodos de férias)  
- **Intervalo de confiança** (faixa de incerteza na previsão)  

O ranking de perdas mostra **onde a queda absoluta é maior** em cada UF, permitindo focar nas regiões críticas.
""")
