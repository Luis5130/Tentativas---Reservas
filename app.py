import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ------------------------
# Carregar dados
# ------------------------
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

def parse_br_number(x):
    """
    Interpreta números no formato BR (ponto como separador de milhares, vírgula como decimal).
    Retorna float ou NaN.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = s.replace(" ", "")
    if "." in s and "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    s = ''.join(ch for ch in s if (ch.isdigit() or ch in ".-"))
    try:
        return float(s)
    except:
        return None

@st.cache_data(ttl=300)
def load_data(url):
    df = pd.read_csv(url)
    if "Mês/Ano" in df.columns and "Tentativa de Reserva" in df.columns:
        df = df.rename(columns={"Mês/Ano": "ds", "Tentativa de Reserva": "y"})
    # Aplicar parsing BR se necessário
    if "y" in df.columns:
        df["y"] = df["y"].apply(parse_br_number)
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    return df

df = load_data(SHEET_CSV)

# Garantir que exista UF
if "UF" not in df.columns:
    st.error("Coluna UF não encontrada nos dados.")
    st.stop()

# Funções de formatação BR
def br_int(n):
    if pd.isna(n):
        return "-"
    i = int(n)
    s = f"{i:,}"
    return s.replace(",", ".")

def br_float(n, dec=2):
    if pd.isna(n):
        return "-"
    s = f"{float(n):,.{dec}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

st.title("📊 Tendência de Reservas + Projeção")

# ------------------------
# Sidebar
# ------------------------
ufs = sorted(df["UF"].dropna().unique())
ufs_selected = st.sidebar.multiselect("Selecione os estados (UF)", ufs, default=ufs[:1])

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
    'upper_window': [30, 61]
})

feriados = pd.concat([feriados_nacionais, ferias_escolares])

# ------------------------
# Projeção por UF (2025) + Gráficos
# ------------------------
# Armazenamento para evitar re-cálculo desnecessário
if "proj_2025_by_uf" not in st.session_state:
    st.session_state.proj_2025_by_uf = {}

# Botão para rodar a projeção 2025 para todas as UFs (opcional)
rodar_projecao = st.button("Rodar projeção 2025 para todas as UFs")

if rodar_projecao:
    st.session_state.proj_2025_by_uf = {}
    for uf in ufs:
        df_u = df[(df["UF"] == uf)][["ds","y"]].copy()
        if df_u.empty:
            st.session_state.proj_2025_by_uf[uf] = 0.0
            continue
        model = Prophet(holidays=feriados)
        model.fit(df_u)
        last_date = df_u["ds"].max()
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] > last_date]
        yhat_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
        st.session_state.proj_2025_by_uf[uf] = float(yhat_2025) if forecast_future is not None else 0.0
    st.success("Projeção 2025 para todas as UFs concluída.")

# Projeção por UF (aplica também se o usuário não apertou o botão, lendo do cache)
proj_2025_by_uf = st.session_state.get("proj_2025_by_uf", {})

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

    # Projeção 2025 (somatório yhat de 2025)
    proj_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
    # Armazena localmente na interface (pode usar para o ranking geral mais abaixo)
    st.session_state.proj_2025_by_uf[uf] = float(proj_2025) if forecast_future is not None else 0.0

    # Detalhes 2025 (mensal)
    monthly_2025 = forecast_future[forecast_future["ds"].dt.year == 2025][["ds","yhat"]].copy()
    # Histórico
    st.subheader(f"📈 Histórico - {uf}")
    fig_hist = px.line(df_prophet, x="ds", y="y", title=f"Histórico - {uf}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Projeção
    st.subheader(f"📊 Projeção - {uf}")
    fig_forecast = px.line(title=f"Projeção de Reservas - {uf}")
    fig_forecast.add_scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Histórico")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Previsão 2025")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Inferior 2025")
    fig_forecast.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines",
                             line=dict(dash="dot", color="gray"), name="Intervalo Superior 2025")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Tabela de Projeção
    st.subheader(f"📊 Tabela de Projeção - {uf}")
    forecast_table = forecast_future[["ds","yhat","yhat_lower","yhat_upper"]].copy()
    forecast_table["Mês/Ano"] = forecast_table["ds"].dt.strftime("%b/%Y")

    st.metric(label=f"Projeção 2025 (UF {uf})", value=br_float(proj_2025_by_uf.get(uf, 0.0), dec=0))

    # Renomear colunas para BR
    forecast_table.rename(columns={
        "yhat": "Previsão 2025",
        "yhat_lower": "Intervalo Inferior 2025",
        "yhat_upper": "Intervalo Superior 2025"
    }, inplace=True)

    st.dataframe(forecast_table[["Mês/Ano","Previsão 2025","Intervalo Inferior 2025","Intervalo Superior 2025"]])

# ------------------------
# Ranking Geral - Todas as UFs (Projetado 2025)
# ------------------------
# Dados de 2023/2024 por UF (executados) para todas as UFs
all_ufs = sorted(df["UF"].dropna().unique())

# Pivot de anos completos (para 2023/2024)
df_all_years = df.copy()
df_all_years["Year"] = df_all_years["ds"].dt.year
pivot_all_years = df_all_years.pivot_table(index="UF", columns="Year", values="y", aggfunc="sum", fill_value=0).reset_index()
for yr in [2023, 2024, 2025]:
    if yr not in pivot_all_years.columns:
        pivot_all_years[yr] = 0

def get_year_all(uf, year):
    row = pivot_all_years[pivot_all_years["UF"] == uf]
    if not row.empty and year in row.columns:
        return int(row.iloc[0][year])
    return 0

# Projeção 2025 para todas as UFs (usa projeção já calculada acima)
proj_2025_by_uf_all = {}
monthly_2025_by_uf_all = {}

for uf in all_ufs:
    df_u = df[df["UF"] == uf][["ds","y"]].copy()
    if df_u.empty:
        proj_2025_by_uf_all[uf] = 0.0
        monthly_2025_by_uf_all[uf] = pd.DataFrame(columns=['ds','yhat'])
        continue
    model = Prophet(holidays=feriados)
    model.fit(df_u)
    last_date = df_u["ds"].max()
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > last_date]
    yhat_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
    proj_2025_by_uf_all[uf] = float(yhat_2025) if forecast_future is not None else 0.0
    monthly_2025_by_uf_all[uf] = forecast_future[forecast_future["ds"].dt.year == 2025][["ds","yhat"]]

# Construir ranking agregado
ranking_all = pd.DataFrame({"UF": all_ufs})
ranking_all["2023 (Executado)"] = ranking_all["UF"].map(lambda uf: get_year_all(uf, 2023))
ranking_all["2024 (Executado)"] = ranking_all["UF"].map(lambda uf: get_year_all(uf, 2024))
ranking_all["2025 (Projetado)"] = ranking_all["UF"].map(lambda uf: proj_2025_by_uf_all.get(uf, 0.0))

ranking_all["Queda 2024/2023 (Real)"] = (ranking_all["2023 (Executado)"] - ranking_all["2024 (Executado)"]).clip(lower=0)
ranking_all["Queda 2025/2023 (Proj)"] = (ranking_all["2023 (Executado)"] - ranking_all["2025 (Projetado)"]).clip(lower=0)
ranking_all["Queda 2025/2024 (Proj)"] = (ranking_all["2024 (Executado)"] - ranking_all["2025 (Projetado)"]).clip(lower=0)

ranking_all["Máxima Queda (Proj)"] = ranking_all[
    ["Queda 2024/2023 (Real)", "Queda 2025/2023 (Proj)", "Queda 2025/2024 (Proj)"]
].max(axis=1)

ranking_all_sorted = ranking_all.sort_values("Máxima Queda (Proj)", ascending=False)

st.subheader("📊 Ver Ranking Geral - Todas as UFs (Projetado 2025)")
# Formatar para BR (inteiros nas 2023/2024, 2025 sem casas)
def to_br_int(val):
    try:
        return br_int(int(val))
    except:
        return br_int(val)

display_all = ranking_all_sorted.copy()
display_all["2023 (Executado)"] = display_all["2023 (Executado)"].apply(to_br_int)
display_all["2024 (Executado)"] = display_all["2024 (Executado)"].apply(to_br_int)
display_all["2025 (Projetado)"] = display_all["2025 (Projetado)"].apply(lambda v: br_int(int(v)) if v is not None else "-")
display_all["Queda 2024/2023 (Real)"] = display_all["Queda 2024/2023 (Real)"].apply(br_int)
display_all["Queda 2025/2023 (Proj)"] = display_all["Queda 2025/2023 (Proj)"].apply(br_int)
display_all["Queda 2025/2024 (Proj)"] = display_all["Queda 2025/2024 (Proj)"].apply(br_int)
display_all["Máxima Queda (Proj)"] = display_all["Máxima Queda (Proj)"].apply(br_int)

st.dataframe(
    display_all[
        ["UF",
         "2023 (Executado)",
         "2024 (Executado)",
         "2025 (Projetado)",
         "Queda 2024/2023 (Real)",
         "Queda 2025/2023 (Proj)",
         "Queda 2025/2024 (Proj)",
         "Máxima Queda (Proj)"]
    ]
)

# Detalhe por UF (opcional)
uf_detail_all = st.selectbox("Ver detalhes de 2025 projetado para uma UF (opcional):",
                             ["Nenhum"] + all_ufs)
if uf_detail_all != "Nenhum":
    monthly_2025 = monthly_2025_by_uf_all.get(uf_detail_all)
    total_2025 = 0.0
    if monthly_2025 is not None and not monthly_2025.empty:
        total_2025 = float(monthly_2025["yhat"].sum())
        st.subheader(f"Detalhes 2025 projetado - {uf_detail_all}")
        fig_detail = px.bar(monthly_2025.rename(columns={'yhat':'Previsão 2025 (mensal)'}),
                            x="ds", y="Previsão 2025 (mensal)",
                            labels={"ds": "Data", "Previsão 2025 (mensal)": "Previsão 2025 (mensal)"},
                            title=f"2025 - {uf_detail_all} (mensal)")
        st.plotly_chart(fig_detail, use_container_width=True)
    else:
        st.write("Sem dados de 2025 projetado para esta UF.")
    st.metric(label=f"Total 2025 projetado - {uf_detail_all}", value=f"{br_int(total_2025)}")

# ------------------------
# ℹ️ Como é calculada a tendência
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo Facebook Prophet, que considera:
- Tendência de longo prazo (crescimento ou queda ao longo do tempo)
- Sazonalidade (padrões anuais, mensais e semanais)
- Feriados e férias escolares
- Intervalo de confiança (faixa de incerteza na previsão)

Os rankings mostram onde a queda absoluta é maior em cada UF, incluindo dados executados (2023/2024) e a projeção para 2025.
""")
