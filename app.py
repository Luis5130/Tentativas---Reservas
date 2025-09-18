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
    Interpreta números no formato BR (ponto como separador de milhares,
    vírgula como separador decimal). Retorna float ou NaN.
    Ex.: "40.917" -> 40917.0, "1.234,56" -> 1234.56
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if "." in s and "," in s:
        # "1.234,56" -> removê milhares, vírgula vira ponto
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        if "." in s:
            # "40.917" -> 40917
            s = s.replace(".", "")
        if "," in s:
            # "447,540" -> 447.540
            s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

@st.cache_data(ttl=300)
def load_data(url):
    df = pd.read_csv(url)
    if "Mês/Ano" in df.columns and "Tentativa de Reserva" in df.columns:
        df = df.rename(columns={"Mês/Ano": "ds", "Tentativa de Reserva": "y"})
    # Aplicar parsing BR
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
if "proj_2025_by_all" not in st.session_state:
    st.session_state["proj_2025_by_all"] = {}

if "monthly_2025_by_uf_all" not in st.session_state:
    st.session_state["monthly_2025_by_uf_all"] = {}

# Botão para rodar a projeção 2025 para todas as UFs
rodar_projecao = st.button("Rodar projeção 2025 para todas as UFs (opcional)")

if rodar_projecao:
    proj_all = {}
    monthly_all = {}
    for uf in sorted(df["UF"].dropna().unique()):
        df_u = df[(df["UF"] == uf)][["ds","y"]].copy()
        if df_u.empty:
            proj_all[uf] = 0.0
            monthly_all[uf] = pd.DataFrame(columns=['ds','yhat'])
            continue
        model = Prophet(holidays=feriados)
        model.fit(df_u)
        last_date = df_u["ds"].max()
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] > last_date]
        yhat_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
        proj_all[uf] = float(yhat_2025) if forecast_future is not None else 0.0
        monthly_all[uf] = forecast_future[forecast_future["ds"].dt.year == 2025][["ds","yhat"]]
    st.session_state["proj_2025_by_all"] = proj_all
    st.session_state["monthly_2025_by_uf_all"] = monthly_all
    st.success("Projeção 2025 para todas as UFs concluída.")

# Validação de projeção atual
proj_2025_by_all = st.session_state.get("proj_2025_by_all", {})
monthly_2025_by_uf_all = st.session_state.get("monthly_2025_by_uf_all", {})

# ------------------------
# Histórico e Projeção por UF (com dados BR)
# ------------------------
st.subheader("🔮 Tendência / Projeção por UF (com Projeção 2025)")
for uf in ufs_selected:
    df_prophet = df_uf[df_uf["UF"] == uf][["ds","y"]].copy()
    if df_prophet.empty:
        continue

    # Checa se há dados relevantes para a UF
    total_2023_2024 = df[(df["UF"] == uf) & (df["ds"].dt.year.isin([2023,2024]))]["y"].sum()
    se_tem_dados = total_2023_2024 is not None and total_2023_2024 > 0

    if not se_tem_dados:
        # Esconder UFs sem dados suficientes
        continue

    model = Prophet(holidays=feriados)
    model.fit(df_prophet)
    last_date = df_prophet["ds"].max()
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > last_date]

    # Projeção 2025 (somatório yhat de 2025)
    proj_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
    # Armazenar para ranking geral (se usado no futuro)
    st.session_state.setdefault("proj_2025_by_all", {})
    st.session_state["proj_2025_by_all"][uf] = float(proj_2025) if forecast_future is not None else 0.0

    # Detalhe mensal de 2025 para o UF
    monthly_2025 = forecast_future[forecast_future["ds"].dt.year == 2025][["ds","yhat"]].copy()
    st.session_state.setdefault("monthly_2025_by_uf_all", {})
    st.session_state["monthly_2025_by_uf_all"][uf] = monthly_2025

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

    # Não exibir o valor bruto da projeção como título separado
    st.dataframe(forecast_table[["Mês/Ano","yhat","yhat_lower","yhat_upper"]].rename(columns={
        "yhat": "Previsão 2025",
        "yhat_lower": "Intervalo Inferior 2025",
        "yhat_upper": "Intervalo Superior 2025"
    }))

# ------------------------
# ℹ️ Como é calculada a tendência
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo Facebook Prophet, que considera:
- Tendência de longo prazo
- Sazonalidade (padrões anuais, mensais e semanais)
- Feriados e férias escolares
- Intervalo de confiança
""")
