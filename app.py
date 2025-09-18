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

def br_float(n, dec=0):
    if pd.isna(n):
        return "-"
    s = f"{float(n):,.{dec}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

st.title("📊 Tendência de Reservas + Projeção")

# ------------------------
# Dados e projeção (sem botão)
# ------------------------
# UFs disponíveis
ufs = sorted(df["UF"].dropna().unique())

# Seleção de UFs pelo usuário
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
# Função para calcular projeções 2025 por UF (sem botão)
# ------------------------
def compute_projection_all(all_uf, horizon, feriados):
    proj = {}
    monthly = {}
    for uf in all_uf:
        df_u = df[(df["UF"] == uf)][["ds","y"]].copy()
        if df_u.empty:
            proj[uf] = 0.0
            monthly[uf] = pd.DataFrame(columns=['ds','yhat'])
            continue
        model = Prophet(holidays=feriados)
        model.fit(df_u)
        last_date = df_u["ds"].max()
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] > last_date]
        yhat_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
        proj[uf] = float(yhat_2025) if forecast_future is not None else 0.0
        monthly[uf] = forecast_future[forecast_future["ds"].dt.year == 2025][["ds","yhat"]]
    return proj, monthly

# Prepara projeção para todas as UFs (executa na inicialização)
all_ufs = sorted(df["UF"].dropna().unique())
proj_2025_by_all, monthly_2025_by_uf_all = compute_projection_all(all_ufs, horizon, feriados)

# ------------------------
# Histórico e Projeção por UF (com dados BR)
# ------------------------
st.subheader("🔮 Histórico e Projeção por UF")
for uf in ufs_selected:
    df_prophet = df_uf[df_uf["UF"] == uf][["ds","y"]].copy()
    if df_prophet.empty:
        continue

    total_2023_2024 = df[(df["UF"] == uf) & (df["ds"].dt.year.isin([2023,2024]))]["y"].sum()
    has_data = (total_2023_2024 is not None) and (total_2023_2024 > 0)

    if not has_data:
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
    # Guarda na sessão (para uso rápido se precisar)
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

    st.dataframe(forecast_table[["Mês/Ano","yhat","yhat_lower","yhat_upper"]].rename(columns={
        "yhat": "Previsão 2025",
        "yhat_lower": "Intervalo Inferior 2025",
        "yhat_upper": "Intervalo Superior 2025"
    }))

# ------------------------
# Cards de resumo executivo
# ------------------------
# Totais globais
total_2023_global = int(df[df["ds"].dt.year == 2023]["y"].sum())
total_2024_global = int(df[df["ds"].dt.year == 2024]["y"].sum())
proj_total_2025 = int(sum([v for v in proj_2025_by_all.values()])) if proj_2025_by_all else 0

# Maior queda (Proj) entre as UFs (2025 vs 2023/2024)
queda_2025_2023 = {}
queda_2025_2024 = {}
for uf in all_ufs:
    y2023 = int(soma_por_ano := df[(df["UF"] == uf) & (df["ds"].dt.year == 2023)]['y'].sum()) if not df[(df["UF"] == uf) & (df["ds"].dt.year == 2023)].empty else 0
    y2024 = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2024)]['y'].sum()) if not df[(df["UF"] == uf) & (df["ds"].dt.year == 2024)].empty else 0
    proj_u = proj_2025_by_all.get(uf, 0.0)
    queda_2025_2023[uf] = max(0, y2023 - int(proj_u))
    queda_2025_2024[uf] = max(0, y2024 - int(proj_u))

uf_maior_queda = max(queda_2025_2023, key=queda_2025_2023.get) if quedan := queda_2025_2023 else None
maior_queda_2025_2023 = queda_2025_2023.get(uf_maior_queda, 0) if uf_maior_queda else 0
# Impacto relativo (em relação ao total de 2023)
percent_impacto = 0.0
if uf_maior_queda:
    total_2023_uf = int(df[(df["UF"] == uf_maior_queda) & (df["ds"].dt.year == 2023)]['y'].sum())
    if total_2023_uf > 0:
        percent_impacto = (maior_queda_2025_2023 / total_2023_uf) * 100

# Layout em cards (4 colunas)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total 2023 (Executado)", value=br_int(total_2023_global))
with col2:
    st.metric(label="Total 2024 (Executado)", value=br_int(total_2024_global))
with col3:
    st.metric(label="Projeção Total 2025", value=br_int(proj_total_2025))

with col4:
    uf_label = uf_maior_queda if uf_maior_queda else "-"
    st.metric(label=f"Maior Queda 2025/2023 (Proj) - UF {uf_label}", value=br_int(maior_queda_2025_2023))

# Linha extra com impacto relativo (opcional)
st.markdown(f"Impacto relativo da maior queda em 2025/2023: {percent_impacto:.2f}%")

# Observação
st.markdown("""
Notas:
- Os números são formatados em BR (1.000,00 etc.). 
- UFs sem dados suficientes são ocultadas das projeções para evitar números enganosos.
- A projeção 2025 é calculada por UF com o modelo Prophet, levando em conta feriados e férias escolares.
""")

# ------------------------
# ℹ️ Como é calculada a tendência
# ------------------------
st.markdown("""
### ℹ️ Como é calculada a tendência
A projeção é feita usando o modelo Facebook Prophet, que considera:
- Tendência de longo prazo
- Sazonalidade (padrões anuais, mensais e semanais)
- Feriados e férias escolares
- Intervalo de confiança (faixa de incerteza na previsão)
""")

# Observação / extras (opcional)
st.markdown("""
Sugestões adicionais:
- Exportar tabelas para CSV/Excel
- Verificar sazonalidade mensal por UF com heatmap
- Ajustar o modelo com dados adicionais (ex.: 2018-2022) para melhor calibração
""")
