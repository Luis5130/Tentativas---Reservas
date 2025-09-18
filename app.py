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
# Meta: projeção 2025 (opcional) armazenada em session_state
# ------------------------
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

    model = Prophet(holidays=feriados)
    model.fit(df_prophet)
    last_date = df_prophet["ds"].max()
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > last_date]

    # Projeção 2025 (somatório yhat de 2025)
    proj_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
    # Armazenar para ranking geral
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

    st.metric(label=f"Projeção 2025 (UF {uf})", value=br_float(proj_2025, dec=0))

    forecast_table.rename(columns={
        "yhat": "Previsão 2025",
        "yhat_lower": "Intervalo Inferior 2025",
        "yhat_upper": "Intervalo Superior 2025"
    }, inplace=True)

    st.dataframe(forecast_table[["Mês/Ano","Previsão 2025","Intervalo Inferior 2025","Intervalo Superior 2025"]])

# ------------------------
# Ranking Geral - Todas as UFs (Projetado 2025)
# ------------------------
todas_ufs = sorted(df["UF"].dropna().unique())

# Função para soma BR por UF/ano (2023, 2024)
def soma_por_ano(uf, ano):
    mask = (df["UF"] == uf) & (df["ds"].dt.year == int(ano))
    total = df.loc[mask, "y"].sum()
    return int(total)

# Construir ranking geral usando todas as UFs
ranking_all = pd.DataFrame({"UF": todas_ufs})

# 2023/2024 executados
ranking_all["2023 (Executado)"] = ranking_all["UF"].map(lambda uf: soma_por_ano(uf, 2023))
ranking_all["2024 (Executado)"] = ranking_all["UF"].map(lambda uf: soma_por_ano(uf, 2024))

# 2025 projetado (do dicionário)
ranking_all["2025 (Projetado)"] = ranking_all["UF"].map(
    lambda uf: proj_2025_by_all.get(uf, 0.0) if "proj_2025_by_all" in locals() else 0.0
)

# Quedas (2025 vs 2023 / 2025 vs 2024)
ranking_all["Queda 2025/2023 (Proj)"] = (ranking_all["2023 (Executado)"] - ranking_all["2025 (Projetado)"]).clip(lower=0)
ranking_all["Queda 2025/2024 (Proj)"] = (ranking_all["2024 (Executado)"] - ranking_all["2025 (Projetado)"]).clip(lower=0)

# Criticidade (maior queda entre as duas)
ranking_all["Crítica (max entre quedas)"] = ranking_all[["Queda 2025/2023 (Proj)", "Queda 2025/2024 (Proj)"]].max(axis=1)

ranking_all_sorted = ranking_all.sort_values("Crítica (max entre quedas)", ascending=False)

# Exibir com nomes BR
st.subheader("📊 Ver Ranking Geral - Todas as UFs (Projetado 2025)")
display_all = ranking_all_sorted.copy()
display_all["2023 (Executado)"] = display_all["2023 (Executado)"].apply(br_int)
display_all["2024 (Executado)"] = display_all["2024 (Executado)"].apply(br_int)
display_all["2025 (Projetado)"] = display_all["2025 (Projetado)"].apply(lambda v: br_int(int(v)) if not pd.isna(v) else "-")
display_all["Queda 2025/2023 (Proj)"] = display_all["Queda 2025/2023 (Proj)"].apply(br_int)
display_all["Queda 2025/2024 (Proj)"] = display_all["Queda 2025/2024 (Proj)"].apply(br_int)
display_all["Crítica (max entre quedas)"] = display_all["Crítica (max entre quedas)"].apply(br_int)

st.dataframe(
    display_all[[
        "UF",
        "2023 (Executado)",
        "2024 (Executado)",
        "2025 (Projetado)",
        "Queda 2025/2023 (Proj)",
        "Queda 2025/2024 (Proj)",
        "Crítica (max entre quedas)"
    ]]
)

# Detalhe por UF (opcional)
uf_detail_all = st.selectbox("Ver detalhes de 2025 projetado para uma UF (opcional):",
                             ["Nenhum"] + todas_ufs)
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
    st.metric(label=f"Total 2025 projetado - {uf_detail_all}", value=br_int(int(total_2025)))

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

# Observação / extras (opcional)
st.markdown("""
Sugestões adicionais:
- Esconder projeções de UFs sem dados suficientes para manter a tela mais limpa
- Exibir o ranking em formato compacto (uma linha por UF) se preferir
- Expor opções de exportar as tabelas para CSV/Excel
""")
