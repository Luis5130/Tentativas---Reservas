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

def br_int_no_dec(n):
    # versão simples sem casas decimais
    if pd.isna(n):
        return "-"
    return f"{int(n):,}".replace(",", ".")

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

    model = Prophet(holidays=feriados)
    model.fit(df_prophet)
    last_date = df_prophet["ds"].max()
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > last_date]

    # Projeção 2025 (somatório yhat de 2025)
    proj_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum()
    # Armazena para ranking geral (caso alguém use mais tarde)
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
# Ranking de Queda Anual (2025 vs 2023 e 2025 vs 2024) - Geral (todas as UFs)
# ------------------------
todas_ufs = sorted(df["UF"].dropna().unique())

def soma_por_ano(uf, ano):
    mask = (df["UF"] == uf) & (df["ds"].dt.year == int(ano))
    total = df.loc[mask, "y"].sum()
    return int(total)

# Construir ranking de quedas (2025 vs 2023) e (2025 vs 2024)
ranking_queda = []
for uf in todas_ufs:
    y2023 = soma_por_ano(uf, 2023)
    y2024 = soma_por_ano(uf, 2024)
    y2025 = proj_2025_by_all.get(uf, 0.0) if 'proj_2025_by_all' in locals() else 0.0
    queda_2025_2023 = max(0, int(y2023) - int(y2025))
    queda_2025_2024 = max(0, int(y2024) - int(y2025))
    ranking_queda.append({
        "UF": uf,
        "2023 (Executado)": int(y2023),
        "2024 (Executado)": int(y2024),
        "2025 (Projetado)": int(y2025),
        "Queda 2025/2023 (Proj)": queda_2025_2023,
        "Queda 2025/2024 (Proj)": queda_2025_2024
    })

df_ranking = pd.DataFrame(ranking_queda)
# Ordem por maior queda (usa o maior entre as duas quedas projetadas)
df_ranking["Máxima Queda (Proj)"] = df_ranking[["Queda 2025/2023 (Proj)", "Queda 2025/2024 (Proj)"]].max(axis=1)
df_ranking_sorted = df_ranking.sort_values("Máxima Queda (Proj)", ascending=False)

st.subheader("📉 Ranking de Queda Anual (2025 vs 2023 / 2024) – Todas as UFs")
st.dataframe(df_ranking_sorted[["UF", "2023 (Executado)", "2024 (Executado)", "2025 (Projetado)", "Queda 2025/2023 (Proj)", "Queda 2025/2024 (Proj)", "Máxima Queda (Proj)"]])

# Insight rápido: quem tem a maior queda 2025/2023 (Proj) e seu impacto
# calculando impacto relativo com base em 2023
def br_percent(n, d):
    try:
        if d == 0:
            return "0,00%"
        p = (n / d) * 100.0
        s = f"{p:,.2f}"
        s = s.replace(".", ",")
        return s + "%"
    except:
        return "0,00%"

# Montar dados para insight
insight_rows = []
for uf in todas_ufs:
    y2023 = int(soma_por_ano(uf, 2023))
    y2024 = int(soma_por_ano(uf, 2024))
    y2025 = proj_2025_by_all.get(uf, 0.0) if 'proj_2025_by_all' in locals() else 0.0
    q23 = max(0, y2023 - int(y2025))
    q24 = max(0, y2024 - int(y2025))
    insight_rows.append({
        "UF": uf,
        "2023": y2023,
        "2024": y2024,
        "2025 Projetado": int(y2025),
        "Queda 2025/2023 (Proj)": int(q23),
        "Queda 2025/2024 (Proj)": int(q24),
        "Impacto 2025/2023": br_percent(int(q23), int(y2023)) if y2023 > 0 else "0,00%"
    })

df_insight = pd.DataFrame(insight_rows)
# Destaque SP na primeira linha para o insight
df_insight_sorted = df_insight.sort_values("Queda 2025/2023 (Proj)", ascending=False)

st.subheader("🔎 Insight: maior queda (2025 vs 2023 / 2024)")
top5 = df_insight_sorted.head(5)
for _, row in top5.iterrows():
    st.write(f"{row['UF']}: Queda 2025/2023 (Proj) = {br_int(row['Queda 2025/2023 (Proj)'])} "
             f"| Queda 2025/2024 (Proj) = {br_int(row['Queda 2025/2024 (Proj)'])} "
             f"| Impacto vs 2023 = {row['Impacto 2025/2023']}")

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
