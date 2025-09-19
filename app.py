import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet

# ------------------------
# Compatibilidade de cache (versões antigas/novas do Streamlit)
# ------------------------
def cache_decorator(ttl=300):
    # Se a API tiver cache_data (Streamlit moderno), usa com ttl
    if hasattr(st, "cache_data"):
        return lambda f: st.cache_data(ttl=ttl)(f)
    else:
        # Versões antigas usam st.cache sem ttl
        return st.cache

# ------------------------
# Carregar dados
# ------------------------
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS7OOWK8wX0B9ulh_Vtmv-R_pbVREiwknncX8oSvnZ4o5wf00gcFhyEEgo3kxW0PmturRda4wL5OCNn/pub?gid=145140176&single=true&output=csv"

@cache_decorator(ttl=300)
def load_data(url):
    df = pd.read_csv(url)

    # Normalizar colunas esperadas, sem apagar colunas novas
    # 1) Mapear Mês/Ano -> ds, caso exista
    if "Mês/Ano" in df.columns:
        df = df.rename(columns={"Mês/Ano": "ds"})

    # 2) ds -> datetime se possível
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # 3) Tentativa de Reserva -> y (para Prophet)
    if "Tentativa de Reserva" in df.columns:
        df["Tentativa de Reserva"] = pd.to_numeric(df["Tentativa de Reserva"], errors="coerce")
        df["y"] = df["Tentativa de Reserva"]

    # 4) Serviços Convertidos (nova coluna de conversão)
    if "Serviços Convertidos" in df.columns:
        df["Serviços Convertidos"] = pd.to_numeric(df["Serviços Convertidos"], errors="coerce")

    return df

df = load_data(SHEET_CSV)

# Garantir que exista UF
if "UF" not in df.columns:
    st.error("Coluna UF não encontrada nos dados.")
    st.stop()

# ------------------------
# Funções utilitárias
# ------------------------
def mes_br_port(dt):
    # representa mês/ano em formato curto (jan/2025)
    month_names = {
        1: "jan", 2: "fev", 3: "mar", 4: "abr",
        5: "mai", 6: "jun", 7: "jul", 8: "ago",
        9: "set", 10: "out", 11: "nov", 12: "dez"
    }
    m = int(dt.month)
    y = dt.year
    return f"{month_names[m]}/{y}"

# ------------------------
# Título
# ------------------------
st.title("Tentativa de Reservas + Tendência + Taxa de Conversão")

# ------------------------
# Sidebar: UF + Período
# ------------------------
ufs = sorted(df["UF"].dropna().unique())
ufs_selected = st.sidebar.multiselect("Selecione os estados (UF)", ufs, default=ufs[:1])

start_date = st.sidebar.date_input("Data inicial", df["ds"].min())
end_date = st.sidebar.date_input("Data final", df["ds"].max())
horizon = st.sidebar.slider("Meses a projetar (para Tentativa de Reserva)", 1, 24, 12)

# Filtrar dados por UF(s) e período
df_uf = df[(df["UF"].isin(ufs_selected)) & (df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] <= pd.to_datetime(end_date))]

# ------------------------
# Feriados nacionais + férias escolares (para Prophet)
# ------------------------
feriados_nacionais = pd.DataFrame({
    'holiday': ['Confraternização', 'Carnaval', 'Paixão de Cristo', 'Tiradentes', 'Dia do Trabalho', 'Corpus Christi', 'Independência', 'Nossa Senhora Aparecida', 'Finados', 'Proclamação da República'],
    'ds': pd.to_datetime(['2023-01-01','2023-02-20','2023-04-07','2023-04-21','2023-05-01', '2023-06-08','2023-09-07','2023-10-12','2023-11-02','2023-11-15']),
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
# Projeção por UF (pré-calc na inicialização) + cache
# ------------------------
def compute_projection_all(all_uf, horizon, feriados, df):
    proj = {}
    monthly = {}
    for uf in all_uf:
        df_u = df[(df["UF"] == uf)][["ds","y"]].copy()
        if df_u.empty:
            proj[uf] = 0.0
            monthly[uf] = pd.DataFrame(columns=['ds','yhat'])
            continue
        df_u["y"] = df_u["y"].astype(float)
        model = Prophet(holidays=feriados)
        model.fit(df_u)
        last_date = df_u["ds"].max()
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] > last_date]
        yhat_2025 = forecast_future[forecast_future["ds"].dt.year == 2025]["yhat"].sum() if not forecast_future.empty else 0.0
        proj[uf] = float(yhat_2025)
        monthly[uf] = forecast_future[forecast_future["ds"].dt.year == 2025][["ds","yhat"]].copy()
    return proj, monthly

# Projeção total por UF (pré-calc) com cache
all_ufs = sorted(df["UF"].dropna().unique())
proj_2025_by_all, monthly_2025_by_uf_all = compute_projection_all(all_ufs, horizon, feriados, df)

# ------------------------
# Histórico por UF + Projeção por UF
# ------------------------
st.subheader("Histórico e Projeção por UF (selecionadas)")
for uf in ufs_selected:
    df_prophet = df[(df["UF"] == uf)][["ds","y","Tentativa de Reserva","Serviços Convertidos"]].copy().sort_values("ds")
    if df_prophet.empty:
        continue

    # Garantir que y seja inteiro (para a projeção)
    if "y" in df_prophet.columns:
        df_prophet["y"] = df_prophet["y"].astype(int)

    # Construir modelo e previsão (mantém a lógica original)
    model = Prophet(holidays=feriados)
    model.fit(df_prophet.rename(columns={"ds":"ds","y":"y"}))
    last_date = df_prophet["ds"].max()
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > last_date]

    # Reforçar inteiros no forecast
    if not forecast_future.empty:
        forecast_future["yhat"] = forecast_future["yhat"].round().astype(int)
        forecast_future["yhat_lower"] = forecast_future["yhat_lower"].round().astype(int)
        forecast_future["yhat_upper"] = forecast_future["yhat_upper"].round().astype(int)

    # Gráfico único com histórico + projeção (2 traces) + banda
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=df_prophet["ds"],
        y=df_prophet["y"],
        mode="lines",
        name="Histórico",
        hovertemplate="Data: %{x|%b/%Y}<br>Reservas: %{y:.0f}"
    ))

    # Projeção 2025
    if not forecast_future.empty:
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat"],
            mode="lines",
            name="Projeção 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Projeção 2025: %{y:.0f}"
        ))
        # Banda de incerteza ( Superior e Inferior )
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_upper"],
            mode="lines",
            line=dict(dash="dot", color="gray"),
            name="Intervalo Superior 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Superior: %{y:.0f}"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_lower"],
            mode="lines",
            line=dict(dash="dot", color="gray"),
            name="Intervalo Inferior 2025",
            hovertemplate="Data: %{x|%b/%Y}<br>Inferior: %{y:.0f}",
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.15)"
        ))

    fig.update_layout(
        title=f"Histórico + Projeção - {uf}",
        xaxis_title="Data",
        yaxis_title="Reservas",
        yaxis=dict(tickformat="d"),  # inteiros no eixo
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Taxa de Conversão (Histórico) — nova seção para este UF
    if {"Tentativa de Reserva","Serviços Convertidos"}.issubset(df_prophet.columns):
        df_prophet_conv = df_prophet[["ds","Tentativa de Reserva","Serviços Convertidos"]].copy()
        df_prophet_conv["tentativas"] = pd.to_numeric(df_prophet_conv["Tentativa de Reserva"], errors="coerce").fillna(0.0)
        df_prophet_conv["convertidos"] = pd.to_numeric(df_prophet_conv["Serviços Convertidos"], errors="coerce").fillna(0.0)
        df_prophet_conv["conversao_pct"] = np.where(
            df_prophet_conv["tentativas"] > 0,
            (df_prophet_conv["convertidos"] / df_prophet_conv["tentativas"]) * 100.0,
            0.0
        )

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=df_prophet_conv["ds"],
            y=df_prophet_conv["conversao_pct"],
            mode="lines+markers",
            name="Taxa de Conversão (%)",
            hovertemplate="Data: %{x|%b/%Y}<br>Conversão: %{y:.2f}%"
        ))
        fig_conv.update_layout(
            title=f"Taxa de Conversão (Serviços Convertidos / Tentativas) - {uf}",
            xaxis_title="Data",
            yaxis_title="Conversão (%)",
            yaxis=dict(range=[0, 100], tickformat=".1f"),
            hovermode="closest"
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        avg_conv = float(df_prophet_conv["conversao_pct"].mean()) if not df_prophet_conv["conversao_pct"].empty else 0.0
        st.metric(label="Conversão média (%) (período exibido)", value=f"{avg_conv:.2f}%")

        # Tabela resumida de conversão por mês
        conv_table = df_prophet_conv[["ds","Tentativa de Reserva","Serviços Convertidos","conversao_pct"]].copy()
        conv_table.rename(columns={
            "ds":"Data",
            "Tentativa de Reserva":"Tentativas",
            "Serviços Convertidos":"Convertidos",
            "conversao_pct":"Conversão (%)"
        }, inplace=True)
        conv_table["Data"] = conv_table["Data"].dt.strftime("%b/%Y")
        conv_table["Conversão (%)"] = conv_table["Conversão (%)"].round(2)
        st.dataframe(conv_table[["Data","Tentativas","Convertidos","Conversão (%)"]], height=260)

        # Exportar conversão histórica (CSV)
        conv_export = conv_table[["Data","Tentativas","Convertidos","Conversão (%)"]].rename(columns={"Data":"Mês/Ano"})
        st.download_button(
            label="Exportar Taxa de Conversão (CSV)",
            data=conv_export.to_csv(index=False).encode('utf-8'),
            file_name=f"conversao_{uf}.csv",
            mime="text/csv"
        )
    else:
        st.write("Dados de Conversão não encontrados para esta UF (faltam as colunas 'Tentativa de Reserva' e/ou 'Serviços Convertidos').")

    # Resumo da UF (executado)
    total_2023_uf = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2023)].get("Tentativa de Reserva").fillna(0).sum()) if "Tentativa de Reserva" in df.columns else 0
    total_2024_uf = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2024)].get("Tentativa de Reserva").fillna(0).sum()) if "Tentativa de Reserva" in df.columns else 0
    proj_uf_2025 = int(proj_2025_by_all.get(uf, 0.0))
    st.markdown(f"Resumo da UF {uf}:")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(label="2023 (Executado)", value=str(total_2023_uf))
    with colB:
        st.metric(label="2024 (Executado)", value=str(total_2024_uf))
    with colC:
        st.metric(label="Projeção 2025 (UF)", value=str(proj_uf_2025))

# ------------------------
# Explicação do modelo
# ------------------------
with st.expander("ℹ️ Como funciona a projeção"):
    st.markdown("""
    O modelo de projeção utiliza o Facebook Prophet, uma ferramenta de previsão de séries temporais.
    - Lida com tendência e sazonalidade
    - Considera feriados/eventos especiais
    - Útil para dados com sazonalidade mensal/annual
    """)
