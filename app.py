# ------------------------
# Explicação do Modelo
# ------------------------
with st.expander("ℹ️ Como funciona a projeção"):
    st.markdown("""
    O modelo de projeção utiliza o **Facebook Prophet**, uma ferramenta de previsão de séries temporais desenvolvida para lidar com dados que apresentam:
    - **Tendência** (crescimento ou queda ao longo do tempo)  
    - **Sazonalidade anual/mensal/semanal**  
    - **Impacto de feriados e eventos especiais** (no caso, feriados nacionais e férias escolares foram incluídos)  

    O Prophet é um modelo **aditivo**, no qual a série é decomposta em:
    - Tendência  
    - Sazonalidade  
    - Efeitos de feriados  

    Ele é robusto para dados faltantes, mudanças de tendência e funciona bem em cenários de negócio.
    """)

# ------------------------
# Ranking Geral de UFs
# ------------------------
st.subheader("📊 Ranking Geral de UFs")

ranking_data = []
for uf in all_ufs:
    total_2023 = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2023)]['y'].sum())
    total_2024 = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2024)]['y'].sum())
    total_2025_real = int(df[(df["UF"] == uf) & (df["ds"].dt.year == 2025)]['y'].sum())
    proj_2025 = int(proj_2025_by_all.get(uf, 0.0))

    total_2025 = total_2025_real + proj_2025  # realizado até agora + projetado

    diff_25_24 = total_2025 - total_2024
    diff_25_23 = total_2025 - total_2023

    ranking_data.append({
        "UF": uf,
        "2023 (Executado)": total_2023,
        "2024 (Executado)": total_2024,
        "2025 (Realizado + Projetado)": total_2025,
        "Δ 2025-2024": diff_25_24,
        "Δ 2025-2023": diff_25_23
    })

df_ranking = pd.DataFrame(ranking_data)

# Ordenar pelo maior crescimento em relação a 2024
df_ranking = df_ranking.sort_values(by="Δ 2025-2024", ascending=False)

st.dataframe(df_ranking)

# Gráfico de barras com as diferenças
fig_rank = go.Figure()
fig_rank.add_trace(go.Bar(
    x=df_ranking["UF"],
    y=df_ranking["Δ 2025-2024"],
    name="2025 - 2024",
    marker_color="steelblue"
))
fig_rank.add_trace(go.Bar(
    x=df_ranking["UF"],
    y=df_ranking["Δ 2025-2023"],
    name="2025 - 2023",
    marker_color="darkorange"
))
fig_rank.update_layout(
    barmode="group",
    title="Comparativo de Crescimento por UF",
    xaxis_title="UF",
    yaxis_title="Diferença de Reservas"
)
st.plotly_chart(fig_rank, use_container_width=True)
