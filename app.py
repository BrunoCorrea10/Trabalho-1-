import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Hidráulico - Trabalho 1",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------------------
# CÁLCULO FÍSICO DO SISTEMA HIDRÁULICO
# -------------------------------------------------------------
P8 = 532000 + 1e5
ro, g, Efic = 1000, 9.8, 0.7
Ds = 0.15
Mi, k, Patm, deltaZ = 1e-3, 0.15e-3, 1e5, 7.5
Ls, Lr, TempoOpr = 12, 36, 3600
Ks_total = 15 + 0.9 + 10
Kr_total = 0.5 + 10 + 0.9 + 1


@st.cache_data
def simular_sistema(dr_val, pkwh_val):
    vazoes_ls = np.linspace(10, 50, 41)
    res = []
    for Q_ls in vazoes_ls:
        Q = Q_ls / 1000.0

        # Sucção
        VelSuc = Q / (np.pi * Ds ** 2 / 4)
        ReSuc = VelSuc * ro * Ds / Mi
        f_s = scipy.optimize.fsolve(lambda x: (np.sqrt(10) / 10) ** x - (2.51 / ReSuc) * x - (1 / (Ds / k)) / 3.7, 0.1)[0]
        PcsTotal = ((1 / (f_s ** 2)) * (Ls / Ds) + Ks_total) * (VelSuc ** 2) / (2 * g)

        # Recalque
        VelReq = Q / (np.pi * dr_val ** 2 / 4)
        ReReq = VelReq * ro * dr_val / Mi
        f_r = scipy.optimize.fsolve(lambda x: (np.sqrt(10) / 10) ** x - (2.51 / ReReq) * x - (1 / (dr_val / k)) / 3.7, 0.1)[0]
        PcrTotal = ((1 / (f_r ** 2)) * (Lr / dr_val) + Kr_total) * (VelReq ** 2) / (2 * g)

        # Carga, Potência e Consumo
        EBomba = (P8 - Patm) / (ro * g) + deltaZ + (PcsTotal + PcrTotal)
        Pot_W = ro * g * Q * EBomba / Efic
        Pot_kW = Pot_W / 1000.0
        ConFin = (pkwh_val * Pot_W / 3.6e6) * TempoOpr

        res.append({
            'Vazao_Ls': Q_ls,
            'Velocidade_Req_ms': VelReq,
            'PerdaCarga_Req_m': PcrTotal,
            'Head_Bomba_m': EBomba,
            'Potencia_kW': Pot_kW,
            'Consumo_RShora': ConFin
        })
    return pd.DataFrame(res)


# -------------------------------------------------------------
# BARRA LATERAL (CONTROLES INTERATIVOS)
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/pipe.png", width=70)
st.sidebar.title("Painel de Controle")
st.sidebar.markdown("---")

dr_cm = st.sidebar.slider(
    "Diâmetro de Recalque (Dr em cm):",
    min_value=8.0,
    max_value=20.0,
    value=15.0,
    step=0.5
)

tarifa_kwh = st.sidebar.number_input(
    "Tarifa de Energia (R$/kWh):",
    value=0.27,
    step=0.01
)

vazao_analise = st.sidebar.selectbox(
    "Vazão para Destaque de Ponto (L/s):",
    options=[20.0, 30.0, 40.0],
    index=2
)

# Rodar simulações (Projeto Atual vs Projeto Original de 10 cm)
df_atual = simular_sistema(dr_cm / 100.0, tarifa_kwh)
df_orig = simular_sistema(0.10, tarifa_kwh)

p_atual = df_atual[df_atual['Vazao_Ls'] == vazao_analise].iloc[0]
p_orig = df_orig[df_orig['Vazao_Ls'] == vazao_analise].iloc[0]

# -------------------------------------------------------------
# CABEÇALHO PRINCIPAL
# -------------------------------------------------------------
st.title("⚡ Dashboard: Cálculo do consumo financeiro e energético de uma bomba")
st.caption("Trabalho 1 - BRUNO CORREA DA SILVA e BRAYAN BECHER SCORCATO")
st.markdown("---")

# -------------------------------------------------------------
# PAINEL DE KPIs SUPERIORES
# -------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

delta_vel = p_atual['Velocidade_Req_ms'] - p_orig['Velocidade_Req_ms']
delta_pot = p_atual['Potencia_kW'] - p_orig['Potencia_kW']
delta_custo = p_atual['Consumo_RShora'] - p_orig['Consumo_RShora']

status_norma = "Em Norma" if p_atual['Velocidade_Req_ms'] <= 2.5 else "Fora da Norma"

kpi1.metric(
    label=f"Velocidade Recalque ({vazao_analise:.0f} L/s)",
    value=f"{p_atual['Velocidade_Req_ms']:.2f} m/s",
    delta=f"{delta_vel:.2f} m/s",
    delta_color="inverse"
)

kpi2.metric(
    label="Status de Norma Técnica",
    value=status_norma
)

kpi3.metric(
    label=f"Potência Elétrica ({vazao_analise:.0f} L/s)",
    value=f"{p_atual['Potencia_kW']:.1f} kW",
    delta=f"{delta_pot:.1f} kW",
    delta_color="inverse"
)

kpi4.metric(
    label=f"Custo Operacional ({vazao_analise:.0f} L/s)",
    value=f"R$ {p_atual['Consumo_RShora']:.2f} / h",
    delta=f"R$ {delta_custo:.2f} / h",
    delta_color="inverse"
)

st.markdown("---")

# -------------------------------------------------------------
# NAVEGAÇÃO POR ABAS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Gráficos do Sistema (Item 1)",
    "📈 Ponto de Operação (H × Q)",
    "🔍 Minerador de Dados (Item 4)",
    "🧪 Ciência de Dados (Item 5)"
])

# --- ABA 1: GRÁFICOS DO SISTEMA ---
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Velocidade no Recalque vs. Limite de Norma")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(df_atual['Vazao_Ls'], df_atual['Velocidade_Req_ms'], color='#2E7D32', linewidth=2.5,
                 label=f'Dr = {dr_cm:.1f} cm (Selecionado)')
        ax1.plot(df_orig['Vazao_Ls'], df_orig['Velocidade_Req_ms'], color='#D9534F', linestyle=':',
                 label='Dr = 10.0 cm (Original)')
        ax1.axhline(y=2.5, color='#E67E22', linestyle='--', label='Limite Recomendado (2.5 m/s)')
        ax1.set_xlabel("Vazão (L/s)")
        ax1.set_ylabel("Velocidade (m/s)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()
        st.pyplot(fig1)

    with col_right:
        st.subheader("2. Comparativo de Custo Operacional (R$/hora)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(df_atual['Vazao_Ls'], df_atual['Consumo_RShora'], color='#2E7D32', linewidth=2.5,
                 label=f'Dr = {dr_cm:.1f} cm')
        ax2.plot(df_orig['Vazao_Ls'], df_orig['Consumo_RShora'], color='#D9534F', linewidth=2, label='Dr = 10.0 cm')
        ax2.fill_between(df_atual['Vazao_Ls'], df_orig['Consumo_RShora'], df_atual['Consumo_RShora'], color='#2E7D32',
                         alpha=0.15, label='Economia Retida')
        ax2.set_xlabel("Vazão (L/s)")
        ax2.set_ylabel("Custo (R$/h)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)

# --- ABA 2: PONTO DE OPERAÇÃO (H x Q) ---
with tab2:
    st.subheader("Ponto de Operação do Sistema ($Q_0, H_0$)")
    st.markdown(
        "Este gráfico ilustra o cruzamento entre a **Curva Característica da Bomba** e a "
        "**Curva de Resistência do Sistema** para os diâmetros de $10\\text{ cm}$ e $15\\text{ cm}$."
    )

    q_plot = np.linspace(0.1, 60, 150)

    H_shutoff = 75.0
    a_bomba = 0.0125
    H_bomba = H_shutoff - a_bomba * (q_plot ** 2)

    H_estatica = 15.0
    K_10 = 0.032
    K_15 = 0.0063

    H_sistema_10 = H_estatica + K_10 * (q_plot ** 2)
    H_sistema_15 = H_estatica + K_15 * (q_plot ** 2)

    fig_hq, ax_hq = plt.subplots(figsize=(9, 5))

    ax_hq.plot(q_plot, H_bomba, color='#1f77b4', linewidth=2.5, label='Curva da Bomba $H(Q)$')
    ax_hq.plot(q_plot, H_sistema_10, color='#d62728', linestyle='--', linewidth=2, label='Sistema ($D_r = 10\\text{ cm}$)')
    ax_hq.plot(q_plot, H_sistema_15, color='#2ca02c', linewidth=2.5, label='Sistema ($D_r = 15\\text{ cm}$ - Selecionado)')

    Q0_10 = np.sqrt((H_shutoff - H_estatica) / (a_bomba + K_10))
    H0_10 = H_shutoff - a_bomba * (Q0_10 ** 2)

    Q0_15 = np.sqrt((H_shutoff - H_estatica) / (a_bomba + K_15))
    H0_15 = H_shutoff - a_bomba * (Q0_15 ** 2)

    ax_hq.plot(Q0_10, H0_10, 'ro', markersize=8)
    ax_hq.annotate(f'Ponto 10 cm\n({Q0_10:.1f} L/s, {H0_10:.1f} m)',
                    xy=(Q0_10, H0_10), xytext=(Q0_10 + 2, H0_10 + 5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    ax_hq.plot(Q0_15, H0_15, 'go', markersize=8)
    ax_hq.annotate(f'Ponto 15 cm\n({Q0_15:.1f} L/s, {H0_15:.1f} m)',
                    xy=(Q0_15, H0_15), xytext=(Q0_15 - 12, H0_15 - 12),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6))

    ax_hq.set_xlabel('Vazão $Q$ (L/s)', fontsize=11)
    ax_hq.set_ylabel('Altura Manométrica $H$ (m)', fontsize=11)
    ax_hq.set_title('Interseção Head × Vazão: Determinação do Ponto de Operação', fontsize=12, fontweight='bold')
    ax_hq.set_ylim(0, 90)
    ax_hq.grid(True, linestyle=':', alpha=0.6)
    ax_hq.legend(loc='upper right')

    st.pyplot(fig_hq)

    st.info(
        "💡 **Leitura de Engenharia:** Note como o aumento do diâmetro de $10\\text{ cm}$ para $15\\text{ cm}$ "
        "reduz a inclinação da curva de resistência do sistema (menos atrito). Isso move o ponto de operação "
        "para uma **altura manométrica menor ($H_0$)**, reduzindo a potência exigida da bomba e gerando a "
        "economia financeira por hora observada no painel principal."
    )

# --- ABA 3: MINERADOR DE DADOS ---
with tab3:
    st.subheader("Minerador Dinâmico de Planilha / Banco de Dados")
    st.write("Filtre em tempo real a tabela de 41 pontos gerada pela simulação:")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        vazao_min = st.slider("Vazão Mínima (L/s):", 10.0, 50.0, 30.0)
    with col_f2:
        custo_max = st.slider("Custo Operacional Máximo (R$/h):", 2.0, 20.0, 10.0)

    df_filtrado = df_atual[(df_atual['Vazao_Ls'] >= vazao_min) & (df_atual['Consumo_RShora'] <= custo_max)]

    st.success(f"Encontrados **{len(df_filtrado)}** pontos operacionais dentro dos critérios estabelecidos.")
    st.dataframe(df_filtrado.style.format({
        'Vazao_Ls': '{:.1f}',
        'Velocidade_Req_ms': '{:.2f} m/s',
        'PerdaCarga_Req_m': '{:.2f} m',
        'Head_Bomba_m': '{:.2f} m',
        'Potencia_kW': '{:.2f} kW',
        'Consumo_RShora': 'R$ {:.2f}'
    }), use_container_width=True)

# --- ABA 4: CIÊNCIA DE DADOS ---
with tab4:
    st.subheader("Análise de Correlação e Dispersão Multivariada")
    col_cd1, col_cd2 = st.columns(2)

    with col_cd1:
        st.markdown("**Mapa de Calor de Correlação (HeatMap)**")
        fig_hm, ax_hm = plt.subplots(figsize=(5, 4))
        sns.heatmap(df_atual.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=ax_hm)
        st.pyplot(fig_hm)

    with col_cd2:
        st.markdown("**Resumo Estatístico das Variáveis**")
        st.dataframe(df_atual.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True)
