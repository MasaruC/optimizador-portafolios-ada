import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cdist
import itertools
import json
import os
import io

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Módulo 3: Programación Dinámica", page_icon="🧮", layout="wide")
st.title("🧮 Módulo 3: Rebalanceo Óptimo (Programación Dinámica)")
st.markdown("Uso de la **Ecuación de Bellman (Backward Induction)** para encontrar cuándo rebalancear considerando los costos de transacción.")

# Recuperar parámetros globales
tickers_input = st.session_state.get('tickers', 'FSM, VOLCABC1.LM, ABX.TO, BVN, BHP')
TICKERS = [t.strip() for t in tickers_input.split(',')]
FECHA_INICIO = st.session_state.get('fecha_inicio', '2015-01-01')
FECHA_FIN = st.session_state.get('fecha_fin', '2024-12-31')
CAPITAL_INICIAL = st.session_state.get('capital', 100000)

# Parámetros específicos de DP
LAMBDA_TC = float(st.session_state.get('dp_tc', 0.0010))
T_PERIODOS = int(st.session_state.get('dp_horizon', 12))
PASO_GRILLA = 0.10  # Salto de discretización según requerimiento

# Intentar cargar pesos óptimos de Markowitz (Módulo 1)
w_optimo = np.ones(len(TICKERS)) / len(TICKERS)  # Equiponderado por defecto
if os.path.exists("resultados_m1.json"):
    with open("resultados_m1.json", "r", encoding="utf-8") as f:
        datos_m1 = json.load(f)
        if "markowitz_max_sharpe" in datos_m1:
            pesos_dict = datos_m1["markowitz_max_sharpe"]["pesos"]
            w_optimo = np.array([pesos_dict.get(t, 1/len(TICKERS)) for t in TICKERS], dtype=float)

# --- 2. DESCARGA Y LIMPIEZA DE DATOS ---
@st.cache_data(show_spinner=False)
def cargar_datos(tickers, inicio, fin) -> pd.DataFrame:
    descarga = yf.download(tickers, start=inicio, end=fin, progress=False)
    if descarga is None or descarga.empty:
        return pd.DataFrame()
    df_raw = pd.DataFrame(descarga)
    datos = df_raw['Close'] if 'Close' in df_raw.columns else df_raw
    if isinstance(datos, pd.Series):
        datos = datos.to_frame(name=tickers[0])
    return datos.ffill().bfill()

with st.spinner('Cargando serie de tiempo y generando estados...'):
    precios = cargar_datos(TICKERS, FECHA_INICIO, FECHA_FIN)
    
    # Preparar rendimientos para los T últimos periodos (Mensuales)
    precios_mensuales = precios.resample('ME').last()
    retornos_mensuales = precios_mensuales.pct_change().dropna()
    
    # Tomar exactamente los últimos T_PERIODOS para la simulación
    if len(retornos_mensuales) > T_PERIODOS:
        retornos_sim = retornos_mensuales.iloc[-T_PERIODOS:]
        fechas_sim = retornos_sim.index
    else:
        retornos_sim = retornos_mensuales
        fechas_sim = retornos_sim.index
        T_PERIODOS = len(retornos_sim)

# --- 3. CREACIÓN DEL ESPACIO DE ESTADOS (GRILLA) ---
@st.cache_data(show_spinner=False)
def generar_grilla(n, paso):
    valores = np.arange(0, 1 + paso, paso)
    grilla = [p for p in itertools.product(valores, repeat=n) if np.isclose(sum(p), 1.0)]
    return np.array(grilla, dtype=float)

S = generar_grilla(len(TICKERS), PASO_GRILLA)
M = len(S)

col1, col2, col3 = st.columns(3)
col1.info(f"**Periodos a simular (T):** {T_PERIODOS} meses")
col2.info(f"**Costo Transaccional (λ):** {LAMBDA_TC}")
col3.info(f"**Estados Generados (Grilla):** {M}")

# --- 4. INDUCCIÓN HACIA ATRÁS (BELLMAN VECTORIZADO) ---
if st.button("🚀 Calcular Política Óptima de Rebalanceo", type="primary", use_container_width=True):
    with st.spinner('Resolviendo Ecuación de Bellman...'):
        # Costo de transacción estático entre cualquier estado i y j
        TC_matrix = LAMBDA_TC * cdist(S, S, metric='cityblock')
        
        # Costo de penalidad (distancia al portafolio óptimo de Markowitz)
        Subopt_cost = np.sum((S - w_optimo)**2, axis=1)
        
        # Matrices de Valor (V) y Política (Decisiones)
        V = np.zeros((T_PERIODOS + 1, M))
        politica = np.zeros((T_PERIODOS, M), dtype=int)
        
        # Backward Induction
        for t in range(T_PERIODOS - 1, -1, -1):
            # Costo = Transición + Suboptimalidad de caer en j + Valor futuro de j
            costo_j = Subopt_cost + V[t+1]
            Cost_matrix = TC_matrix + costo_j # Broadcasting
            
            mejor_accion = np.argmin(Cost_matrix, axis=1)
            V[t] = Cost_matrix[np.arange(M), mejor_accion]
            politica[t] = mejor_accion

        # --- 5. SIMULACIÓN HACIA ADELANTE (FORWARD SIMULATION) ---
        W_bh = np.zeros(T_PERIODOS + 1)
        W_sr = np.zeros(T_PERIODOS + 1)
        W_dp = np.zeros(T_PERIODOS + 1)
        
        W_bh[0] = W_sr[0] = W_dp[0] = CAPITAL_INICIAL
        
        w_bh = w_optimo.copy()
        w_sr = w_optimo.copy()
        w_dp = w_optimo.copy()
        
        rebalanceos_dp = []
        
        for t in range(T_PERIODOS):
            ret = np.array(retornos_sim.iloc[t].values, dtype=float)
            
            # 1. Estrategia Buy & Hold (Sin rebalancear)
            ret_bh = np.dot(w_bh, ret)
            W_bh[t+1] = W_bh[t] * (1 + ret_bh)
            w_bh = w_bh * (1 + ret) / (1 + ret_bh) # Deriva de pesos
            
            # 2. Estrategia Siempre Rebalancear (Forzar w_optimo cada mes)
            tc_sr = LAMBDA_TC * np.sum(np.abs(w_sr - w_optimo))
            capital_post_tc_sr = W_sr[t] * (1 - tc_sr)
            ret_sr = np.dot(w_optimo, ret)
            W_sr[t+1] = capital_post_tc_sr * (1 + ret_sr)
            w_sr = w_optimo * (1 + ret) / (1 + ret_sr)
            
            # 3. Estrategia Programación Dinámica (Inteligente)
            # Encontrar el estado actual en la grilla más cercano a w_dp
            idx_s = int(np.argmin(np.sum(np.abs(S - w_dp), axis=1)))
            idx_a = int(politica[t, idx_s])
            w_dp_nuevo = S[idx_a]
            
            if idx_s != idx_a:
                rebalanceos_dp.append(t)
                
            tc_dp = LAMBDA_TC * np.sum(np.abs(w_dp - w_dp_nuevo))
            capital_post_tc_dp = W_dp[t] * (1 - tc_dp)
            
            ret_dp = np.dot(w_dp_nuevo, ret)
            W_dp[t+1] = capital_post_tc_dp * (1 + ret_dp)
            w_dp = w_dp_nuevo * (1 + ret) / (1 + ret_dp)

        st.success("¡Optimización de Programación Dinámica Completada!")

        # --- 6. VISUALIZACIÓN DE RESULTADOS ---
        st.divider()
        st.subheader("Rendimiento de Estrategias (Últimos 12 meses)")
        
        fechas_plot = [fechas_sim[0] - pd.DateOffset(months=1)] + list(fechas_sim)
        
        fig_riqueza = go.Figure()
        fig_riqueza.add_trace(go.Scatter(x=fechas_plot, y=W_dp, mode='lines+markers', name=f'DP Optimizado (${W_dp[-1]:,.2f})', line=dict(color='green', width=3)))
        fig_riqueza.add_trace(go.Scatter(x=fechas_plot, y=W_sr, mode='lines', name=f'Siempre Rebalancear (${W_sr[-1]:,.2f})', line=dict(color='red', dash='dash')))
        fig_riqueza.add_trace(go.Scatter(x=fechas_plot, y=W_bh, mode='lines', name=f'Buy & Hold (${W_bh[-1]:,.2f})', line=dict(color='gray', dash='dot')))
        
        # Marcar los momentos exactos de rebalanceo sugeridos por DP
        for t_reb in rebalanceos_dp:
            fig_riqueza.add_vline(x=fechas_plot[t_reb+1], line_width=1.5, line_dash="dash", line_color="purple", annotation_text="Rebalanceo", annotation_position="bottom right")

        fig_riqueza.update_layout(title="Simulación de Riqueza con Costos de Transacción", xaxis_title="Fecha", yaxis_title="Capital (USD)", hovermode="x unified")
        st.plotly_chart(fig_riqueza, use_container_width=True)

        st.divider()
        
        # Heatmap de la Matriz V (Costos Futuros Óptimos)
        st.subheader("Mapa de Calor (Heatmap): Costos Óptimos J*(t,s)")
        fig_heatmap = px.imshow(V, labels=dict(x="ID del Estado (Combinación de Pesos)", y="Periodo de Tiempo (Meses)", color="Costo Acumulado"), 
                                title="Matriz de Costos de Bellman", color_continuous_scale="Viridis", aspect="auto")
        fig_heatmap.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # --- 7. EXPORTACIÓN DE RESULTADOS ---
        metrics_m3 = {
            "parametros_dp": {
                "lambda_tc": float(LAMBDA_TC),
                "periodos_t": int(T_PERIODOS),
                "paso_grilla": float(PASO_GRILLA),
                "total_estados": int(M)
            },
            "riqueza_final": {
                "buy_and_hold": float(W_bh[-1]),
                "siempre_rebalancear": float(W_sr[-1]),
                "dp_optimizado": float(W_dp[-1])
            },
            "timeline_rebalanceos_dp": [int(x) for x in rebalanceos_dp],
            "trayectoria_dp": W_dp.tolist()
        }
        
        with open("resultados_m3.json", "w", encoding="utf-8") as f:
            json.dump(metrics_m3, f, ensure_ascii=False, indent=2)

        buffer = io.BytesIO()
        df_export = pd.DataFrame(V)
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, header=False, sheet_name='Costos_DP')
        buffer.seek(0)

        st.download_button(
            label="📥 Descargar Matriz de Costos Bellman (Excel)",
            data=buffer,
            file_name="matriz_dp_costos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👈 Presiona el botón para ejecutar la inducción hacia atrás de Bellman.")