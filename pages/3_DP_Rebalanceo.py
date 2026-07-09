import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cdist
import time
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
            if w_optimo.sum() > 0:
                w_optimo /= w_optimo.sum()

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

# Leemos la frecuencia global elegida
frecuencia_sel = st.session_state.get('frecuencia', 'Mensual')

with st.spinner(f'Cargando serie de tiempo y generando estados ({frecuencia_sel})...'):
    precios = cargar_datos(TICKERS, FECHA_INICIO, FECHA_FIN)
    TICKERS = list(precios.columns)
    
    # Recargar pesos óptimos alineados
    w_optimo = np.ones(len(TICKERS)) / len(TICKERS)
    if os.path.exists("resultados_m1.json"):
        with open("resultados_m1.json", "r", encoding="utf-8") as f:
            datos_m1 = json.load(f)
            if "markowitz_max_sharpe" in datos_m1:
                pesos_dict = datos_m1["markowitz_max_sharpe"]["pesos"]
                w_optimo = np.array([pesos_dict.get(t, 1/len(TICKERS)) for t in TICKERS], dtype=float)
                if w_optimo.sum() > 0:
                    w_optimo /= w_optimo.sum()

    # 1. Mapeo dinámico de frecuencia
    mapa_frecuencias = {"Semanal": "W-FRI", "Mensual": "ME", "Trimestral": "QE"}
    codigo_freq = mapa_frecuencias.get(frecuencia_sel, "ME")

    # 2. Resampleo adaptativo
    precios_resampleados = precios.resample(codigo_freq).last()
    retornos_resampleados = precios_resampleados.pct_change().dropna()
    
    # 3. Tomar exactamente los últimos T_PERIODOS para la simulación
    if len(retornos_resampleados) > T_PERIODOS:
        retornos_sim = retornos_resampleados.iloc[-T_PERIODOS:]
        fechas_sim = retornos_sim.index
    else:
        retornos_sim = retornos_resampleados
        fechas_sim = retornos_sim.index
        T_PERIODOS = len(retornos_sim)

# Calcular retornos diarios, mu y Sigma para gráficos
retornos_diarios = pd.DataFrame(np.log(precios / precios.shift(1))).dropna()
mu = retornos_diarios.mean().to_numpy(dtype=float) * 252
Sigma = retornos_diarios.cov().to_numpy(dtype=float) * 252

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

# --- PANEL DE CÓDIGO VISUAL (Ecuación de Bellman) ---
CODIGO_DP = [
    "def induccion_hacia_atras(S, T):",
    "    V = inicializar_matriz_ceros(T, M)",
    "    for t in reversed(range(T)):",
    "        # 1. Costo futuro de caer en estado j",
    "        costo_j = Subopt_cost + V[t+1]",
    "        # 2. Sumar costo de transición (TC)",
    "        Cost_matrix = TC_matrix + costo_j",
    "        # 3. Ecuación de Bellman (Minimizar)",
    "        V[t] = min(Cost_matrix, axis=1)",
    "        politica[t] = argmin(Cost_matrix, axis=1)",
    "    return V, politica"
]

def renderizar_codigo(linea_activa):
    html = "<div style='background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 14px; line-height: 1.5;'>"
    for i, linea in enumerate(CODIGO_DP):
        linea_format = linea.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
        if i == linea_activa:
            html += f"<div style='background-color: #062f4a; border-left: 3px solid #3794ff; padding-left: 5px; width: 100%;'>{linea_format}</div>"
        else:
            html += f"<div style='padding-left: 8px;'>{linea_format}</div>"
    html += "</div>"
    return html

# --- 4. INDUCCIÓN HACIA ATRÁS (ANIMADA) ---
st.subheader("Resolviendo la Ecuación de Bellman")

col_graf, col_cod = st.columns([2, 1])

with col_graf:
    grafico_placeholder = st.empty()
    metricas_placeholder = st.empty()

with col_cod:
    st.markdown("**Algoritmo DP (Vivo):**")
    codigo_placeholder = st.empty()
    codigo_placeholder.markdown(renderizar_codigo(-1), unsafe_allow_html=True)

if st.button("🚀 Calcular Política Óptima de Rebalanceo", type="primary", use_container_width=True):
    with st.spinner('Construyendo la matriz de costos paso a paso...'):
        codigo_placeholder.markdown(renderizar_codigo(0), unsafe_allow_html=True)
        time.sleep(0.3)
        
        # Inicialización
        codigo_placeholder.markdown(renderizar_codigo(1), unsafe_allow_html=True)
        TC_matrix = LAMBDA_TC * cdist(S, S, metric='cityblock')
        Subopt_cost = np.sum((S - w_optimo)**2, axis=1)
        
        V = np.zeros((T_PERIODOS + 1, M))
        politica = np.zeros((T_PERIODOS, M), dtype=int)
        time.sleep(0.3)
        
        # --- BUCLE DE INDUCCIÓN HACIA ATRÁS ---
        for t in range(T_PERIODOS - 1, -1, -1):
            codigo_placeholder.markdown(renderizar_codigo(2), unsafe_allow_html=True)
            
            codigo_placeholder.markdown(renderizar_codigo(4), unsafe_allow_html=True)
            costo_j = Subopt_cost + V[t+1]
            time.sleep(0.1)
            
            codigo_placeholder.markdown(renderizar_codigo(6), unsafe_allow_html=True)
            Cost_matrix = TC_matrix + costo_j
            
            codigo_placeholder.markdown(renderizar_codigo(8), unsafe_allow_html=True)
            mejor_accion = np.argmin(Cost_matrix, axis=1)
            V[t] = Cost_matrix[np.arange(M), mejor_accion]
            
            codigo_placeholder.markdown(renderizar_codigo(9), unsafe_allow_html=True)
            politica[t] = mejor_accion
            
            # Animación del Heatmap
            fig_heatmap = px.imshow(
                V, 
                labels=dict(x="ID del Estado (Combinación de Pesos)", y="Periodo (t)", color="Costo Acumulado"), 
                title=f"Matriz de Costos de Bellman (Calculando periodo t={t})", 
                color_continuous_scale="Viridis", 
                aspect="auto"
            )
            fig_heatmap.update_layout(yaxis=dict(autorange="reversed"))
            
            # Línea trazadora que muestra el progreso del algoritmo hacia atrás
            fig_heatmap.add_hline(y=t, line_width=2, line_dash="dash", line_color="red")
            
            grafico_placeholder.plotly_chart(fig_heatmap, use_container_width=True)
            metricas_placeholder.info(f"🧮 Optimizando decisiones para el mes {t} considerando el impacto en el mes {T_PERIODOS}")
            time.sleep(0.15) # Ajusta la velocidad de la animación aquí

        codigo_placeholder.markdown(renderizar_codigo(10), unsafe_allow_html=True)
        metricas_placeholder.success("¡Matriz de Costos completada! Simulando ruta óptima hacia adelante...")
        time.sleep(1)

        # --- 5. SIMULACIÓN HACIA ADELANTE (FORWARD SIMULATION) ---
        # Obtenemos TODOS los periodos del histórico para que sea comparable con M1, M2 y M4
        precios_res_full = precios.resample(codigo_freq).last()
        retornos_sim_full = precios_res_full.pct_change().dropna()
        T_full = len(retornos_sim_full)
        fechas_sim_full = np.append([precios_res_full.index[0]], retornos_sim_full.index)
        
        W_bh = np.zeros(T_full + 1)
        W_sr = np.zeros(T_full + 1)
        W_dp = np.zeros(T_full + 1)
        
        W_bh[0] = W_sr[0] = W_dp[0] = CAPITAL_INICIAL
        
        w_bh = w_optimo.copy()
        w_sr = w_optimo.copy()
        w_dp = w_optimo.copy()
        
        rebalanceos_dp = []
        
        for t in range(T_full):
            ret = np.array(retornos_sim_full.iloc[t].values, dtype=float)
            
            # 1. Estrategia Buy & Hold
            ret_bh = np.dot(w_bh, ret)
            W_bh[t+1] = W_bh[t] * (1 + ret_bh)
            w_bh = w_bh * (1 + ret) / (1 + ret_bh)
            
            # 2. Estrategia Siempre Rebalancear
            if t > 0:
                tc_sr = LAMBDA_TC * np.sum(np.abs(w_sr - w_optimo))
                capital_post_tc_sr = W_sr[t] * (1 - tc_sr)
                ret_sr = np.dot(w_optimo, ret)
                W_sr[t+1] = capital_post_tc_sr * (1 + ret_sr)
                w_sr = w_optimo * (1 + ret) / (1 + ret_sr)
            else:
                ret_sr = np.dot(w_sr, ret)
                W_sr[t+1] = W_sr[t] * (1 + ret_sr)
                w_sr = w_sr * (1 + ret) / (1 + ret_sr)
            
            # 3. Estrategia Programación Dinámica
            if t > 0:
                # Buscar el estado más cercano en nuestra grilla
                idx_s = int(np.argmin(np.sum(np.abs(S - w_dp), axis=1)))
                idx_a = int(politica[min(t, T_PERIODOS-1), idx_s]) # Usamos la política calculada
                w_dp_nuevo = S[idx_a]
                
                if idx_s != idx_a:
                    rebalanceos_dp.append(t)
                    tc_dp = LAMBDA_TC * np.sum(np.abs(w_dp - w_dp_nuevo))
                    capital_post_tc_dp = W_dp[t] * (1 - tc_dp)
                    ret_dp = np.dot(w_dp_nuevo, ret)
                    W_dp[t+1] = capital_post_tc_dp * (1 + ret_dp)
                    w_dp = w_dp_nuevo * (1 + ret) / (1 + ret_dp)
                else:
                    ret_dp = np.dot(w_dp, ret)
                    W_dp[t+1] = W_dp[t] * (1 + ret_dp)
                    w_dp = w_dp * (1 + ret) / (1 + ret_dp)
            else:
                ret_dp = np.dot(w_dp, ret)
                W_dp[t+1] = W_dp[t] * (1 + ret_dp)
                w_dp = w_dp * (1 + ret) / (1 + ret_dp)

        # --- 6. VISUALIZACIÓN DE RESULTADOS FINALES ---
        st.divider()
        st.subheader(f"Rendimiento de Estrategias (Histórico Completo - Frecuencia {frecuencia_sel})")
        
        fig_riqueza = go.Figure()
        fig_riqueza.add_trace(go.Scatter(x=fechas_sim_full, y=W_dp, mode='lines', name=f'DP Optimizado (${W_dp[-1]:,.2f})', line=dict(color='green', width=3)))
        fig_riqueza.add_trace(go.Scatter(x=fechas_sim_full, y=W_sr, mode='lines', name=f'Siempre Rebalancear (${W_sr[-1]:,.2f})', line=dict(color='red', dash='dash')))
        fig_riqueza.add_trace(go.Scatter(x=fechas_sim_full, y=W_bh, mode='lines', name=f'Buy & Hold (${W_bh[-1]:,.2f})', line=dict(color='gray', dash='dot')))
        
        for t_reb in rebalanceos_dp:
            fig_riqueza.add_vline(x=fechas_sim_full[t_reb+1], line_width=1.5, line_dash="dash", line_color="purple", annotation_text="Rebalanceo", annotation_position="bottom right")

        fig_riqueza.update_layout(title="Comparativa de Crecimiento de Capital", xaxis_title="Fecha", yaxis_title="Capital (USD)", hovermode="x unified")
        st.plotly_chart(fig_riqueza, use_container_width=True)

        # Calcular drawdowns históricos para Módulos 3
        def calcular_drawdown(riqueza_path):
            cum_max = np.maximum.accumulate(riqueza_path)
            with np.errstate(divide='ignore', invalid='ignore'):
                dd = (riqueza_path - cum_max) / cum_max
                dd = np.nan_to_num(dd, nan=0.0)
            return dd * 100  # En porcentaje

        dd_dp = calcular_drawdown(W_dp)
        dd_sr = calcular_drawdown(W_sr)
        dd_bh = calcular_drawdown(W_bh)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=fechas_sim_full, y=dd_dp, mode='lines', name='Drawdown DP Optimizado', line=dict(color='green')))
        fig_dd.add_trace(go.Scatter(x=fechas_sim_full, y=dd_sr, mode='lines', name='Drawdown Siempre Rebalancear', line=dict(color='red', dash='dash')))
        fig_dd.add_trace(go.Scatter(x=fechas_sim_full, y=dd_bh, mode='lines', name='Drawdown Buy & Hold', line=dict(color='gray', dash='dot')))
        
        # Añadir líneas de rebalanceo en el gráfico de drawdown también para mayor contexto
        for t_reb in rebalanceos_dp:
            fig_dd.add_vline(x=fechas_sim_full[t_reb+1], line_width=1.0, line_dash="dash", line_color="purple")
            
        fig_dd.update_layout(
            title=f'Caídas Máximas de las Estrategias (Drawdown - Rebalanceo {frecuencia_sel})',
            xaxis_title='Fecha',
            yaxis_title='Drawdown (%)',
            hovermode="x unified"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # --- NUEVOS GRÁFICOS ADICIONALES (Misma lógica que Módulo 1 y Módulo 2) ---
        st.divider()
        st.subheader("📊 Análisis de Composición y Riesgo del Portafolio Objetivo")
        
        c1, c2 = st.columns(2)
        with c1:
            # Gráfico de Dona de Pesos Objetivos
            df_pesos = pd.DataFrame({'Ticker': TICKERS, 'Peso (%)': (w_optimo * 100).round(2)})
            df_pesos_filtrado = df_pesos[df_pesos['Peso (%)'] > 0.01]
            fig_dona = px.pie(
                df_pesos_filtrado, 
                values='Peso (%)', 
                names='Ticker', 
                title='Composición del Portafolio Objetivo (Pesos %)', 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_dona.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig_dona, use_container_width=True)
            
        with c2:
            # Gráfico de Barras Agrupadas: Retorno vs Volatilidad de activos
            df_activos = pd.DataFrame({
                'Ticker': TICKERS,
                'Retorno Anualizado (%)': (mu * 100).round(2),
                'Volatilidad Anualizada (%)': (np.sqrt(np.diag(Sigma)) * 100).round(2)
            })
            df_long = df_activos.melt(id_vars='Ticker', value_vars=['Retorno Anualizado (%)', 'Volatilidad Anualizada (%)'],
                                      var_name='Métrica', value_name='Valor (%)')
            fig_activos = px.bar(
                df_long,
                x='Ticker',
                y='Valor (%)',
                color='Métrica',
                barmode='group',
                title='Desempeño Individual de los Activos',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig_activos, use_container_width=True)

        # Mapa de Calor de Correlación en un expansor
        with st.expander("🔍 Ver Matriz de Correlación de los Activos"):
            correlaciones = retornos_diarios.corr()
            fig_heatmap = px.imshow(
                correlaciones,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                title="Matriz de Correlación de Activos (Retornos Diarios)"
            )
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
            "trayectoria_dp": W_dp.tolist(),
            "fechas_completas": [d.strftime("%Y-%m-%d") for d in fechas_sim_full],
            "grilla_estados": S.tolist(),
            "matriz_politica": politica.tolist()
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
