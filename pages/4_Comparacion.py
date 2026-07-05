import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import os
import io

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Módulo 4: Comparación Cruzada", page_icon="🏆", layout="wide")
st.title("🏆 Módulo 4: Panel de Comparación Cruzada de Estrategias")
st.markdown("Consolidación y evaluación de rendimiento de las 7 estrategias financieras implementadas en el sistema.")

# Verificar la existencia de los archivos de entrada indispensables
archivos_requeridos = ["resultados_m1.json", "resultados_m2.json", "resultados_m3.json"]
missing_files = [f for f in archivos_requeridos if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Faltan los siguientes archivos de resultados: {', '.join(missing_files)}")
    st.info("💡 Por favor, asegúrate de entrar y ejecutar el análisis en el **Módulo 1**, **Módulo 2** y **Módulo 3** previamente para generar los datos necesarios.")
    st.stop()

# --- 2. CARGA DE RESULTADOS PREVIOS (JSON) ---
with open("resultados_m1.json", "r", encoding="utf-8") as f:
    res_m1 = json.load(f)
with open("resultados_m2.json", "r", encoding="utf-8") as f:
    res_m2 = json.load(f)
with open("resultados_m3.json", "r", encoding="utf-8") as f:
    res_m3 = json.load(f)

# Recuperar parámetros y vectores de pesos
TICKERS = res_m1["tickers"]
N = len(TICKERS)
FECHA_INICIO = st.session_state.get('fecha_inicio', '2015-01-01')
FECHA_FIN = st.session_state.get('fecha_fin', '2024-12-31')
CAPITAL_INICIAL = st.session_state.get('capital', 100000)
LAMBDA_TC = res_m3["parametros_dp"]["lambda_tc"]
T_PERIODOS = res_m3["parametros_dp"]["periodos_t"]

# Reconstrucción de vectores de pesos desde los JSON
w_eq = np.ones(N) / N
w_mk = np.array([res_m1["markowitz_max_sharpe"]["pesos"].get(t, 1/N) for t in TICKERS], dtype=float)
w_ga = np.array([res_m2["nsga2_max_sharpe"]["pesos"].get(t, 1/N) for t in TICKERS], dtype=float)

# --- 3. DESCARGA Y PREPARACIÓN DE LA SERIE DE TIEMPO SINCRO ---
@st.cache_data(show_spinner=False)
def cargar_datos_dinamicos(tickers, inicio, fin, t_horizonte, freq_code) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    descarga = yf.download(tickers, start=inicio, end=fin, progress=False)
    if descarga is None or descarga.empty:
        return pd.DataFrame(), pd.DatetimeIndex([])
    
    df_raw = pd.DataFrame(descarga)
    datos = df_raw['Close'] if 'Close' in df_raw.columns else df_raw
    
    if isinstance(datos, pd.Series):
        datos = datos.to_frame(name=tickers[0])
    
    precios_resampleados = datos.ffill().bfill().resample(freq_code).last()
    retornos_resampleados = precios_resampleados.pct_change().dropna()
    
    # Filtrar exactamente para los últimos T periodos evaluados
    retornos_sim = retornos_resampleados.iloc[-t_horizonte:]
    fechas_sim = pd.DatetimeIndex(retornos_sim.index)
    
    return retornos_sim, fechas_sim

# Lógica dinámica de frecuencia y factores de anualización
frecuencia_sel = st.session_state.get('frecuencia', 'Mensual')
mapa_frecuencias = {"Semanal": "W-FRI", "Mensual": "ME", "Trimestral": "QE"}
codigo_freq = mapa_frecuencias.get(frecuencia_sel, "ME")
factor_anual = {"Semanal": 52, "Mensual": 12, "Trimestral": 4}.get(frecuencia_sel, 12)

with st.spinner(f'Sincronizando series de tiempo históricas ({frecuencia_sel})...'):
    retornos_periodo, fechas_backtest = cargar_datos_dinamicos(TICKERS, FECHA_INICIO, FECHA_FIN, T_PERIODOS, codigo_freq)

# --- 4. ENGINE DE SIMULACIÓN DE LAS 7 ESTRATEGIAS ---
n_periodos = len(retornos_periodo)

# Inicializar matrices de riqueza (Tamaño: n_periodos + 1)
W = np.zeros((7, n_periodos + 1))
W[:, 0] = CAPITAL_INICIAL

# Inicializar estados dinámicos de pesos (Buy & Hold)
w_dynamic_bh_eq = w_eq.copy()
w_dynamic_bh_mk = w_mk.copy()
w_dynamic_bh_ga = w_ga.copy()

# Simulación periodo a periodo coordinada
for t in range(n_periodos):
    ret_t = np.array(retornos_periodo.iloc[t].to_numpy(), dtype=float)
    
    # Estrategia 1: Buy & Hold Equiponderado
    r_1 = np.dot(w_dynamic_bh_eq, ret_t)
    W[0, t+1] = W[0, t] * (1 + r_1)
    w_dynamic_bh_eq = w_dynamic_bh_eq * (1 + ret_t) / (1 + r_1)
    
    # Estrategia 2: Equiponderado (Con Rebalanceo según frecuencia)
    tc_2 = LAMBDA_TC * np.sum(np.abs(w_eq - w_eq)) if t > 0 else 0  
    W[1, t+1] = (W[1, t] * (1 - tc_2)) * (1 + np.dot(w_eq, ret_t))
    
    # Estrategia 3: Markowitz Máximo Sharpe (Buy & Hold)
    r_3 = np.dot(w_dynamic_bh_mk, ret_t)
    W[2, t+1] = W[2, t] * (1 + r_3)
    w_dynamic_bh_mk = w_dynamic_bh_mk * (1 + ret_t) / (1 + r_3)
    
    # Estrategia 4: Markowitz Máximo Sharpe (Con Rebalanceo según frecuencia)
    tc_4 = LAMBDA_TC * np.sum(np.abs(w_mk - w_mk)) if t > 0 else 0 
    W[3, t+1] = (W[3, t] * (1 - tc_4)) * (1 + np.dot(w_mk, ret_t))
    
    # Estrategia 5: NSGA-II Genético (Buy & Hold)
    r_5 = np.dot(w_dynamic_bh_ga, ret_t)
    W[4, t+1] = W[4, t] * (1 + r_5)
    w_dynamic_bh_ga = w_dynamic_bh_ga * (1 + ret_t) / (1 + r_5)
    
    # Estrategia 6: NSGA-II Genético (Con Rebalanceo según frecuencia)
    tc_6 = LAMBDA_TC * np.sum(np.abs(w_ga - w_ga)) if t > 0 else 0
    W[5, t+1] = (W[5, t] * (1 - tc_6)) * (1 + np.dot(w_ga, ret_t))

# Estrategia 7: Programación Dinámica Inteligente 
try:
    W[6, :] = np.array(res_m3["trayectoria_dp"])
except KeyError:
    st.error("⚠️ Falta la trayectoria DP. Por favor, vuelve al Módulo 3 y presiona 'Calcular' de nuevo para actualizar el JSON.")
    st.stop()

# Nombres de las estrategias actualizados a la frecuencia
nombres_estrategias = [
    "1. Equiponderado (Buy & Hold)",
    f"2. Equiponderado (Rebalanceado {frecuencia_sel})",
    "3. Markowitz Máx Sharpe (Buy & Hold)",
    f"4. Markowitz Máx Sharpe (Rebalanceado {frecuencia_sel})",
    "5. NSGA-II GA (Buy & Hold)",
    f"6. NSGA-II GA (Rebalanceado {frecuencia_sel})",
    "7. Programación Dinámica (Bellman Óptimo)"
]

# --- 5. CÁLCULO DE MÉTRICAS FINANCIERAS AVANZADAS ---
metricas_lista = []

# Ajuste dinámico del margen inicial del gráfico
if frecuencia_sel == "Semanal":
    offset = pd.DateOffset(weeks=1)
elif frecuencia_sel == "Trimestral":
    offset = pd.DateOffset(months=3)
else:
    offset = pd.DateOffset(months=1)

fechas_completas = [fechas_backtest[0] - offset] + list(fechas_backtest)

for idx, nombre in enumerate(nombres_estrategias):
    riqueza_path = W[idx, :]
    retornos_path = np.diff(riqueza_path) / riqueza_path[:-1]
    
    ret_total = float((riqueza_path[-1] - riqueza_path[0]) / riqueza_path[0] * 100)
    
    # Anualización dinámica según la frecuencia seleccionada
    mu_anual = np.mean(retornos_path) * factor_anual
    sigma_anual = np.std(retornos_path) * np.sqrt(factor_anual)
    
    # Sharpe Ratio
    sharpe = float(mu_anual / sigma_anual) if sigma_anual > 0 else 0.0
    
    # Sortino Ratio (Downside deviation)
    ret_negativos = retornos_path[retornos_path < 0]
    downside_std = np.sqrt(np.mean(ret_negativos**2)) * np.sqrt(factor_anual) if len(ret_negativos) > 0 else 1e-6
    sortino = float(mu_anual / downside_std)
    
    # Max Drawdown
    df_temp = pd.Series(riqueza_path)
    cum_max = df_temp.cummax()
    drawdowns = (df_temp - cum_max) / cum_max
    max_dd = float(drawdowns.min() * 100)
    
    metricas_lista.append({
        "Estrategia": nombre,
        "Riqueza Final ($)": float(riqueza_path[-1]),
        "Retorno Total (%)": ret_total,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_dd
    })

df_resumen = pd.DataFrame(metricas_lista)

# --- 6. INTERFAZ GRÁFICA Y VISUALIZACIONES ---
st.subheader("📊 Gráfico Evolutivo de Riqueza Superpuesto (7 Curvas)")

total_meses = len(fechas_completas)

# 1. Inicializar la "memoria" para que el gráfico inicie en el primer punto (vacío)
if 'mes_actual' not in st.session_state:
    st.session_state.mes_actual = 1

# 2. Panel de control temporal
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    if st.button("▶️ Animación Lenta", use_container_width=True):
        st.session_state.mes_actual = 0  # Usamos 0 como señal para disparar el bucle automático

with col_ctrl2:
    if st.button("⏭️ Avanzar 1 Mes", use_container_width=True):
        if st.session_state.mes_actual < total_meses:
            st.session_state.mes_actual += 1

with col_ctrl3:
    if st.button("⏮️ Regresar 1 Mes", use_container_width=True):
        if st.session_state.mes_actual > 1:
            st.session_state.mes_actual -= 1

grafico_carrera = st.empty()
st.divider()

colores = ['#95a5a6', '#3498db', '#f1c40f', '#e67e22', '#9b59b6', '#34495e', '#2ecc71']
min_y = np.min(W) * 0.95
max_y = np.max(W) * 1.05

# 3. Lógica de renderizado dinámico
if st.session_state.mes_actual == 0:
    # MODO ANIMADO (Bucle automático)
    for paso in range(1, total_meses + 1, 1):
        fig_7curvas = go.Figure()
        for idx, nombre in enumerate(nombres_estrategias):
            fig_7curvas.add_trace(go.Scatter(
                x=fechas_completas[:paso], y=W[idx, :paso], mode='lines',
                name=nombre, line=dict(color=colores[idx], width=3.5 if idx==6 else 2)
            ))
            
        fig_7curvas.update_layout(
            title=f"Simulación Histórica - Evaluando hasta: {fechas_completas[paso-1].strftime('%B %Y')}",
            xaxis_title="Línea de Tiempo (Meses)", yaxis_title="Capital acumulado (USD)",
            xaxis=dict(range=[fechas_completas[0], fechas_completas[-1]]), 
            yaxis=dict(range=[min_y, max_y]), 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        grafico_carrera.plotly_chart(fig_7curvas, use_container_width=True)
        import time
        time.sleep(0.15) 
        
    # Al terminar la animación, fijamos el estado al último mes para que no se reinicie solo
    st.session_state.mes_actual = total_meses

else:
    # MODO MANUAL (Controlado por los botones de Avanzar/Regresar)
    paso = st.session_state.mes_actual
    fig_7curvas = go.Figure()
    
    for idx, nombre in enumerate(nombres_estrategias):
        fig_7curvas.add_trace(go.Scatter(
            x=fechas_completas[:paso], y=W[idx, :paso], 
            # Añadir marcadores visuales solo si ya llegamos al final de la línea temporal
            mode='lines+markers' if (idx==6 and paso == total_meses) else 'lines',
            name=nombre, line=dict(color=colores[idx], width=3.5 if idx==6 else 2)
        ))
        
    titulo_fecha = fechas_completas[paso-1].strftime('%B %Y')
    fig_7curvas.update_layout(
        title=f"Simulación Histórica - Evaluando hasta: {titulo_fecha}",
        xaxis_title="Línea de Tiempo (Meses)", yaxis_title="Capital acumulado (USD)",
        xaxis=dict(range=[fechas_completas[0], fechas_completas[-1]]), 
        yaxis=dict(range=[min_y, max_y]), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    grafico_carrera.plotly_chart(fig_7curvas, use_container_width=True)

st.subheader("📋 Cuadro de Métricas Consolidadas y Rankings Automáticos")

col_rank1, col_rank2 = st.columns(2)

with col_rank1:
    st.markdown("**🏆 Ranking Automático por Sharpe Ratio (Eficiencia)**")
    df_sharpe_rank = df_resumen.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
    st.dataframe(df_sharpe_rank.style.highlight_max(axis=0, subset=["Sharpe Ratio", "Sortino Ratio"], color="#0b5c1e"), use_container_width=True)

with col_rank2:
    st.markdown("**💰 Ranking Automático por Riqueza Final Absoluta**")
    df_wealth_rank = df_resumen.sort_values(by="Riqueza Final ($)", ascending=False).reset_index(drop=True)
    st.dataframe(df_wealth_rank.style.highlight_max(axis=0, subset=["Riqueza Final ($)", "Retorno Total (%)"], color="#0b5c1e"), use_container_width=True)

st.divider()

st.subheader("📊 Análisis Comparativo Visual de Métricas Clave")
col_bar1, col_bar2 = st.columns(2)

with col_bar1:
    fig_bar_wealth = px.bar(
        df_resumen.sort_values(by="Riqueza Final ($)"), 
        x="Riqueza Final ($)", y="Estrategia", orientation='h',
        title="Comparativa de Capital Final Alcanzado (USD)",
        color="Riqueza Final ($)", color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_bar_wealth, use_container_width=True)

with col_bar2:
    fig_bar_sharpe = px.bar(
        df_resumen.sort_values(by="Sharpe Ratio"), 
        x="Sharpe Ratio", y="Estrategia", orientation='h',
        title="Comparativa de Relación Riesgo-Retorno (Sharpe Ratio)",
        color="Sharpe Ratio", color_continuous_scale="Cividis"
    )
    st.plotly_chart(fig_bar_sharpe, use_container_width=True)

# --- 7. EXPORTACIÓN DE RESULTADOS A EXCEL ---
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df_resumen.to_excel(writer, index=False, sheet_name='Comparativo_General')
    df_sharpe_rank.to_excel(writer, index=False, sheet_name='Ranking_Sharpe')
    df_wealth_rank.to_excel(writer, index=False, sheet_name='Ranking_Riqueza')
buffer.seek(0)

st.download_button(
    label="📥 Descargar Reporte Completo de Comparación y Rankings (Excel)",
    data=buffer,
    file_name="reporte_comparativo_portafolios.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
