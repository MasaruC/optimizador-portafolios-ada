import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
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
def cargar_datos_mensuales(tickers, inicio, fin, t_horizonte) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    descarga = yf.download(tickers, start=inicio, end=fin, progress=False)
    if descarga is None or descarga.empty:
        return pd.DataFrame(), pd.DatetimeIndex([])
    
    df_raw = pd.DataFrame(descarga)
    datos = df_raw['Close'] if 'Close' in df_raw.columns else df_raw
    
    if isinstance(datos, pd.Series):
        datos = datos.to_frame(name=tickers[0])
    
    precios_m = datos.ffill().bfill().resample('ME').last()
    retornos_m = precios_m.pct_change().dropna()
    
    # Filtrar exactamente para los últimos T periodos evaluados por el Módulo 3
    retornos_sim = retornos_m.iloc[-t_horizonte:]
    
    # --- CORRECCIÓN PYLANCE ---
    # Convertimos explícitamente el índice genérico a un DatetimeIndex
    fechas_sim = pd.DatetimeIndex(retornos_sim.index)
    
    return retornos_sim, fechas_sim

with st.spinner('Sincronizando series de tiempo históricas para el Backtest definitivo...'):
    retornos_mensuales, fechas_backtest = cargar_datos_mensuales(TICKERS, FECHA_INICIO, FECHA_FIN, T_PERIODOS)

with st.spinner('Sincronizando series de tiempo históricas para el Backtest definitivo...'):
    retornos_mensuales, fechas_backtest = cargar_datos_mensuales(TICKERS, FECHA_INICIO, FECHA_FIN, T_PERIODOS)

# --- 4. ENGINE DE SIMULACIÓN DE LAS 7 ESTRATEGIAS ---
n_meses = len(retornos_mensuales)

# Inicializar matrices de riqueza (Tamaño: n_meses + 1)
W = np.zeros((7, n_meses + 1))
W[:, 0] = CAPITAL_INICIAL

# Inicializar estados dinámicos de pesos para las estrategias "Sin Rebalanceo" (Buy & Hold)
w_dynamic_bh_eq = w_eq.copy()
w_dynamic_bh_mk = w_mk.copy()
w_dynamic_bh_ga = w_ga.copy()

# Simulación mes a mes coordinada
for t in range(n_meses):
    ret_mes = np.array(retornos_mensuales.iloc[t].to_numpy(), dtype=float)
    
    # Estrategia 1: Buy & Hold Equiponderado (Sin Rebalanceo)
    r_1 = np.dot(w_dynamic_bh_eq, ret_mes)
    W[0, t+1] = W[0, t] * (1 + r_1)
    w_dynamic_bh_eq = w_dynamic_bh_eq * (1 + ret_mes) / (1 + r_1)
    
    # Estrategia 2: Equiponderado Mensual (Con Rebalanceo Continuo)
    tc_2 = LAMBDA_TC * np.sum(np.abs(w_eq - w_eq)) if t > 0 else 0  # Costo de volver a equilibrar a 1/N
    W[1, t+1] = (W[1, t] * (1 - tc_2)) * (1 + np.dot(w_eq, ret_mes))
    
    # Estrategia 3: Markowitz Máximo Sharpe (Sin Rebalanceo - Buy & Hold)
    r_3 = np.dot(w_dynamic_bh_mk, ret_mes)
    W[2, t+1] = W[2, t] * (1 + r_3)
    w_dynamic_bh_mk = w_dynamic_bh_mk * (1 + ret_mes) / (1 + r_3)
    
    # Estrategia 4: Markowitz Máximo Sharpe (Con Rebalanceo Mensual Fijo)
    tc_4 = LAMBDA_TC * np.sum(np.abs(w_mk - w_mk)) if t > 0 else 0 
    W[3, t+1] = (W[3, t] * (1 - tc_4)) * (1 + np.dot(w_mk, ret_mes))
    
    # Estrategia 5: NSGA-II Genético (Sin Rebalanceo - Buy & Hold)
    r_5 = np.dot(w_dynamic_bh_ga, ret_mes)
    W[4, t+1] = W[4, t] * (1 + r_5)
    w_dynamic_bh_ga = w_dynamic_bh_ga * (1 + ret_mes) / (1 + r_5)
    
    # Estrategia 6: NSGA-II Genético (Con Rebalanceo Mensual Fijo)
    tc_6 = LAMBDA_TC * np.sum(np.abs(w_ga - w_ga)) if t > 0 else 0
    W[5, t+1] = (W[5, t] * (1 - tc_6)) * (1 + np.dot(w_ga, ret_mes))

# Estrategia 7: Programación Dinámica Inteligente 
# Cargamos la trayectoria real exportada por el Módulo 3 para ver la volatilidad
try:
    W[6, :] = np.array(res_m3["trayectoria_dp"])
except KeyError:
    st.error("⚠️ Falta la trayectoria DP. Por favor, vuelve al Módulo 3 y presiona 'Calcular' de nuevo para actualizar el JSON.")
    st.stop()

# Nombres de las estrategias
nombres_estrategias = [
    "1. Equiponderado (Buy & Hold)",
    "2. Equiponderado (Rebalanceado Mensual)",
    "3. Markowitz Máx Sharpe (Buy & Hold)",
    "4. Markowitz Máx Sharpe (Rebalanceado Mensual)",
    "5. NSGA-II GA (Buy & Hold)",
    "6. NSGA-II GA (Rebalanceado Mensual)",
    "7. Programación Dinámica (Bellman Óptimo)"
]

# --- 5. CÁLCULO DE MÉTRICAS FINANCIERAS AVANZADAS ---
metricas_lista = []
fechas_completas = [fechas_backtest[0] - pd.DateOffset(months=1)] + list(fechas_backtest)

for idx, nombre in enumerate(nombres_estrategias):
    riqueza_path = W[idx, :]
    retornos_path = np.diff(riqueza_path) / riqueza_path[:-1]
    
    ret_total = float((riqueza_path[-1] - riqueza_path[0]) / riqueza_path[0] * 100)
    
    # Anualización de retornos promedio y desviaciones (base mensual -> anual = x12)
    mu_anual = np.mean(retornos_path) * 12
    sigma_anual = np.std(retornos_path) * np.sqrt(12)
    
    # Sharpe Ratio
    sharpe = float(mu_anual / sigma_anual) if sigma_anual > 0 else 0.0
    
    # Sortino Ratio (Downside deviation)
    ret_negativos = retornos_path[retornos_path < 0]
    downside_std = np.sqrt(np.mean(ret_negativos**2)) * np.sqrt(12) if len(ret_negativos) > 0 else 1e-6
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

fig_7curvas = go.Figure()
colores = ['#95a5a6', '#3498db', '#f1c40f', '#e67e22', '#9b59b6', '#34495e', '#2ecc71']

for idx, nombre in enumerate(nombres_estrategias):
    fig_7curvas.add_trace(go.Scatter(
        x=fechas_completas, y=W[idx, :], mode='lines+markers' if idx==6 else 'lines',
        name=nombre, line=dict(color=colores[idx], width=3.5 if idx==6 else 2)
    ))

fig_7curvas.update_layout(
    title="Evolución del Capital de Inversión Inicial por Estrategia",
    xaxis_title="Línea de Tiempo (Meses)", yaxis_title="Capital acumulado (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)
st.plotly_chart(fig_7curvas, use_container_width=True)

st.divider()

# Tablas de Rankings Automáticos obligatorios
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

# Gráfico de barras comparativo de dos paneles
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