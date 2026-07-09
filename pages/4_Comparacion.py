# -- Módulo 4 (Reescrito con Tipado Estricto) --

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import io
import time
from typing import Any, Optional, List, Dict, Tuple

# ==========================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================================
st.set_page_config(page_title="Módulo 4: Comparación Cruzada", page_icon="🏆", layout="wide")
st.title("🏆 Módulo 4: Panel de Comparación Cruzada de Estrategias")
st.markdown("Consolidación y evaluación de las 7 estrategias con **un único motor de backtesting justo**.")

# ==========================================================
# 2. CARGA ROBUSTA DE JSONs (SIN CACHÉ PARA EVITAR DATOS VIEJOS)
# ==========================================================
def cargar_json(ruta: str) -> Optional[dict]:
    """Lee el JSON directamente del disco en cada ejecución."""
    if not os.path.exists(ruta):
        return None
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"❌ No se pudo leer `{ruta}`: {e}")
        return None

# Botón para forzar la lectura limpia si Streamlit se queda pegado
if st.sidebar.button("🔄 Forzar Recarga de Datos", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

res_m1_opt = cargar_json("resultados_m1.json")
res_m2_opt = cargar_json("resultados_m2.json")
res_m3_opt = cargar_json("resultados_m3.json")

archivos: Dict[str, Optional[dict]] = {
    "Módulo 1 (resultados_m1.json)": res_m1_opt,
    "Módulo 2 (resultados_m2.json)": res_m2_opt,
    "Módulo 3 (resultados_m3.json)": res_m3_opt
}
faltantes = [k for k, v in archivos.items() if v is None]
if faltantes:
    st.error(f"❌ Faltan o están corruptos: {', '.join(faltantes)}")
    st.info("💡 Ejecuta primero los Módulos 1, 2 y 3 para generar los JSON requeridos.")
    st.stop()

# Afirmaciones para el type checker (Pylance)
assert res_m1_opt is not None
assert res_m2_opt is not None
assert res_m3_opt is not None

res_m1: dict = res_m1_opt
res_m2: dict = res_m2_opt
res_m3: dict = res_m3_opt

def validar_campo(data: dict, *campos: str, archivo: str = "") -> Any:
    nodo: Any = data
    for c in campos:
        if not isinstance(nodo, dict) or c not in nodo:
            st.error(f"❌ Campo `{' → '.join(campos)}` no encontrado en {archivo}.")
            st.stop()
        nodo = nodo[c]
    return nodo

fecha_inicio_sidebar = str(st.session_state.get('fecha_inicio', ''))
fecha_fin_sidebar = str(st.session_state.get('fecha_fin', ''))
fecha_inicio_json = str(res_m1.get("configuracion", {}).get("fecha_inicio", ""))

if fecha_inicio_sidebar and fecha_inicio_json and fecha_inicio_sidebar != fecha_inicio_json:
    st.warning("⚠️ **Datos Desactualizados:** Las fechas en el Sidebar no coinciden con las del JSON. "
               "Ve a los Módulos 1, 2 y 3, pulsa sus botones de ejecución para generar los nuevos datos, "
               "o usa el botón **'🔄 Forzar Recarga'** en el menú lateral izquierdo.")
# ==========================================================
# 3. EXTRACCIÓN DE PARÁMETROS DESDE LOS JSON (sin session_state)
# ==========================================================
TICKERS          = list(validar_campo(res_m1, "tickers", archivo="M1"))
N                = len(TICKERS)
config_m1        = validar_campo(res_m1, "configuracion", archivo="M1")
FECHA_INICIO     = str(config_m1["fecha_inicio"])
FECHA_FIN        = str(config_m1["fecha_fin"])
CAPITAL_INICIAL  = float(config_m1["capital_inicial"])
frecuencia_sel   = str(config_m1["frecuencia"])
DIAS_ANIO        = int(config_m1.get("dias_anio", 252))
RF               = float(config_m1.get("rf", 0.0))

# Extraemos lambda_tc desde la configuración de M2 (o M3 como respaldo)
config_m2        = validar_campo(res_m2, "configuracion", archivo="M2")
LAMBDA_TC        = float(config_m2.get("lambda_tc", float(res_m3.get("parametros_dp", {}).get("lambda_tc", 0.0010))))

factor_anual = {"Semanal": 52, "Mensual": 12, "Trimestral": 4}.get(frecuencia_sel, 12)

# Parámetros DP desde M3
params_dp = res_m3.get("parametros_dp", {})
LAMBDA_TC   = float(params_dp.get("lambda_tc", 0.0010))
PASO_GRILLA = float(params_dp.get("paso_grilla", 0.10))
T_DP        = int(params_dp.get("periodos_t", 12))

# NUEVO: Cargar grilla y política exactas de M3
S_matriz = np.array(res_m3.get("grilla_estados", []), dtype=float)
politica_matriz = np.array(res_m3.get("matriz_politica", []), dtype=int)

# ==========================================================
# 4. EXTRACCIÓN DE PESOS ÓPTIMOS
# ==========================================================
def extraer_pesos(data: dict, clave_estrategia: str, archivo: str) -> np.ndarray:
    pesos_dict = validar_campo(data, clave_estrategia, "pesos", archivo=archivo)
    pesos_list = [float(pesos_dict.get(t, 0.0)) for t in TICKERS]
    pesos = np.array(pesos_list, dtype=float)
    s = float(pesos.sum())
    if s <= 0:
        st.error(f"❌ Pesos inválidos en {archivo}.")
        st.stop()
    return pesos / s

w_markowitz = extraer_pesos(res_m1, "markowitz_max_sharpe", "M1")
w_nsga2     = extraer_pesos(res_m2, "nsga2_max_sharpe",     "M2")
w_eq        = np.ones(N) / N
w_target_dp = w_markowitz.copy()  # M3 usó w_markowitz como target

# ==========================================================
# 5. DESCARGA UNIFICADA DEL HISTÓRICO (mismo para todos)
# ==========================================================
@st.cache_data(show_spinner=False)
def descargar_precios(tickers: List[str], inicio: str, fin: str) -> pd.DataFrame:
    descarga = yf.download(tickers, start=inicio, end=fin, progress=False)
    if descarga is None or descarga.empty:
        return pd.DataFrame()
    df = pd.DataFrame(descarga)
    datos = df['Close'] if 'Close' in df.columns else df
    if isinstance(datos, pd.Series):
        datos = datos.to_frame(name=tickers[0])
    return datos.ffill().bfill()

with st.spinner("Descargando histórico unificado para backtest justo..."):
    precios = descargar_precios(TICKERS, FECHA_INICIO, FECHA_FIN)
    if precios.empty:
        st.error("❌ No se pudieron descargar precios. Verifica los tickers y fechas.")
        st.stop()

mapa_freq = {"Semanal": "W-FRI", "Mensual": "ME", "Trimestral": "QE"}
codigo_freq = mapa_freq.get(frecuencia_sel, "ME")
precios_res   = precios.resample(codigo_freq).last()
retornos_periodo = precios_res.pct_change().dropna()
T = len(retornos_periodo)
fechas_retorno = retornos_periodo.index
fechas_sim = pd.to_datetime(np.append([precios_res.index[0]], fechas_retorno))

st.info(f"📅 Periodos de backtest: **{T}**  |  Frecuencia: **{frecuencia_sel}**  |  "
        f"Capital inicial: **${CAPITAL_INICIAL:,.2f}**  |  λ_TC: **{LAMBDA_TC:.4f}**")

# ==========================================================
# 6. MOTOR ÚNICO DE BACKTESTING
# ==========================================================
def backtest_estrategia(retornos_df: pd.DataFrame, w_objetivo: np.ndarray, capital_inicial: float,
                        modo: str = "buy_hold", lambda_tc: float = 0.0, w_target_dp: Optional[np.ndarray] = None,
                        S_dp: np.ndarray = None, politica_dp: np.ndarray = None, T_dp: int = 12) -> Tuple[np.ndarray, List[int], float]:
    R = retornos_df.to_numpy(dtype=float)
    T_len, N_len = R.shape
    w_objetivo = np.asarray(w_objetivo, dtype=float)
    if w_target_dp is None:
        w_target_dp = w_objetivo.copy()

    riqueza = np.zeros(T_len + 1)
    riqueza[0] = capital_inicial
    w_actual = w_objetivo.copy()
    rebalanceos: List[int] = []
    costo_total = 0.0

    for t in range(T_len):
        ret = R[t]

        if modo == "rebalancear" and t > 0:
            delta = np.sum(np.abs(w_actual - w_objetivo))
            tc = lambda_tc * delta
            costo_total += tc * riqueza[t]
            cap_post = riqueza[t] * (1 - tc)
            ret_port = float(np.dot(w_objetivo, ret))
            riqueza[t + 1] = cap_post * (1 + ret_port)
            w_actual = w_objetivo * (1 + ret) / (1 + ret_port)
            rebalanceos.append(t)

        elif modo == "dp" and t > 0:
            # Usamos la grilla y política exactas del Módulo 3
            if S_dp is not None and politica_dp is not None and S_dp.size > 0:
                idx_s = int(np.argmin(np.sum(np.abs(S_dp - w_actual), axis=1)))
                idx_a = int(politica_dp[min(t, T_dp - 1), idx_s])
                w_dp_nuevo = S_dp[idx_a]
                
                if idx_s != idx_a:
                    tc = lambda_tc * np.sum(np.abs(w_actual - w_dp_nuevo))
                    costo_total += tc * riqueza[t]
                    cap_post = riqueza[t] * (1 - tc)
                    ret_port = float(np.dot(w_dp_nuevo, ret))
                    riqueza[t + 1] = cap_post * (1 + ret_port)
                    w_actual = w_dp_nuevo * (1 + ret) / (1 + ret_port)
                    rebalanceos.append(t)
                else:
                    ret_port = float(np.dot(w_actual, ret))
                    riqueza[t + 1] = riqueza[t] * (1 + ret_port)
                    w_actual = w_actual * (1 + ret) / (1 + ret_port)
            else:
                # Fallback por si no hay matriz
                desv = np.sum(np.abs(w_actual - w_target_dp))
                if desv > 0.15: # Umbral simple
                    tc = lambda_tc * desv
                    costo_total += tc * riqueza[t]
                    cap_post = riqueza[t] * (1 - tc)
                    ret_port = float(np.dot(w_target_dp, ret))
                    riqueza[t + 1] = cap_post * (1 + ret_port)
                    w_actual = w_target_dp * (1 + ret) / (1 + ret_port)
                    rebalanceos.append(t)
                else:
                    ret_port = float(np.dot(w_actual, ret))
                    riqueza[t + 1] = riqueza[t] * (1 + ret_port)
                    w_actual = w_actual * (1 + ret) / (1 + ret_port)

        else:  # buy_hold o primer periodo
            ret_port = float(np.dot(w_actual, ret))
            riqueza[t + 1] = riqueza[t] * (1 + ret_port)
            w_actual = w_actual * (1 + ret) / (1 + ret_port)

    return riqueza, rebalanceos, costo_total

# ==========================================================
# 7. EJECUCIÓN DE LAS 7 ESTRATEGIAS CON EL MISMO MOTOR
# ==========================================================
estrategias: List[Tuple[str, str, np.ndarray, float]] = [
    ("1. Equiponderado (Buy & Hold)",                       "buy_hold",    w_eq,        0.0),
    (f"2. Equiponderado (Rebalanceado {frecuencia_sel})",   "rebalancear", w_eq,        LAMBDA_TC),
    ("3. Markowitz Máx. Sharpe (Buy & Hold)",               "buy_hold",    w_markowitz, 0.0),
    (f"4. Markowitz (Rebalanceado {frecuencia_sel})",       "rebalancear", w_markowitz, LAMBDA_TC),
    ("5. NSGA-II (Buy & Hold)",                             "buy_hold",    w_nsga2,     0.0),
    (f"6. NSGA-II (Rebalanceado {frecuencia_sel})",         "rebalancear", w_nsga2,     LAMBDA_TC),
    ("7. Programación Dinámica (Bellman)",                  "dp",          w_target_dp, LAMBDA_TC),
]

with st.spinner("Ejecutando backtesting unificado de las 7 estrategias..."):
    trayectorias: Dict[str, np.ndarray] = {}
    rebalanceos_dict: Dict[str, List[int]] = {}
    costos_dict: Dict[str, float] = {}
    for nombre, modo, w, lam in estrategias:
        w_path, reb, cost = backtest_estrategia(
            retornos_periodo, w, CAPITAL_INICIAL, modo, lam, w_target_dp,
            S_matriz, politica_matriz, T_DP  # <--- AÑADIDO
        )
        trayectorias[nombre]    = w_path
        rebalanceos_dict[nombre] = reb
        costos_dict[nombre]      = cost

# Matriz W (estrategias x tiempo) para animación
nombres_estrategias = [e[0] for e in estrategias]
W = np.vstack([trayectorias[n] for n in nombres_estrategias])

# ==========================================================
# 8. CÁLCULO CORRECTO DE MÉTRICAS
# ==========================================================
def calcular_metricas(riqueza_path: np.ndarray, factor_anual: int, rf: float = 0.0) -> Dict[str, Any]:
    riqueza_path = np.asarray(riqueza_path, dtype=float)
    retornos = np.diff(riqueza_path) / riqueza_path[:-1]
    if len(retornos) < 2:
        return {
            "Riqueza Final ($)": 0.0, "Retorno Total (%)": 0.0,
            "Sharpe Ratio": 0.0, "Sortino Ratio": 0.0,
            "Max Drawdown (%)": 0.0, "Volatilidad Anual (%)": 0.0
        }

    ret_total = (riqueza_path[-1] / riqueza_path[0] - 1) * 100
    mu_p = np.mean(retornos)
    sd_p = np.std(retornos, ddof=1)
    mu_anual  = mu_p * factor_anual
    sig_anual = sd_p * np.sqrt(factor_anual)
    sharpe = (mu_anual - rf) / sig_anual if sig_anual > 0 else 0.0

    downside = retornos[retornos < 0]
    if len(downside) > 1:
        dd_std = np.std(downside, ddof=1) * np.sqrt(factor_anual)
        sortino = (mu_anual - rf) / dd_std if dd_std > 0 else float('nan')
    else:
        sortino = float('nan')

    cum_max = np.maximum.accumulate(riqueza_path)
    drawdowns = (riqueza_path - cum_max) / cum_max
    max_dd = drawdowns.min() * 100

    return {
        "Riqueza Final ($)":     float(riqueza_path[-1]),
        "Retorno Total (%)":     float(ret_total),
        "Sharpe Ratio":          float(sharpe),
        "Sortino Ratio":         float(sortino),
        "Max Drawdown (%)":      float(max_dd),
        "Volatilidad Anual (%)": float(sig_anual * 100),
    }

metricas_lista: List[Dict[str, Any]] = []
for nombre, *_ in estrategias:
    m = calcular_metricas(trayectorias[nombre], factor_anual, RF)
    m["Estrategia"] = nombre
    m["Costo Total TC ($)"] = float(costos_dict[nombre])
    m["N° Rebalanceos"]     = len(rebalanceos_dict[nombre])
    metricas_lista.append(m)

# Nombres actualizados para aclarar que son métricas históricas del backtest
columnas_orden = [
    "Estrategia", 
    "Riqueza Final ($)", 
    "Retorno Total Histórico (%)",
    "Sharpe Ratio Histórico", 
    "Sortino Ratio Histórico", 
    "Max Drawdown (%)",
    "Volatilidad Anual Histórica (%)", 
    "Costo Total TC ($)", 
    "N° Rebalanceos"
]

# Renombramos las llaves del diccionario para que coincidan con las columnas
df_resumen = pd.DataFrame(metricas_lista).rename(columns={
    "Retorno Total (%)": "Retorno Total Histórico (%)",
    "Sharpe Ratio": "Sharpe Ratio Histórico",
    "Sortino Ratio": "Sortino Ratio Histórico",
    "Volatilidad Anual (%)": "Volatilidad Anual Histórica (%)"
})[columnas_orden]

# ==========================================================
# 9. INTERFAZ GRÁFICA — ANIMACIÓN DE LAS 7 CURVAS
# ==========================================================
st.subheader("📊 Evolución de Riqueza Superpuesta (7 Estrategias)")

total_periodos = len(fechas_sim)
if 'mes_actual' not in st.session_state:
    st.session_state.mes_actual = 1

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶️ Animación Lenta", use_container_width=True):
        st.session_state.mes_actual = 0
with c2:
    if st.button("⏭️ Avanzar 1 Periodo", use_container_width=True):
        if st.session_state.mes_actual < total_periodos:
            st.session_state.mes_actual += 1
with c3:
    if st.button("⏮️ Regresar 1 Periodo", use_container_width=True):
        if st.session_state.mes_actual > 1:
            st.session_state.mes_actual -= 1

grafico_carrera = st.empty()
st.divider()

colores = ['#95a5a6', '#3498db', '#f1c40f', '#e67e22',
           '#9b59b6', '#34495e', '#2ecc71']
min_y = float(np.min(W)) * 0.95
max_y = float(np.max(W)) * 1.05

def dibujar_fig(paso: int) -> go.Figure:
    fig = go.Figure()
    for idx, nombre in enumerate(nombres_estrategias):
        fig.add_trace(go.Scatter(
            x=fechas_sim[:paso], y=W[idx, :paso],
            mode='lines+markers' if (idx == 6 and paso == total_periodos) else 'lines',
            name=nombre,
            line=dict(color=colores[idx], width=3.5 if idx == 6 else 2)
        ))
    # Corrección de type checker usando pd.Timestamp nativo
    titulo_dt = pd.Timestamp(fechas_sim[paso - 1])
    titulo = titulo_dt.strftime('%Y-%m-%d')
    fig.update_layout(
        title=f"Backtest Unificado — Evaluando hasta: {titulo}",
        xaxis_title="Línea de Tiempo", yaxis_title="Capital acumulado (USD)",
        xaxis=dict(range=[fechas_sim[0], fechas_sim[-1]]),
        yaxis=dict(range=[min_y, max_y]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig

if st.session_state.mes_actual == 0:
    for paso in range(1, total_periodos + 1):
        grafico_carrera.plotly_chart(dibujar_fig(paso), use_container_width=True)
        time.sleep(0.12)
    st.session_state.mes_actual = total_periodos
else:
    grafico_carrera.plotly_chart(dibujar_fig(st.session_state.mes_actual),
                                  use_container_width=True)

    # Calcular y graficar drawdowns de las 7 estrategias
    def calcular_drawdown(riqueza_path):
        cum_max = np.maximum.accumulate(riqueza_path)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = (riqueza_path - cum_max) / cum_max
            dd = np.nan_to_num(dd, nan=0.0)
        return dd * 100

    fig_dd_all = go.Figure()
    for idx, nombre in enumerate(nombres_estrategias):
        dd_path = calcular_drawdown(W[idx])
        fig_dd_all.add_trace(go.Scatter(
            x=fechas_sim, y=dd_path,
            mode='lines',
            name=nombre,
            line=dict(color=colores[idx], width=2.5 if idx == 6 else 1.5)
        ))
    fig_dd_all.update_layout(
        title="Evolución de Caídas Máximas (Drawdown - 7 Estrategias)",
        xaxis_title="Línea de Tiempo", yaxis_title="Drawdown (%)",
        xaxis=dict(range=[fechas_sim[0], fechas_sim[-1]]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig_dd_all, use_container_width=True)

# ==========================================================
# 10. TABLAS DE RANKING
# ==========================================================
st.subheader("📋 Cuadro de Métricas Consolidadas y Rankings")

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**🏆 Ranking por Sharpe Ratio Histórico (Eficiencia Riesgo-Retorno)**")
    df_sharpe = df_resumen.sort_values("Sharpe Ratio Histórico", ascending=False).reset_index(drop=True)
    st.dataframe(df_sharpe.style.highlight_max(
        subset=["Sharpe Ratio Histórico", "Sortino Ratio Histórico"], color="#0b5c1e"
    ), use_container_width=True)

with rc2:
    st.markdown("**💰 Ranking por Riqueza Final Absoluta**")
    df_wealth = df_resumen.sort_values("Riqueza Final ($)", ascending=False).reset_index(drop=True)
    st.dataframe(df_wealth.style.highlight_max(
        subset=["Riqueza Final ($)", "Retorno Total Histórico (%)"], color="#0b5c1e"
    ), use_container_width=True)

st.divider()

# ==========================================================
# 11. GRÁFICOS COMPARATIVOS DE BARRAS
# ==========================================================
st.subheader("📊 Análisis de Frontera Eficiente y Distribución de Activos")

c1, c2 = st.columns(2)
with c1:
    # Scatter Plot: Riesgo vs Retorno de las 7 estrategias
    fig_risk_return = px.scatter(
        df_resumen,
        x='Volatilidad Anual Histórica (%)',
        y='Retorno Total Histórico (%)',
        color='Sharpe Ratio Histórico',
        size='Riqueza Final ($)',
        text='Estrategia',
        title='Perfil Riesgo-Retorno de las 7 Estrategias (Backtest)',
        color_continuous_scale='Viridis',
        labels={
            'Volatilidad Anual Histórica (%)': 'Volatilidad Anual Histórica (%)', 
            'Retorno Total Histórico (%)': 'Retorno Total Histórico (%)'
        }
    )
    fig_risk_return.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig_risk_return.update_layout(xaxis=dict(range=[df_resumen['Volatilidad Anual Histórica (%)'].min() * 0.9, df_resumen['Volatilidad Anual Histórica (%)'].max() * 1.1]))
    st.plotly_chart(fig_risk_return, use_container_width=True)

with c2:
    # Grouped Bar Chart: Comparativa de pesos por método
    df_pesos_comp = pd.DataFrame({
        'Ticker': TICKERS * 3,
        'Peso (%)': np.concatenate([w_markowitz * 100, w_nsga2 * 100, w_eq * 100]).round(2),
        'Método': ['Markowitz'] * len(TICKERS) + ['NSGA-II (GA)'] * len(TICKERS) + ['Equiponderado'] * len(TICKERS)
    })
    fig_pesos_comp = px.bar(
        df_pesos_comp,
        x='Ticker',
        y='Peso (%)',
        color='Método',
        barmode='group',
        title='Comparación de Pesos de Portafolio por Método',
        color_discrete_sequence=['#e74c3c', '#9b59b6', '#3498db']
    )
    st.plotly_chart(fig_pesos_comp, use_container_width=True)

st.subheader("📊 Análisis Comparativo Visual (Métricas Clave)")
bc1, bc2 = st.columns(2)
with bc1:
    fig_w = px.bar(df_resumen.sort_values("Riqueza Final ($)"),
                   x="Riqueza Final ($)", y="Estrategia", orientation='h',
                   title="Capital Final Alcanzado (USD)",
                   color="Riqueza Final ($)", color_continuous_scale="Viridis")
    st.plotly_chart(fig_w, use_container_width=True)

with bc2:
    fig_s = px.bar(df_resumen.sort_values("Sharpe Ratio Histórico"),
                   x="Sharpe Ratio Histórico", y="Estrategia", orientation='h',
                   title="Relación Riesgo-Retorno (Sharpe Ratio Histórico)",
                   color="Sharpe Ratio Histórico", color_continuous_scale="Cividis")
    st.plotly_chart(fig_s, use_container_width=True)

bc3, bc4 = st.columns(2)
with bc3:
    fig_dd = px.bar(df_resumen.sort_values("Max Drawdown (%)"),
                    x="Max Drawdown (%)", y="Estrategia", orientation='h',
                    title="Máximo Drawdown (%) — menos negativo es mejor",
                    color="Max Drawdown (%)", color_continuous_scale="RdBu")
    st.plotly_chart(fig_dd, use_container_width=True)

with bc4:
    fig_tc = px.bar(df_resumen.sort_values("Costo Total TC ($)"),
                    x="Costo Total TC ($)", y="Estrategia", orientation='h',
                    title="Costo Total Acumulado por Transacciones (USD)",
                    color="Costo Total TC ($)", color_continuous_scale="Magma")
    st.plotly_chart(fig_tc, use_container_width=True)

# ==========================================================
# 12. EXPORTACIÓN A EXCEL
# ==========================================================
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df_resumen.to_excel(writer, index=False, sheet_name='Comparativo_General')
    df_sharpe.to_excel(writer,  index=False, sheet_name='Ranking_Sharpe')
    df_wealth.to_excel(writer,  index=False, sheet_name='Ranking_Riqueza')
    pd.DataFrame({n: trayectorias[n] for n in nombres_estrategias}).to_excel(
        writer, index=False, sheet_name='Trayectorias'
    )
buffer.seek(0)

st.download_button(
    label="📥 Descargar Reporte Completo (Excel)",
    data=buffer,
    file_name="reporte_comparativo_portafolios.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
