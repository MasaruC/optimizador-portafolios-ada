import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import time
import json
import io

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Módulo 1: Markowitz", page_icon="📊", layout="wide")
st.title("📊 Módulo 1: Datos y Optimización de Markowitz")

# Recuperar parámetros globales del sidebar
tickers_input = st.session_state.get('tickers', 'FSM, VOLCABC1.LM, ABX.TO, BVN, BHP')
TICKERS = [t.strip() for t in tickers_input.split(',')]
FECHA_INICIO = st.session_state.get('fecha_inicio', '2015-01-01')
FECHA_FIN = st.session_state.get('fecha_fin', '2024-12-31')
CAPITAL_INICIAL = st.session_state.get('capital', 100000)
DIAS_ANIO = 252
RF = 0.0

st.markdown(f"**Universo:** {len(TICKERS)} acciones | **Horizonte:** {FECHA_INICIO} a {FECHA_FIN} | **Capital:** ${CAPITAL_INICIAL:,}")

# --- 2. DESCARGA Y LIMPIEZA DE DATOS (CACHEADA) ---
@st.cache_data(show_spinner=False)
def cargar_datos(tickers, inicio, fin) -> pd.DataFrame:
    descarga = yf.download(tickers, start=inicio, end=fin, progress=False)
    
    # Corrección Pylance (Línea 31): Validación y casteo explícito a DataFrame
    if descarga is None or descarga.empty:
        return pd.DataFrame()
        
    df_raw = pd.DataFrame(descarga)
    
    # Manejo seguro de columnas
    if 'Close' in df_raw.columns:
        datos = df_raw['Close']
    else:
        datos = df_raw
        
    if isinstance(datos, pd.Series):
        datos = datos.to_frame(name=tickers[0])
        
    return datos.ffill().bfill()

with st.spinner('Descargando y procesando datos financieros...'):
    precios = cargar_datos(TICKERS, FECHA_INICIO, FECHA_FIN)
    TICKERS_VALIDOS = list(precios.columns)
    N = len(TICKERS_VALIDOS)

# --- 3. CÁLCULO DE RETORNOS, MU Y SIGMA ---
# Corrección Pylance (Línea 46): Forzamos a que el resultado sea interpretado como DataFrame
retornos = pd.DataFrame(np.log(precios / precios.shift(1))).dropna()
mu = retornos.mean().to_numpy(dtype=float) * DIAS_ANIO
Sigma = retornos.cov().to_numpy(dtype=float) * DIAS_ANIO

# --- 4. OPTIMIZACIÓN DE MARKOWITZ Y CALLBACK ---
def estadisticas(w):
    w = np.asarray(w, dtype=float)
    r = float(w @ mu)
    v = float(np.sqrt(w @ Sigma @ w))
    sh = (r - RF) / v if v > 0 else -1e9
    return r, v, sh

def neg_sharpe(w):
    return -estadisticas(w)[2]

def varianza(w):
    return float(w @ Sigma @ w)

restriccion_suma = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
limites = tuple((0.0, 1.0) for _ in range(N))
w0 = np.ones(N) / N

# --- CÁLCULO DE LA FRONTERA EFICIENTE (Se mantiene igual) ---
@st.cache_data(show_spinner=False)
def calcular_frontera(_mu, _Sigma, _w0, _limites, _N):
    frontera = []
    objetivos = np.linspace(_mu.min() + 0.001, _mu.max() - 0.001, 100)
    for obj in objetivos:
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: w @ _mu - obj})
        res = minimize(lambda w: float(w @ _Sigma @ w), _w0, method='SLSQP', bounds=_limites, constraints=cons)
        r_f, v_f, sh_f = estadisticas(res.x)
        frontera.append((v_f, r_f))
    return np.array(frontera)

frontera = calcular_frontera(mu, Sigma, w0, limites, N)
vol_ind = np.sqrt(np.diag(Sigma)) # Riesgo individual de cada acción

# --- 5. INTERFAZ GRÁFICA Y VISUALIZACIÓN DEL PROCESO ---
st.subheader("Búsqueda del Portafolio Óptimo (Máximo Sharpe)")

# 1. Definir el pseudocódigo que vamos a mostrar
CODIGO_ALGORITMO = [
    "def optimizar_markowitz(w_inicial):",
    "    while no_converge:",
    "        # Evaluar riesgo y retorno actual",
    "        r, v = calcular_estadisticas(w)",
    "        # Calcular gradientes y restricciones",
    "        direccion = evaluar_slsqp()",
    "        # Actualizar pesos del portafolio",
    "        w_nuevo = aplicar_paso(w, direccion)",
    "    return w_optimo"
]

def renderizar_codigo(linea_activa):
    """Genera HTML para mostrar el código con la línea activa resaltada"""
    html = "<div style='background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 14px; line-height: 1.5;'>"
    for i, linea in enumerate(CODIGO_ALGORITMO):
        # Escapar espacios para mantener la indentación en HTML
        linea_format = linea.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
        
        if i == linea_activa:
            # Sombreado azul estilo VS Code
            html += f"<div style='background-color: #062f4a; border-left: 3px solid #3794ff; padding-left: 5px; width: 100%;'>{linea_format}</div>"
        else:
            html += f"<div style='padding-left: 8px;'>{linea_format}</div>"
    html += "</div>"
    return html

# 2. Crear las columnas para la interfaz lado a lado
col_grafico, col_codigo = st.columns([2, 1])

with col_grafico:
    grafico_placeholder = st.empty()
    metricas_placeholder = st.empty()

with col_codigo:
    st.markdown("**Ejecución en vivo:**")
    codigo_placeholder = st.empty()
    codigo_placeholder.markdown(renderizar_codigo(-1), unsafe_allow_html=True) # Sin resaltar al inicio

iniciar_animacion = st.button("▶️ Ejecutar y Visualizar Algoritmo")

# 3. Lógica de ejecución y sombreado con INTERPOLACIÓN
if iniciar_animacion:
    # Empezamos el historial guardando el peso inicial (equiponderado)
    historial_w = [w0.copy()]
    historial_v = []
    historial_r = []
    
    # Calcular y guardar el punto exacto de inicio
    r_ini, v_ini, sh_ini = estadisticas(w0)
    historial_v.append(v_ini)
    historial_r.append(r_ini)
    
    codigo_placeholder.markdown(renderizar_codigo(0), unsafe_allow_html=True)
    time.sleep(0.5)
    
    def callback_optimizacion(xk):
        w_prev = historial_w[-1]
        w_curr = xk.copy()
        
        # --- LA MAGIA: Interpolación de pesos ---
        # Creamos 10 "cuadros" intermedios entre la iteración anterior y la actual
        num_frames = 10 
        
        for alpha in np.linspace(1/num_frames, 1.0, num_frames):
            # Transición lineal de los pesos del portafolio
            w_inter = (1 - alpha) * w_prev + alpha * w_curr
            
            # Calculamos dónde cae este portafolio intermedio en el plano Riesgo/Retorno
            r_temp, v_temp, sh_temp = estadisticas(w_inter)
            
            historial_v.append(v_temp)
            historial_r.append(r_temp)
            
            # --- Actualización del Gráfico Cuadro por Cuadro ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=frontera[:, 0], y=frontera[:, 1], mode='lines', name='Frontera', line=dict(color='rgba(0, 0, 255, 0.3)')))
            fig.add_trace(go.Scatter(x=vol_ind, y=mu, mode='markers+text', name='Acciones', text=TICKERS_VALIDOS, textposition='top center', marker=dict(color='gray', size=10)))
            
            # Dibujamos el camino recorrido hasta ahora
            fig.add_trace(go.Scatter(x=historial_v, y=historial_r, mode='lines', name='Ruta', line=dict(color='orange', width=3, dash='dot')))
            
            # Destacamos el punto exacto que se está evaluando en este milisegundo
            fig.add_trace(go.Scatter(x=[v_temp], y=[r_temp], mode='markers', name='Explorando...', marker=dict(color='red', symbol='star', size=12)))
            
            fig.update_layout(title='Optimizando (Interpolación Activa)...', xaxis_title='Riesgo', yaxis_title='Retorno', showlegend=False)
            
            grafico_placeholder.plotly_chart(fig, use_container_width=True)
            metricas_placeholder.success(f"Explorando... Sharpe: {sh_temp:.4f} | Volatilidad: {v_temp*100:.2f}%")
            
            # Pausa muy corta para lograr 20-30 FPS visuales
            time.sleep(0.04) 
            
        # Al terminar la animación de los puntos, animamos el pseudocódigo
        pasos_bucle = [2, 3, 4, 5, 6, 7]
        for paso in pasos_bucle:
            codigo_placeholder.markdown(renderizar_codigo(paso), unsafe_allow_html=True)
            time.sleep(0.1) 
            
        # Guardamos el peso actual para que sea el punto de partida de la siguiente iteración
        historial_w.append(w_curr)

    with st.spinner("Ejecutando motor matemático..."):
        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=limites, constraints=restriccion_suma, callback=callback_optimizacion)
        
        w_sharpe = res.x
        r_s, v_s, sh_s = estadisticas(w_sharpe)
        
        codigo_placeholder.markdown(renderizar_codigo(8), unsafe_allow_html=True)
        metricas_placeholder.success(f"¡Óptimo! Sharpe final: {sh_s:.4f} | Volatilidad: {v_s*100:.2f}% | Retorno: {r_s*100:.2f}%")
        
        # Gráfico Final (Limpio y estático)
        fig_final = go.Figure()
        fig_final.add_trace(go.Scatter(x=frontera[:, 0], y=frontera[:, 1], mode='lines', name='Frontera Eficiente', line=dict(color='blue')))
        fig_final.add_trace(go.Scatter(x=vol_ind, y=mu, mode='markers+text', name='Acciones', text=TICKERS_VALIDOS, textposition='top center', marker=dict(color='gray', size=10)))
        fig_final.add_trace(go.Scatter(x=historial_v, y=historial_r, mode='lines', name='Ruta Recorrida', line=dict(color='rgba(255, 165, 0, 0.5)', width=2)))
        fig_final.add_trace(go.Scatter(x=[v_s], y=[r_s], mode='markers', name=f'Máx Sharpe ({sh_s:.2f})', marker=dict(color='red', symbol='star', size=16)))
        fig_final.update_layout(title='Frontera Eficiente de Markowitz (Final)', xaxis_title='Riesgo', yaxis_title='Retorno')
        
        grafico_placeholder.plotly_chart(fig_final, use_container_width=True)

else:
    # ESTADO POR DEFECTO: Cálculo silencioso sin animación
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=limites, constraints=restriccion_suma)
    
    # EXTRAER VARIABLES
    w_sharpe = res.x
    r_s, v_s, sh_s = estadisticas(w_sharpe)
    
    # Renderizar el gráfico final directamente
    fig_final = go.Figure()
    fig_final.add_trace(go.Scatter(x=frontera[:, 0], y=frontera[:, 1], mode='lines', name='Frontera Eficiente', line=dict(color='blue')))
    fig_final.add_trace(go.Scatter(x=vol_ind, y=mu, mode='markers+text', name='Acciones', text=TICKERS_VALIDOS, textposition='top center', marker=dict(color='gray', size=10)))
    fig_final.add_trace(go.Scatter(x=[v_s], y=[r_s], mode='markers', name=f'Máx Sharpe ({sh_s:.2f})', marker=dict(color='red', symbol='star', size=15)))
    fig_final.update_layout(title='Frontera Eficiente de Markowitz (Final)', xaxis_title='Riesgo', yaxis_title='Retorno')
    
    grafico_placeholder.plotly_chart(fig_final, use_container_width=True)
    metricas_placeholder.success(f"Sharpe óptimo: {sh_s:.4f} | Volatilidad: {v_s*100:.2f}% | Retorno: {r_s*100:.2f}%")
    codigo_placeholder.markdown(renderizar_codigo(8), unsafe_allow_html=True) # Mostrar en la línea final "return w_optimo"
# --- 6. SIMULACIÓN DE RIQUEZA ---
st.subheader("Simulación de Evolución de Riqueza")

if 'w_sharpe' in locals():
    mapa_frecuencias = {"Semanal": "W-FRI", "Mensual": "ME", "Trimestral": "QE"}
    frecuencia_sel = st.session_state.get('frecuencia', 'Mensual')
    codigo_freq = mapa_frecuencias.get(frecuencia_sel, "ME")
    factor_anual = {"Semanal": 52, "Mensual": 12, "Trimestral": 4}.get(frecuencia_sel, 12)
    LAMBDA_TC = float(st.session_state.get('dp_tc', 0.0010))

    precios_res = precios.resample(codigo_freq).last()
    retornos_res = precios_res.pct_change().dropna()

    # Motor unificado de backtesting (idéntico al de M4)
    def backtest_estrategia(retornos_df, w_objetivo, capital_inicial, modo="buy_hold", lambda_tc=0.0):
        R = retornos_df.to_numpy(dtype=float)
        T_len = R.shape[0]
        w_obj = np.asarray(w_objetivo, dtype=float)
        riqueza = np.zeros(T_len + 1)
        riqueza[0] = capital_inicial
        w_actual = w_obj.copy()
        
        for t in range(T_len):
            ret = R[t]
            if modo == "rebalancear" and t > 0:
                delta = np.sum(np.abs(w_actual - w_obj))
                tc = lambda_tc * delta
                cap_post = riqueza[t] * (1 - tc)
                ret_port = float(np.dot(w_obj, ret))
                riqueza[t + 1] = cap_post * (1 + ret_port)
                w_actual = w_obj * (1 + ret) / (1 + ret_port)
            else: # buy_hold
                ret_port = float(np.dot(w_actual, ret))
                riqueza[t + 1] = riqueza[t] * (1 + ret_port)
                w_actual = w_actual * (1 + ret) / (1 + ret_port)
        return riqueza

    riqueza_eq = backtest_estrategia(retornos_res, np.ones(N)/N, CAPITAL_INICIAL, "rebalancear", LAMBDA_TC)
    riqueza_mk = backtest_estrategia(retornos_res, w_sharpe, CAPITAL_INICIAL, "rebalancear", LAMBDA_TC)
    riqueza_bh = backtest_estrategia(retornos_res, w_sharpe, CAPITAL_INICIAL, "buy_hold", 0.0)

    fechas_sim = np.append([precios_res.index[0]], retornos_res.index)

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=fechas_sim, y=riqueza_mk, mode='lines', name=f'Markowitz Rebalanceado (${riqueza_mk[-1]:,.2f})', line=dict(color='red')))
    fig_sim.add_trace(go.Scatter(x=fechas_sim, y=riqueza_eq, mode='lines', name=f'Equiponderado (${riqueza_eq[-1]:,.2f})', line=dict(color='blue')))
    fig_sim.add_trace(go.Scatter(x=fechas_sim, y=riqueza_bh, mode='lines', name=f'Buy & Hold (${riqueza_bh[-1]:,.2f})', line=dict(color='gray')))
    fig_sim.update_layout(title=f'Crecimiento de Estrategias (Rebalanceo {frecuencia_sel})', xaxis_title='Fecha', yaxis_title='Capital (USD)')
    st.plotly_chart(fig_sim, use_container_width=True)

    # --- 7. EXPORTACIÓN DE RESULTADOS ---
    metrics = {
        "tickers": TICKERS_VALIDOS,
        "configuracion": {
            "fecha_inicio": str(FECHA_INICIO),
            "fecha_fin": str(FECHA_FIN),
            "capital_inicial": CAPITAL_INICIAL,
            "frecuencia": frecuencia_sel,
            "dias_anio": DIAS_ANIO,
            "rf": RF
        },
        "markowitz_max_sharpe": {
            "retorno": float(r_s),
            "riesgo": float(v_s),
            "sharpe_teorico": float(sh_s), # Cambiamos el nombre para evitar confusión con el de M4
            "pesos": dict(zip(TICKERS_VALIDOS, w_sharpe.round(6).tolist()))
        },
        "simulacion_riqueza_final": {
            "buy_and_hold": float(riqueza_bh[-1]),
            "equiponderado": float(riqueza_eq[-1]),
            "markowitz": float(riqueza_mk[-1])
        },
        "trayectorias": {
            "fechas": [d.strftime("%Y-%m-%d") for d in fechas_sim],
            "buy_and_hold": riqueza_bh.tolist(),
            "equiponderado": riqueza_eq.tolist(),
            "markowitz": riqueza_mk.tolist()
        }
    }
    with open("resultados_m1.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    buffer = io.BytesIO()
    df_pesos_completo = pd.DataFrame({'Ticker': TICKERS_VALIDOS, 'Peso (%)': (w_sharpe * 100).round(2)})
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_pesos_completo.to_excel(writer, index=False, sheet_name='Pesos_Optimos')
    buffer.seek(0)

    st.download_button(
        label="📥 Descargar Pesos del Portafolio Óptimo (Excel)",
        data=buffer,
        file_name="pesos_markowitz.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("💡 La simulación de riqueza aparecerá aquí una vez que se complete la optimización del portafolio.")
