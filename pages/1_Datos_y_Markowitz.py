import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
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

# --- 4. OPTIMIZACIÓN DE MARKOWITZ ---
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

res_sharpe = minimize(neg_sharpe, w0, method='SLSQP', bounds=limites, constraints=restriccion_suma)
w_sharpe = res_sharpe.x
r_s, v_s, sh_s = estadisticas(w_sharpe)

res_minvar = minimize(varianza, w0, method='SLSQP', bounds=limites, constraints=restriccion_suma)
w_minvar = res_minvar.x
r_m, v_m, sh_m = estadisticas(w_minvar)

ret_portafolio = retornos.dot(w_sharpe)
ret_negativos = ret_portafolio[ret_portafolio < 0]
downside_dev = np.sqrt(np.mean(ret_negativos**2)) * np.sqrt(DIAS_ANIO)
sortino_s = (r_s - RF) / downside_dev if downside_dev > 0 else 0

# --- CÁLCULO DE LA FRONTERA EFICIENTE ---
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

# --- 5. INTERFAZ GRÁFICA (MÉTRICAS Y GRÁFICOS) ---
st.subheader("Indicadores del Portafolio Máximo Sharpe")
col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{sh_s:.4f}")
col2.metric("Sortino Ratio", f"{sortino_s:.4f}")
col3.metric("Volatilidad Anual", f"{v_s*100:.2f}%")

st.divider()

col_graf1, col_graf2 = st.columns([2, 1])

with col_graf1:
    fig_frontera = go.Figure()
    vol_ind = np.sqrt(np.diag(Sigma))
    
    fig_frontera.add_trace(go.Scatter(x=frontera[:, 0], y=frontera[:, 1], mode='lines', name='Frontera Eficiente', line=dict(color='blue')))
    fig_frontera.add_trace(go.Scatter(x=vol_ind, y=mu, mode='markers+text', name='Acciones', text=TICKERS_VALIDOS, textposition='top center', marker=dict(color='gray', size=10)))
    fig_frontera.add_trace(go.Scatter(x=[v_s], y=[r_s], mode='markers', name=f'Máx Sharpe ({sh_s:.2f})', marker=dict(color='red', symbol='star', size=15)))
    fig_frontera.add_trace(go.Scatter(x=[v_m], y=[r_m], mode='markers', name='Mín Varianza', marker=dict(color='green', size=12)))
    
    fig_frontera.update_layout(title='Frontera Eficiente de Markowitz', xaxis_title='Riesgo (Volatilidad)', yaxis_title='Retorno Esperado', hovermode='closest')
    st.plotly_chart(fig_frontera, use_container_width=True)

with col_graf2:
    df_pesos = pd.DataFrame({'Ticker': TICKERS_VALIDOS, 'Peso': w_sharpe})
    df_pesos = df_pesos[df_pesos['Peso'] > 0.01]
    fig_pie = px.pie(df_pesos, values='Peso', names='Ticker', title='Composición Máx. Sharpe', hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- 6. SIMULACIÓN DE RIQUEZA ---
st.subheader("Simulación de Evolución de Riqueza")
precios_mensuales = precios.resample('ME').last()
retornos_mensuales = precios_mensuales.pct_change().dropna()
n_meses = len(retornos_mensuales)

riqueza_bh = np.zeros(n_meses + 1)
riqueza_mk = np.zeros(n_meses + 1)
riqueza_eq = np.zeros(n_meses + 1)
riqueza_bh[0] = riqueza_mk[0] = riqueza_eq[0] = CAPITAL_INICIAL

pesos_bh = w0.copy()
w_eq = np.ones(N) / N

for i in range(n_meses):
    # Corrección Pylance (Línea 151): Asegurar arreglo numérico
    ret_mes = np.array(retornos_mensuales.iloc[i].values, dtype=float)
    
    riqueza_eq[i+1] = riqueza_eq[i] * (1 + np.dot(w_eq, ret_mes))
    riqueza_mk[i+1] = riqueza_mk[i] * (1 + np.dot(w_sharpe, ret_mes))
    
    ret_portafolio_bh = np.dot(pesos_bh, ret_mes)
    riqueza_bh[i+1] = riqueza_bh[i] * (1 + ret_portafolio_bh)
    pesos_bh = pesos_bh * (1 + ret_mes) / (1 + ret_portafolio_bh)

fechas_sim = np.append([precios_mensuales.index[0]], retornos_mensuales.index)

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(x=fechas_sim, y=riqueza_mk, mode='lines', name=f'Markowitz Rebalanceado (${riqueza_mk[-1]:,.2f})', line=dict(color='red')))
fig_sim.add_trace(go.Scatter(x=fechas_sim, y=riqueza_eq, mode='lines', name=f'Equiponderado Mensual (${riqueza_eq[-1]:,.2f})', line=dict(color='blue')))
fig_sim.add_trace(go.Scatter(x=fechas_sim, y=riqueza_bh, mode='lines', name=f'Buy & Hold (${riqueza_bh[-1]:,.2f})', line=dict(color='gray')))

fig_sim.update_layout(title='Crecimiento de Estrategias (2015-2024)', xaxis_title='Fecha', yaxis_title='Capital (USD)')
st.plotly_chart(fig_sim, use_container_width=True)

# --- 7. EXPORTACIÓN DE RESULTADOS ---
metrics = {
    "tickers": TICKERS_VALIDOS,
    "markowitz_max_sharpe": {
        "retorno": float(r_s), "riesgo": float(v_s), "sharpe": float(sh_s),
        "pesos": dict(zip(TICKERS_VALIDOS, w_sharpe.round(4).tolist()))
    },
    "simulacion_riqueza_final": {
        "buy_and_hold": float(riqueza_bh[-1]),
        "equiponderado": float(riqueza_eq[-1]),
        "markowitz": float(riqueza_mk[-1])
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