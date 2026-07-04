import streamlit as st
import datetime

# Configuración principal de la página
st.set_page_config(page_title="Optimización de Portafolios", page_icon="📈", layout="wide")

st.title("Sistema de Optimización de Portafolio con GA y DP 🚀")
st.markdown("""
Bienvenido al sistema integrador. Este proyecto consolida 3 potentes métodos de optimización financiera:
* **Módulo 1:** Markowitz (Media-Varianza) y Frontera Eficiente.
* **Módulo 2:** NSGA-II (Algoritmo Genético Multiobjetivo).
* **Módulo 3:** Programación Dinámica (Rebalanceo óptimo con Ecuación de Bellman).
* **Módulo 4:** Comparación cruzada de todos los métodos.

👈 **Usa el menú lateral para configurar los parámetros globales y navega por los módulos usando las páginas.**
""")

# --- CONFIGURACIÓN DEL SIDEBAR ---
st.sidebar.header("⚙️ Parámetros Globales")

# Almacenar en session_state para compartir con otras páginas
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = 'FSM, VOLCABC1.LM, ABX.TO, BVN, BHP'

tickers_input = st.sidebar.text_input("Tickers (Separados por coma)", value=st.session_state['tickers'])
st.session_state['tickers'] = tickers_input

col1, col2 = st.sidebar.columns(2)
with col1:
    st.session_state['fecha_inicio'] = st.date_input("Fecha Inicio", datetime.date(2015, 1, 1))
with col2:
    st.session_state['fecha_fin'] = st.date_input("Fecha Fin", datetime.date(2024, 12, 31))

st.session_state['capital'] = st.sidebar.number_input("Capital Inicial (USD)", value=100000, step=10000)

st.sidebar.divider()
st.sidebar.subheader("Parámetros NSGA-II")
st.session_state['nsga_pop'] = st.sidebar.slider("Población (MU)", 50, 300, 100)
st.session_state['nsga_gen'] = st.sidebar.slider("Generaciones", 30, 200, 80)

st.sidebar.divider()
st.sidebar.subheader("Parámetros Programación Dinámica")
st.session_state['dp_tc'] = st.sidebar.slider("Costo Transacción (λ_TC)", 0.0001, 0.0100, 0.0010, format="%.4f")
st.session_state['dp_horizon'] = st.sidebar.slider("Horizonte (periodos)", 4, 52, 12)

st.sidebar.divider()
st.session_state['frecuencia'] = st.sidebar.selectbox("Frecuencia Rebalanceo", ["Semanal", "Mensual", "Trimestral"], index=1)

if st.sidebar.button("Ejecutar Análisis Completo", use_container_width=True, type="primary"):
    st.sidebar.success("¡Parámetros guardados! Navega a los módulos para ver los resultados.")