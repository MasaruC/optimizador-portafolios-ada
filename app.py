import streamlit as st
import datetime

# 1. Configuración principal de la página (Debe ser el primer comando de Streamlit)
st.set_page_config(
    page_title="Sistema de Optimización de Portafolios",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS personalizado para estética Premium
st.markdown("""
<style>
    /* Estilos globales y paleta institucional UNMSM */
    :root {
        --primary-blue: #0f2b5c;
        --secondary-blue: #1e3c72;
        --light-blue: #e2eafc;
        --accent-cyan: #028090;
        --accent-gold: #d4af37;
        --card-bg: #ffffff;
        --hover-shadow: rgba(15, 43, 92, 0.15);
    }
    
    /* Encabezado Principal */
    .main-header {
        background: linear-gradient(135deg, #0f2b5c 0%, #1e3c72 50%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 43, 92, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 80% 20%, rgba(2, 128, 144, 0.3) 0%, transparent 50%);
        pointer-events: none;
    }
    .main-header h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.8rem !important;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.95;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Caja de Información Académica */
    .academic-box {
        background: rgba(255, 255, 255, 0.9);
        border-left: 6px solid #0f2b5c;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 2.5rem;
    }
    .academic-title {
        font-weight: 700;
        color: #0f2b5c;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Grid de Miembros del Grupo */
    .member-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 2.5rem;
    }
    .member-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #0f2b5c !important;
    }
    .member-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(15, 43, 92, 0.12);
        border-color: #1e3c72;
    }
    .member-avatar {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #e2eafc 0%, #b6ccfe 100%);
        color: #0f2b5c !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
        border: 2px solid #ffffff;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    .member-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f2b5c !important;
        margin-bottom: 0.2rem;
    }
    .member-role {
        font-size: 0.85rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tarjetas de Conceptos Algorítmicos */
    .concept-container {
        background: #ffffff;
        color: #1e293b !important;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-bottom: 2rem;
    }
    .concept-container p,
    .concept-container li,
    .concept-container ul,
    .concept-container ol,
    .concept-container h5,
    .concept-container h6,
    .concept-container strong,
    .concept-container span:not(.custom-badge) {
        color: #1e293b !important;
    }
    .concept-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .concept-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f2b5c !important;
        margin: 0;
    }
    .badge-container {
        display: flex;
        gap: 0.5rem;
    }
    .custom-badge {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.4rem 0.8rem;
        border-radius: 50px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-paradigm {
        background-color: #e0f2fe;
        color: #0369a1 !important;
    }
    .badge-complexity {
        background-color: #fee2e2;
        color: #b91c1c !important;
    }
    .badge-framework {
        background-color: #f0fdf4;
        color: #166534 !important;
    }
    
    /* Sección de Guía del Sistema */
    .guide-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    .guide-card {
        background: #f8fafc;
        color: #1e293b !important;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        position: relative;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
    }
    .guide-step {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        background-color: #0f2b5c;
        color: white !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0 auto 1rem auto;
        box-shadow: 0 3px 6px rgba(15, 43, 92, 0.2);
    }
    .guide-title {
        font-weight: 600;
        color: #0f2b5c !important;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .guide-desc {
        font-size: 0.9rem;
        color: #64748b !important;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# --- CONFIGURACIÓN DEL SIDEBAR (Preservado e Intacto) ---
st.sidebar.header("⚙️ Parámetros Globales")

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
st.session_state['dp_tc'] = st.sidebar.slider("Costo Transacción (λ_TC)", 0.0001, 0.0100, 0.0010, step=0.0001, format="%.4f")
st.session_state['dp_horizon'] = st.sidebar.slider("Horizonte (periodos)", 4, 52, 12)

st.sidebar.divider()
st.session_state['frecuencia'] = st.sidebar.selectbox("Frecuencia Rebalanceo", ["Semanal", "Mensual", "Trimestral"], index=1)

if st.sidebar.button("Ejecutar Análisis Completo", use_container_width=True, type="primary"):
    st.sidebar.success("¡Parámetros guardados! Navega a los módulos para ver los resultados.")


# --- VISTA PRINCIPAL DE LA PÁGINA ---

# Encabezado Banner
st.markdown("""
<div class="main-header">
    <h1>📈 Sistema de Optimización de Portafolios</h1>
    <p>Plataforma académica interactiva para la optimización financiera mediante paradigmas algorítmicos avanzados</p>
</div>
""", unsafe_allow_html=True)

# Pestañas principales de navegación
tab_presentacion, tab_glosario, tab_guia = st.tabs([
    "🎓 Carátula de Presentación",
    "📚 Consulta de Conceptos Algorítmicos",
    "🚀 Guía del Sistema"
])

# ==========================================
# PESTAÑA 1: CARÁTULA DE PRESENTACIÓN
# ==========================================
with tab_presentacion:
    st.markdown("""
    <div class="academic-box">
        <div class="academic-title">Universidad Nacional Mayor de San Marcos</div>
        <p style="margin: 0; color: #475569; font-size: 1.05rem;">
            <strong>Facultad de Ingeniería de Sistemas e Informática (FISI)</strong><br>
            Curso: <strong>Análisis y Diseño de Algoritmos (ADA)</strong><br>
            Docente: <strong>Ernesto Cancho-Rodriguez</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("👥 Integrantes del Grupo G4")
    
    # Grid de miembros
    integrantes = [
        {"name": "Juan Masaru Campos Luque", "initials": "JC"},
        {"name": "Angel Gabriel León García", "initials": "AL"},
        {"name": "Alvaro Mathias Melendez Bustamante", "initials": "AM"},
        {"name": "Basadre Pérez Christian Raúl", "initials": "BP"},
        {"name": "Herrera Chavarria Ronnie Rodrigo", "initials": "HH"},
        {"name": "Córdova Guerra Josué Rodrigo", "initials": "CG"},
        {"name": "Cabrejos Palomino Christian Daniel", "initials": "CP"},
        {"name": "Saccaco Oscco Christopher", "initials": "SO"}
    ]
    
    # Renderizar en 4 columnas usando contenedores HTML responsivos
    html_miembros = '<div class="member-grid">'
    for integrante in integrantes:
        html_miembros += (
            f'<div class="member-card">'
            f'<div class="member-avatar">{integrante["initials"]}</div>'
            f'<div class="member-name">{integrante["name"]}</div>'
            f'<div class="member-role">G4 - UNMSM</div>'
            f'</div>'
        )
    html_miembros += '</div>'
    st.markdown(html_miembros, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Resumen del Proyecto
    st.info("""
    💡 **Acerca de este proyecto:**  
    Este sistema evalúa e implementa de manera unificada tres paradigmas algorítmicos fundamentales aplicados a la ingeniería financiera moderna. 
    Permite modelar portafolios eficientes bajo la teoría clásica de Markowitz, resolver fronteras de Pareto multiobjetivo mediante heurísticas evolutivas (NSGA-II) 
    y trazar políticas óptimas de rebalanceo discreto bajo costos de transacción reales utilizando Programación Dinámica (Ecuación de Bellman).
    """)


# ==========================================
# PESTAÑA 2: CONSULTA DE CONCEPTOS ALGORÍTMICOS
# ==========================================
with tab_glosario:
    st.markdown("""
    <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
        Consulte las definiciones teóricas, formulaciones matemáticas, paradigmas de diseño y análisis de complejidad de cada uno de los módulos algorítmicos del sistema.
    </p>
    """, unsafe_allow_html=True)
    
    subtab_m1, subtab_m2, subtab_m3, subtab_m4 = st.tabs([
        "📊 Módulo 1: Markowitz",
        "🧬 Módulo 2: NSGA-II",
        "🧮 Módulo 3: Programación Dinámica",
        "🏆 Módulo 4: Backtesting & Métricas"
    ])
    
    # SUBTAB: MARKOWITZ
    with subtab_m1:
        st.markdown(r"""
        <div class="concept-container">
            <div class="concept-header">
                <div class="concept-title">Modelo de Media-Varianza de Markowitz</div>
                <div class="badge-container">
                    <span class="custom-badge badge-paradigm">Optimización Matemática Continua</span>
                    <span class="custom-badge badge-complexity">O(N³)</span>
                    <span class="custom-badge badge-framework">SLSQP (SciPy)</span>
                </div>
            </div>
            <p><strong>Fundamento Teórico:</strong> Introducido por Harry Markowitz en 1952. Propone que un inversor puede optimizar el rendimiento de su portafolio diversificando la asignación de capital basándose en la relación matemática entre el retorno esperado y el riesgo (varianza y covarianza entre activos).</p>
            <p><strong>Objetivo Matemático:</strong> Encontrar el vector de pesos <i>w</i> de activos que maximiza el Ratio de Sharpe (rendimiento por unidad de riesgo excedente) sujeto a que los pesos sumen 1 y no haya ventas en corto (0 &le; w<sub>i</sub> &le; 1):</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""
        \max_{w} \quad S(w) = \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}} \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0
        """)
        
        st.markdown(r"""
        <div class="concept-container" style="margin-top: 1.5rem;">
            <h5>Detalles de Implementación e Ingeniería de Algoritmos:</h5>
            <ul>
                <li><strong>Algoritmo de Optimización:</strong> Se emplea <strong>Programación Cuadrática Secuencial (SLSQP)</strong> provista por SciPy. Es un método iterativo para resolver problemas de optimización no lineal sujetos a restricciones de igualdad y desigualdad.</li>
                <li><strong>Complejidad Temporal:</strong> Principalmente determinada por la manipulación matemática de matrices de covarianza, lo cual requiere O(N<sup>3</sup>) operaciones de álgebra lineal por iteración, donde N es el número de activos.</li>
                <li><strong>Ventajas:</strong> Garantiza óptimos globales para problemas convexos bien definidos y proporciona la base científica para trazar la <em>Frontera Eficiente</em>.</li>
                <li><strong>Limitaciones:</strong> Altamente sensible a errores de estimación en los parámetros históricos de entrada (&mu; y &Sigma;). Asume distribuciones de retorno normales e ignora los costos de transacción en la optimización estática.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # SUBTAB: NSGA-II
    with subtab_m2:
        st.markdown(r"""
        <div class="concept-container">
            <div class="concept-header">
                <div class="concept-title">Algoritmo Genético Multiobjetivo (NSGA-II)</div>
                <div class="badge-container">
                    <span class="custom-badge badge-paradigm">Metaheurística Evolutiva</span>
                    <span class="custom-badge badge-complexity">O(M · P²)</span>
                    <span class="custom-badge badge-framework">DEAP (Python)</span>
                </div>
            </div>
            <p><strong>Fundamento Teórico:</strong> El <em>Non-dominated Sorting Genetic Algorithm II</em> (Deb et al., 2002) es un algoritmo evolutivo diseñado para resolver problemas de optimización con múltiples objetivos en conflicto. En lugar de fusionar los objetivos en una sola función escalar (como en Markowitz), NSGA-II mantiene y evoluciona una población de soluciones para aproximar el <strong>Frente de Pareto</strong> óptimo.</p>
            <p><strong>Objetivos de Optimización del Portafolio:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""
        f_1(w) = \max \left( w^T \mu \right) \quad \text{(Maximizar Retorno)}
        """)
        st.latex(r"""
        f_2(w) = \min \left( \sqrt{w^T \Sigma w} \right) \quad \text{(Minimizar Riesgo)}
        """)
        
        st.markdown(r"""
        <div class="concept-container" style="margin-top: 1.5rem;">
            <h5>Detalles de Implementación e Ingeniería de Algoritmos:</h5>
            <ul>
                <li><strong>Operadores Genéticos:</strong> Codificación real donde cada individuo representa los pesos del portafolio. Se utiliza cruzamiento uniforme de mezcla, mutación gaussiana adaptada para mantener la restricción &Sigma; w<sub>i</sub> = 1, y selección de sobrevivientes por torneo de dominancia y distancia de hacinamiento (crowding distance).</li>
                <li><strong>Complejidad Temporal:</strong> O(M &middot; P<sup>2</sup>) por generación, donde P es el tamaño de la población y M el número de objetivos (M = 2). Esto se debe al algoritmo de ordenamiento no dominado de los individuos en cada ciclo. Es extremadamente paralelizable.</li>
                <li><strong>Ventajas:</strong> No requiere que las funciones objetivo sean convexas o diferenciables. Permite incorporar restricciones complejas o no-lineales fácilmente (ej. cardinalidad de activos, lotes mínimos) y entrega un espectro completo de opciones óptimas en un solo ciclo de ejecución.</li>
                <li><strong>Limitaciones:</strong> No garantiza alcanzar la solución óptima exacta (convergencia heurística) y requiere sintonizar hiperparámetros críticos como tasas de mutación, tamaño de población y generaciones.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # SUBTAB: PROGRAMACIÓN DINÁMICA
    with subtab_m3:
        st.markdown(r"""
        <div class="concept-container">
            <div class="concept-header">
                <div class="concept-title">Programación Dinámica (Ecuación de Bellman)</div>
                <div class="badge-container">
                    <span class="custom-badge badge-paradigm">Optimización Temporal Discreta</span>
                    <span class="custom-badge badge-complexity">O(T · |S|²)</span>
                    <span class="custom-badge badge-framework">Backward Induction</span>
                </div>
            </div>
            <p><strong>Fundamento Teórico:</strong> El rebalanceo de portafolios bajo costos de transacción es un problema de decisión secuencial en tiempo discreto. Utilizando el Principio de Optimalidad de Bellman, dividimos el problema en subproblemas de una sola etapa que se resuelven recursivamente de manera inversa (inducción hacia atrás).</p>
            <p><strong>Ecuación de Bellman del Portafolio:</strong> En cada periodo t y para cada estado w (pesos actuales), el valor óptimo V<sub>t</sub>(w) representa la máxima riqueza esperada acumulada, decidiendo si rebalancear hacia el portafolio ideal w<sub>opt</sub> (pagando costos de transacción &lambda;<sub>TC</sub>) o mantener el portafolio actual (dejando que derive según los retornos históricos del mercado):</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""
        V_t(w) = \max_{d \in \{\text{Mantener}, \text{Rebalancear}\}} \Big\{ U_d(w) + \gamma \cdot \mathbb{E} \left[ V_{t+1}(w') \right] \Big\}
        """)
        
        st.markdown(r"""
        <div class="concept-container" style="margin-top: 1.5rem;">
            <h5>Detalles de Implementación e Ingeniería de Algoritmos:</h5>
            <ul>
                <li><strong>Discretización de Estados (Grilla):</strong> Para computar la ecuación, se construye un espacio de estados finito S combinatorio discreto de pesos de portafolio con un paso definido (&Delta; = 10%). La transición de estados calcula el movimiento pasivo del portafolio por el retorno de los activos en el periodo.</li>
                <li><strong>Complejidad Temporal (Maldición de la Dimensionalidad):</strong> La complejidad es O(T &middot; |S|<sup>2</sup>), donde |S| representa el tamaño de la grilla de estados. Crece de forma hiper-exponencial con el número de activos N. Por ejemplo, para N = 5 y paso 10%, el espacio tiene |S| = 1001 estados, requiriendo millones de evaluaciones de distancia combinatoria.</li>
                <li><strong>Ventajas:</strong> Ofrece una política óptima global de rebalanceo dinámica y adaptativa en presencia de fricciones reales (costos de transacción), evitando el rebalanceo innecesario si los costos superan las ganancias esperadas de la diversificación.</li>
                <li><strong>Limitaciones:</strong> Inviable computacionalmente para portafolios de tamaño mediano/grande debido a la explosión dimensional del espacio de estados discreto.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""
        |S| = \binom{N + k - 1}{k} \quad \text{donde} \quad k = \frac{1}{\Delta}
        """)

    # SUBTAB: COMPARACIÓN & MÉTRICAS
    with subtab_m4:
        st.markdown(r"""
        <div class="concept-container">
            <div class="concept-header">
                <div class="concept-title">Backtesting de Estrategias y Métricas de Rendimiento</div>
                <div class="badge-container">
                    <span class="custom-badge badge-paradigm">Simulación Walk-Forward</span>
                    <span class="custom-badge badge-complexity">O(D · H)</span>
                    <span class="custom-badge badge-framework">Evaluación Financiera</span>
                </div>
            </div>
            <p><strong>Fundamento Teórico:</strong> El backtesting es el proceso que simula de forma retrospectiva el desempeño histórico real de las estrategias propuestas. A fin de garantizar una comparación justa, se implementa un <strong>único motor de backtesting unificado</strong> que evalúa 7 estrategias simultáneamente aplicando comisiones por costos de transacción y reajustes basados en precios históricos diarios reales de los activos.</p>
            <p><strong>Métricas Financieras Clave Calculadas:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        c_met1, c_met2, c_met3 = st.columns(3)
        with c_met1:
            st.markdown("""
            <div style="background-color: #f8fafc; color: #1e293b; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 8px; text-align: center; height: 100%;">
                <strong style="color: #0f2b5c;">Ratio de Sharpe Anualizado</strong><br>
                Mide el exceso de retorno promedio del portafolio por unidad de volatilidad (riesgo total).
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""
            \text{Sharpe} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p}
            """)
        with c_met2:
            st.markdown("""
            <div style="background-color: #f8fafc; color: #1e293b; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 8px; text-align: center; height: 100%;">
                <strong style="color: #0f2b5c;">Ratio de Sortino</strong><br>
                Evalúa el retorno excedente ajustado únicamente por la volatilidad de los retornos negativos (riesgo a la baja).
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""
            \text{Sortino} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_{downside}}
            """)
        with c_met3:
            st.markdown("""
            <div style="background-color: #f8fafc; color: #1e293b; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 8px; text-align: center; height: 100%;">
                <strong style="color: #0f2b5c;">Maximum Drawdown (MDD)</strong><br>
                La máxima caída porcentual observada en el valor acumulado del portafolio desde su punto pico hasta su valle.
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""
            \text{MDD} = \max_{\tau \le t} \left( \frac{Peak_{\tau} - Value_t}{Peak_{\tau}} \right)
            """)


# ==========================================
# PESTAÑA 3: GUÍA DEL SISTEMA
# ==========================================
with tab_guia:
    st.markdown("""
    <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
        Para garantizar el correcto funcionamiento del panel de comparación unificado, siga esta secuencia ordenada de ejecución:
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="guide-grid">
        <div class="guide-card">
            <div class="guide-step">1</div>
            <div class="guide-title">Parámetros Globales</div>
            <div class="guide-desc">Defina en el menú lateral de la izquierda los activos (tickers), rango de fechas e hiperparámetros.</div>
        </div>
        <div class="guide-card">
            <div class="guide-step">2</div>
            <div class="guide-title">Módulo 1: Markowitz</div>
            <div class="guide-desc">Navegue al Módulo 1 y presione "Ejecutar y Visualizar Algoritmo" para calcular los pesos óptimos y generar <code>resultados_m1.json</code>.</div>
        </div>
        <div class="guide-card">
            <div class="guide-step">3</div>
            <div class="guide-title">Módulo 2: NSGA-II</div>
            <div class="guide-desc">Navegue al Módulo 2 y presione "Ejecutar Evolución NSGA-II" para encontrar el frente de Pareto y guardar <code>resultados_m2.json</code>.</div>
        </div>
        <div class="guide-card">
            <div class="guide-step">4</div>
            <div class="guide-title">Módulo 3: Dinámica</div>
            <div class="guide-desc">Navegue al Módulo 3 y pulse "Calcular Política Óptima" para modelar el rebalanceo y crear <code>resultados_m3.json</code>.</div>
        </div>
        <div class="guide-card">
            <div class="guide-step">5</div>
            <div class="guide-title">Módulo 4: Comparación</div>
            <div class="guide-desc">Navegue al Módulo 4 para ver la comparativa de desempeño y métricas cruzadas bajo un entorno unificado de backtest.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("""
    ⚠️ **Nota sobre cambios de parámetros:** Si modifica los tickers, fechas o capital en el sidebar, **deberá volver a ejecutar consecutivamente los Módulos 1, 2 y 3** para actualizar los datos intermedios guardados en formato JSON. Si la comparación del Módulo 4 no refleja los cambios, presione el botón **🔄 Forzar Recarga de Datos** en el sidebar.
    """)
