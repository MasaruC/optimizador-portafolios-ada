import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from deap import base, creator, tools, algorithms
import time
import random
import json
import io

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Módulo 2: NSGA-II", page_icon="🧬", layout="wide")
st.title("🧬 Módulo 2: Algoritmo Genético NSGA-II")
st.markdown("Optimización Multiobjetivo: **Maximizar Retorno** y **Minimizar Riesgo** simultáneamente.")

# Recuperar parámetros globales
tickers_input = st.session_state.get('tickers', 'FSM, VOLCABC1.LM, ABX.TO, BVN, BHP')
TICKERS = [t.strip() for t in tickers_input.split(',')]
FECHA_INICIO = st.session_state.get('fecha_inicio', '2015-01-01')
FECHA_FIN = st.session_state.get('fecha_fin', '2024-12-31')
POBLACION = st.session_state.get('nsga_pop', 100)
GENERACIONES = st.session_state.get('nsga_gen', 80)
DIAS_ANIO = 252
RF = 0.0

col_params1, col_params2 = st.columns(2)
col_params1.info(f"**Individuos (Población):** {POBLACION}")
col_params2.info(f"**Generaciones:** {GENERACIONES}")

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

with st.spinner('Cargando datos para el algoritmo genético...'):
    precios = cargar_datos(TICKERS, FECHA_INICIO, FECHA_FIN)
    TICKERS_VALIDOS = list(precios.columns)
    N = len(TICKERS_VALIDOS)

retornos = pd.DataFrame(np.log(precios / precios.shift(1))).dropna()
mu = retornos.mean().to_numpy(dtype=float) * DIAS_ANIO
Sigma = retornos.cov().to_numpy(dtype=float) * DIAS_ANIO

# --- 3. CONFIGURACIÓN DE DEAP (NSGA-II) ---
# Usamos # type: ignore para que Pylance omita la metaprogramación
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti) # type: ignore

def crear_individuo(n):
    pesos = np.random.random(n)
    pesos /= np.sum(pesos)
    return creator.Individual(pesos.tolist()) # type: ignore

def evaluar_portafolio(ind, mu_vec, sigma_mat):
    w = np.array(ind, dtype=float)
    if np.sum(w) == 0:
        return -1e9, 1e9
    w /= np.sum(w)
    retorno = float(np.dot(w, mu_vec))
    riesgo = float(np.sqrt(np.dot(w.T, np.dot(sigma_mat, w))))
    return retorno, riesgo

def mutacion_portafolio(ind, indpb):
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] += random.gauss(0, 0.1)
            ind[i] = max(0.001, ind[i])
    suma = sum(ind)
    for i in range(len(ind)):
        ind[i] /= suma
    return ind,

def cruce_portafolio(ind1, ind2):
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    sum1, sum2 = sum(ind1), sum(ind2)
    for i in range(len(ind1)):
        ind1[i] /= sum1
        ind2[i] /= sum2
    return ind1, ind2

toolbox = base.Toolbox()
toolbox.register("individual", crear_individuo, N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
toolbox.register("evaluate", evaluar_portafolio, mu_vec=mu, sigma_mat=Sigma)
toolbox.register("mate", cruce_portafolio)
toolbox.register("mutate", mutacion_portafolio, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# --- PANEL DE CÓDIGO VISUAL ---
CODIGO_NSGA2 = [
    "def evolucion_nsga2(poblacion_inicial):",
    "    pop = evaluar_fitness(poblacion_inicial)",
    "    for gen in range(GENERACIONES):",
    "        # 1. Selección por Torneo",
    "        padres = seleccion_torneo(pop)",
    "        # 2. Recombinación (Cruce)",
    "        hijos = aplicar_cruce(padres)",
    "        # 3. Mutación Genética",
    "        hijos = aplicar_mutacion(hijos)",
    "        # 4. Evaluación de nueva generación",
    "        evaluar_fitness(hijos)",
    "        # 5. Reemplazo (Supervivencia del más apto)",
    "        pop = seleccionar_mejores(pop + hijos)",
    "    return extraer_frente_pareto(pop)"
]

def renderizar_codigo(linea_activa):
    html = "<div style='background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 14px; line-height: 1.5;'>"
    for i, linea in enumerate(CODIGO_NSGA2):
        linea_format = linea.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
        if i == linea_activa:
            html += f"<div style='background-color: #062f4a; border-left: 3px solid #3794ff; padding-left: 5px; width: 100%;'>{linea_format}</div>"
        else:
            html += f"<div style='padding-left: 8px;'>{linea_format}</div>"
    html += "</div>"
    return html

# --- 4. EJECUCIÓN DEL ALGORITMO VISUAL ---
st.subheader("Evolución del Frente de Pareto en Tiempo Real")

col_graf, col_cod = st.columns([2, 1])

with col_graf:
    grafico_placeholder = st.empty()
    metricas_placeholder = st.empty()

with col_cod:
    st.markdown("**Algoritmo Genético (Vivo):**")
    codigo_placeholder = st.empty()
    codigo_placeholder.markdown(renderizar_codigo(-1), unsafe_allow_html=True)

if st.button("🚀 Ejecutar Evolución NSGA-II", type="primary", use_container_width=True):
    # Variables para guardar el historial del mejor Sharpe global
    mejor_sharpe_historico = -1
    
    with st.spinner(f'Evolucionando {POBLACION} individuos por {GENERACIONES} generaciones...'):
        random.seed(42)
        codigo_placeholder.markdown(renderizar_codigo(0), unsafe_allow_html=True)
        time.sleep(0.3)
        
        pop = toolbox.population(n=POBLACION) # type: ignore
        codigo_placeholder.markdown(renderizar_codigo(1), unsafe_allow_html=True)
        
        fitnesses = list(map(toolbox.evaluate, pop)) # type: ignore
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        pop = toolbox.select(pop, len(pop)) # type: ignore
        time.sleep(0.3)
            
        # --- BUCLE GENERACIONAL ANIMADO ---
        for gen in range(GENERACIONES):
            codigo_placeholder.markdown(renderizar_codigo(2), unsafe_allow_html=True)
            
            # 1. Torneo
            codigo_placeholder.markdown(renderizar_codigo(4), unsafe_allow_html=True)
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring] # type: ignore
            
            # 2. Cruce
            codigo_placeholder.markdown(renderizar_codigo(6), unsafe_allow_html=True)
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:
                    toolbox.mate(ind1, ind2) # type: ignore
                    del ind1.fitness.values, ind2.fitness.values
                    
            # 3. Mutación
            codigo_placeholder.markdown(renderizar_codigo(8), unsafe_allow_html=True)
            for ind in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(ind) # type: ignore
                    del ind.fitness.values
                    
            # 4. Evaluar
            codigo_placeholder.markdown(renderizar_codigo(10), unsafe_allow_html=True)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) # type: ignore
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            # 5. Selección de la nueva generación
            codigo_placeholder.markdown(renderizar_codigo(12), unsafe_allow_html=True)
            pop = toolbox.select(pop + offspring, POBLACION) # type: ignore
            
            # --- ACTUALIZACIÓN VISUAL POR GENERACIÓN ---
            # Extraer datos de la población actual
            riesgos_gen = [ind.fitness.values[1] for ind in pop]
            retornos_gen = [ind.fitness.values[0] for ind in pop]
            
            # Actualizar métricas intermedias
            sharpes_gen = [(ret - RF) / rsk if rsk > 0 else 0 for ret, rsk in zip(retornos_gen, riesgos_gen)]
            mejor_sharpe_gen = max(sharpes_gen)
            if mejor_sharpe_gen > mejor_sharpe_historico:
                mejor_sharpe_historico = mejor_sharpe_gen
                
            # Renderizar el gráfico de dispersión (el "Enjambre")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=riesgos_gen, y=retornos_gen, mode='markers',
                name=f'Población Gen {gen+1}',
                marker=dict(color='rgba(0, 150, 255, 0.6)', size=8, line=dict(width=1, color='DarkSlateGrey'))
            ))
            fig.update_layout(
                title=f'Búsqueda NSGA-II - Generación {gen+1}/{GENERACIONES}',
                xaxis_title='Riesgo (Volatilidad)', yaxis_title='Retorno Esperado',
                xaxis=dict(range=[min(riesgos_gen)*0.9, max(riesgos_gen)*1.1]), # Mantener la escala estable ayuda a la animación
                yaxis=dict(range=[min(retornos_gen)*0.9, max(retornos_gen)*1.1]),
                hovermode='closest'
            )
            grafico_placeholder.plotly_chart(fig, use_container_width=True)
            metricas_placeholder.info(f"🧬 Procesando Generación {gen+1} | Mejor Sharpe Actual: {mejor_sharpe_historico:.4f}")
            
            # Pequeña pausa para apreciar el movimiento (ajusta a tu gusto)
            time.sleep(0.05) 
            
        # --- FIN DEL BUCLE: EXTRAER RESULTADOS FINALES ---
        codigo_placeholder.markdown(renderizar_codigo(13), unsafe_allow_html=True)
        frente_pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        riesgos = [ind.fitness.values[1] for ind in frente_pareto]
        retornos_pareto = [ind.fitness.values[0] for ind in frente_pareto]
        sharpes = [(ret - RF) / rsk if rsk > 0 else 0 for ret, rsk in zip(retornos_pareto, riesgos)]
        
        idx_mejor_sharpe = np.argmax(sharpes)
        mejor_ind = frente_pareto[idx_mejor_sharpe]
        mejor_pesos = np.array(mejor_ind) / np.sum(mejor_ind)
        mejor_ret = retornos_pareto[idx_mejor_sharpe]
        mejor_rsk = riesgos[idx_mejor_sharpe]
        mejor_sharpe = sharpes[idx_mejor_sharpe]

        metricas_placeholder.success("¡Evolución completada con éxito!")

        # --- 5. VISUALIZACIÓN FINAL (Estática) ---
        st.divider()
        st.subheader("Resultados del Mejor Individuo (Máximo Sharpe Evolutivo)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe Ratio Evolutivo", f"{mejor_sharpe:.4f}")
        c2.metric("Retorno Esperado", f"{mejor_ret*100:.2f}%")
        c3.metric("Riesgo (Volatilidad)", f"{mejor_rsk*100:.2f}%")

        col_g1, col_g2 = st.columns([2, 1])
        
        with col_g1:
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Scatter(
                x=riesgos, y=retornos_pareto, mode='markers',
                name='Frente de Pareto Final',
                marker=dict(color='blue', size=8, opacity=0.7)
            ))
            fig_pareto.add_trace(go.Scatter(
                x=[mejor_rsk], y=[mejor_ret], mode='markers',
                name='Mejor Sharpe Evolutivo',
                marker=dict(color='red', symbol='star', size=16)
            ))
            fig_pareto.update_layout(
                title='Frente de Pareto Obtenido por NSGA-II',
                xaxis_title='Riesgo (Volatilidad)',
                yaxis_title='Retorno Esperado',
                hovermode='closest'
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

        with col_g2:
            df_pesos_ga = pd.DataFrame({'Ticker': TICKERS_VALIDOS, 'Peso': mejor_pesos})
            df_pesos_ga = df_pesos_ga[df_pesos_ga['Peso'] > 0.01]
            fig_pie_ga = px.pie(
                df_pesos_ga, values='Peso', names='Ticker',
                title='Composición Evolutiva', hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie_ga, use_container_width=True)

        # --- 6. EXPORTACIÓN ---
        metrics_m2 = {
            "tickers": TICKERS_VALIDOS,
            "nsga2_max_sharpe": {
                "retorno": float(mejor_ret),
                "riesgo": float(mejor_rsk),
                "sharpe": float(mejor_sharpe),
                "pesos": dict(zip(TICKERS_VALIDOS, mejor_pesos.round(4).tolist()))
            }
        }
        with open("resultados_m2.json", "w", encoding="utf-8") as f:
            json.dump(metrics_m2, f, ensure_ascii=False, indent=2)

        buffer = io.BytesIO()
        df_export = pd.DataFrame({'Ticker': TICKERS_VALIDOS, 'Peso (%)': (mejor_pesos * 100).round(2)})
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Pesos_NSGA2')
        buffer.seek(0)

        st.download_button(
            label="📥 Descargar Pesos Evolutivos (Excel)",
            data=buffer,
            file_name="pesos_nsga2.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👈 Presiona el botón de arriba para iniciar la simulación genética.")
