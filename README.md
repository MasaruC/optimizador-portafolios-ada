# 📈 Sistema de Optimización de Portafolios con GA y DP

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-success)

Proyecto integrador para el curso de **Análisis y Diseño de Algoritmos** (UNMSM). Este sistema implementa y compara tres paradigmas algorítmicos de optimización financiera (Markowitz, NSGA-II y Programación Dinámica) bajo un entorno de backtesting unificado y riguroso, evaluando el impacto real de los costos de transacción.

---

## 🚀 Características Principales

* **Módulo 1 (Markowitz):** Optimización clásica de Media-Varianza usando Programación Cuadrática Secuencial (SLSQP) para encontrar el portafolio de Máximo Sharpe y trazar la Frontera Eficiente.
* **Módulo 2 (NSGA-II):** Algoritmo Genético Multiobjetivo (mediante DEAP) que approxima el Frente de Pareto óptimo sin depender de supuestos de diferenciabilidad.
* **Módulo 3 (Programación Dinámica):** Resolución de la Ecuación de Bellman mediante inducción hacia atrás sobre una grilla combinatoria discreta para determinar la política óptima de rebalanceo considerando costos de transacción.
* **Módulo 4 (Backtesting Unificado):** Motor de simulación que evalúa 7 estrategias simultáneamente bajo las mismas condiciones históricas, aplicando costos de transacción reales y calculando métricas financieras (Sharpe, Sortino, Max Drawdown).
* **Interfaz Educativa:** Visualización de la ejecución de los algoritmos en tiempo real (paso a paso) sincronizada con paneles de pseudocódigo estilo IDE.

---

## 🛠️ Stack Tecnológico

* **Frontend/Interfaz:** Streamlit, Plotly.
* **Backend/Cálculo:** Python, NumPy, Pandas, SciPy, DEAP.
* **Datos:** Yahoo Finance API (`yfinance`).
* **Arquitectura:** Protocolo BATD (Boundary, Application, Task, Data).

---

## ⚙️ Instalación y Ejecución Local

Sigue estos pasos para ejecutar el sistema en tu máquina local.

### 1. Pre-requisitos
Asegúrate de tener instalado Python 3.10 o superior y Git.

### 2. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

### 3. Crear y activar entorno virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar dependencias
Crea un archivo `requirements.txt` con las siguientes librerías (o instala manualmente):
```bash
pip install streamlit yfinance numpy pandas scipy deap plotly openpyxl
```

### 5. Ejecutar la aplicación
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador web predeterminado (`http://localhost:8501`).

---

## 📋 Guía de Uso

Para que el sistema funcione correctamente y el Módulo 4 pueda realizar la comparación, debes ejecutar los módulos en orden:

1. **Configuración Global:** En el menú lateral izquierdo (`app.py`), define los tickers, fechas, capital inicial, frecuencia de rebalanceo y parámetros algorítmicos.
2. **Módulo 1:** Ve a la página del Módulo 1 y pulsa "Ejecutar y Visualizar Algoritmo". Esto generará el archivo `resultados_m1.json`.
3. **Módulo 2:** Ve al Módulo 2 y pulsa "Ejecutar Evolución NSGA-II". Esto generará `resultados_m2.json`.
4. **Módulo 3:** Ve al Módulo 3 y pulsa "Calcular Política Óptima". Esto generará `resultados_m3.json`.
5. **Módulo 4:** Finalmente, ve al Módulo 4. El sistema leerá los JSON previos, descargará el histórico unificado y ejecutará el backtest de las 7 estrategias.

> **💡 Tip:** Si cambias las fechas o tickers en el menú lateral, recuerda volver a ejecutar los Módulos 1, 2 y 3 para actualizar los archivos JSON. Si el Módulo 4 no refleja los cambios, usa el botón "🔄 Forzar Recarga de Datos" en el sidebar.

---

## 📊 Hallazgos Financieros y Conclusiones

El motor unificado de backtesting reveló conclusiones empíricas clave sobre el comportamiento de los algoritmos en un horizonte de 10 años con datos reales:

1. **Victoria de los Enfoques Continuos:** Las estrategias de Markowitz y NSGA-II con rebalanceo mensual superaron ampliamente al resto, alcanzando una riqueza final de ~$251k. La paridad entre ambos demuestra que los algoritmos genéticos pueden igualar a la optimización matemática sin requerir supuestos de convexidad.
2. **Derrota de la Programación Dinámica:** A pesar de su elegancia teórica, la DP finalizó con pérdidas ($89k). Su grilla discreta (saltos del 10%) y su política de "inactividad para ahorrar comisiones" provocaron una deriva extrema de los pesos, resultando en un Máximo Drawdown devastador (-69%).
3. **Rebalanceo vs. Buy & Hold:** El rebalanceo forzado mensual superó al Buy & Hold por más de $44k. Pagar comisiones (~$850) para forzar la toma de ganancias y comprar activos devaluados (reversión a la media) demostró ser una estrategia altamente rentable a largo plazo.

---

## 👥 Equipo de Desarrollo (Grupo G4 - UNMSM)

* Juan Masaru Campos Luque
* Angel Gabriel León García
* Alvaro Mathias Melendez Bustamante
* Basadre Pérez Christian Raúl
* Herrera Chavarria Ronnie Rodrigo
* Córdova Guerra Josué Rodrigo
* Cabrejos Palomino Christian Daniel
* Saccaco Oscco Christopher

**Docente:** Ernesto Cancho-Rodriguez

---

## 📄 Licencia

Este proyecto es de uso académico bajo los lineamientos de la Universidad Nacional Mayor de San Marcos. Los resultados generados son simulaciones educativas y no constituyen asesoramiento financiero real.
