---
name: spacecraft-telemetry-hmm
description: Specialized skill for monitoring spacecraft health, running HMM diagnostics, simulating orbital telemetry, and executing aerospace contingency protocols.
---

# Spacecraft Telemetry & Health Monitoring (HMM) Skill

Este skill proporciona procedimientos estandarizados para ingenieros de vuelo y operadores de estaciones terrenas que monitorean subsistemas satelitales mediante Modelos Ocultos de Markov (HMM).

## Capacidades

1. **Inferencia de Salud Interna:** Decodificar el estado de salud interno de la nave (`Nominal`, `Degradación`, `Crítica`) a partir de la calidad de señal de radiofrecuencia observada desde estaciones DSN/ESTRACK (`Estable`, `Intermitente`, `Perdida`).
2. **Algoritmo de Viterbi:** Reconstrucción óptima en espacio logarítmico de la trayectoria histórica de la nave para análisis post-vuelo o auditoría de anomalías.
3. **Procesador de Streaming:** Estimación recursiva en vivo $P(S_t \mid O_{1:t})$ con el Algoritmo Forward y filtrado bayesiano.
4. **Análisis de Fiabilidad Monte Carlo:** Proyección de supervivencia $R(t)$ y tiempo medio hasta la primera falla (MTTF).
5. **Protocolos de Contingencia:**
   - **Nominal (Verde):** Operación científica al 100%, enlace en Banda X.
   - **Degradación (Amarillo):** Alerta Nivel 2, telemetría a 1 Hz, aislamiento de instrumentos secundarios.
   - **Crítica (Rojo):** Alerta Nivel 1, activación inmediata de **Safe-Hold Mode**, orientación autónoma solar, conmutación a transpondedor LGA omnidireccional en Banda S.

## Rutinas Disponibles en el Proyecto

- **Simulación Principal:** `python simulacion_hmm_espacial.py`
- **Grafo de Estados NetworkX:** `src/markov_graph.py`
- **Calibración Baum-Welch:** `src/baum_welch.py`
- **Simulador Monte Carlo:** `src/monte_carlo.py`
- **Monitor en Tiempo Real:** `src/telemetry_stream.py`

## Parámetros de Referencia de la Misión

- **Matriz de Transición A:**
  - Nominal $\to$ [0.90, 0.08, 0.02]
  - Degradación $\to$ [0.40, 0.45, 0.15]
  - Crítica $\to$ [0.05, 0.15, 0.80]
- **Matriz de Emisión B:**
  - Nominal $\to$ [0.85, 0.12, 0.03]
  - Degradación $\to$ [0.30, 0.50, 0.20]
  - Crítica $\to$ [0.05, 0.15, 0.80]
- **Distribución Estacionaria Asintótica:**
  - $\pi_\infty = [58.9\%, 19.8\%, 21.3\%]$
