---
name: aerospace-telemetry-research
description: Engineering research protocols and methodologies to derive transition matrices, emission matrices, and initial states from real spacecraft telemetry, reliability physics (MIL-HDBK/ECSS), and ground station link budgets.
---

# Aerospace Telemetry Parameter Research & Grounding Skill

Esta skill describe la metodología de ingeniería de sistemas espaciales para obtener y calibrar de forma realista los parámetros $(A, B, \pi)$ de un Modelo Oculto de Markov (HMM).

---

## 1. Cómo Obtener la Matriz de Transición de Estados ($A$)

La matriz de transición $A$ modela la probabilidad de que la nave cambie de condición de salud interna en una hora ($\Delta t = 1\text{ h}$). En la industria aeroespacial (NASA / ESA) se deriva mediante tres fuentes:

### A. Física de Fallos y Tasas FIT (MIL-HDBK-217F / ECSS-Q-ST-30-02C)
Cada componente crítico de la nave (giróscopos, celdas solares, tubos TWTA, ruedas de reacción, microprocesadores rad-hard) cuenta con una tasa de fallas conocida $\lambda$ expresada en **FIT** (*Failures in Time*: fallos por cada $10^9$ horas operativas):

1. **Tasa de transición continua $\to$ discreta:**
   La tasa de transición infinitesimal $q_{ij}$ se discretiza para el paso horario $\Delta t$:
   $$A_{ij} \approx q_{ij} \Delta t \quad (i \neq j)$$
   $$A_{ii} = 1 - \sum_{j \neq i} A_{ij}$$

2. **Ejemplo en misión orbital:**
   - La tasa acumulada de fallos menores de subsistemas (potencia, actitud, térmico) es $\lambda_{\text{degrad}} \approx 8 \times 10^{-2} \text{ h}^{-1} \implies A_{01} = 0.08$.
   - La tasa de fallo catastrófico súbito (ej. impacto de micrometeorito crítico, latch-up por radiación cósmica) es $\lambda_{\text{crit}} \approx 2 \times 10^{-2} \text{ h}^{-1} \implies A_{02} = 0.02$.
   - Persistencia nominal: $A_{00} = 1 - (0.08 + 0.02) = 0.90$.

### B. Análisis FMECA (Failure Modes, Effects, and Criticality Analysis)
El análisis FMECA clasifica las anomalías en niveles de severidad:
- **Nivel I (Menor / Degradación):** Pérdida de redundancia en bus de corriente o calentamiento de amplificador.
- **Nivel II (Crítico):** Falla de computadora de a bordo principal o desorientación del sensor estelar (*Star Tracker*).
Las probabilidades de propagación de un fallo Nivel I hacia Nivel II definen el término $A_{12} = 0.15$.

---

## 2. Cómo Obtener la Matriz de Emisión de Telemetría ($B$)

La matriz $B$ conecta el estado físico interno del satélite con lo que recibe la antena terrestre en la estación DSN (Goldstone, Madrid, Canberra).

Se deriva mediante el **Balance de Enlace de Radiofrecuencia (Link Budget)**:

$$\frac{E_b}{N_0} = EIRP + \left(\frac{G}{T}\right) - L_{\text{espacio}} - L_{\text{apuntamiento}} - k_B - R_{\text{datos}}$$

A partir de la relación energía de bit a densidad de ruido ($E_b/N_0$), se calcula la probabilidad de error de bit ($BER$) y la tasa de pérdida de tramas de telemetría (*Frame Error Rate - FER*):

### A. Estado 0: Operación Nominal
- Antena de alta ganancia (HGA) orientada a Tierra con precisión $< 0.1^\circ$.
- Margen de enlace $> +12\text{ dB}$.
- $FER < 10^{-4} \implies$ **Señal Estable ($B_{00} = 0.85 - 0.95$)**. Las pequeñas pérdidas (12% intermitente, 3% pérdida) provienen de lluvia o atenuación troposférica en la estación terrena.

### B. Estado 1: Degradación de Sistemas
- La nave experimenta oscilación de actitud (*attitude wobble* de $\pm 2^\circ$) o reducción de potencia de RF de 20W a 8W.
- Margen de enlace marginal ($+2$ a $+4\text{ dB}$).
- Paquetes se descartan por sumas de verificación (*checksum errors*):
  - **Señal Intermitente ($B_{11} = 0.50$)**.
  - **Señal Estable ($B_{10} = 0.30$)** durante fases favorables de giro.
  - **Señal Perdida ($B_{12} = 0.20$)** en desalineaciones transitorias.

### C. Estado 2: Anomalía Crítica
- Desorientación total (*tumbling* de la nave) o fallo del subsistema de telecomunicaciones.
- La antena directiva pierde el vector Tierra ($L_{\text{apuntamiento}} > 25\text{ dB}$).
- El receptor en tierra cae bajo el umbral de sincronismo (*Carrier Lock Lost*):
  - **Señal Perdida ($B_{22} = 0.80$)**.
  - Raras ráfagas intermitentes ($B_{21} = 0.15$) o picos residuales ($B_{20} = 0.05$).

---

## 3. Cómo Obtener la Distribución Inicial ($\pi$)

La distribución inicial $\pi = P(S_0)$ proviene de la fase **LEOP (Launch and Early Orbit Phase)**:
- Post-separación del cohete lanzador:
  - Nominal ($85\%$): Separación limpia con giro amortiguado.
  - Degradación ($10\%$): Despliegue incompleto de paneles o giros residuales elevados.
  - Crítica ($5\%$): Falla de encendido de computadora o cortocircuito de batería en el despliegue.

---

## 4. Fuentes de Datos Abiertos de Telemetría Real de la NASA y ESA

Para entrenar y validar tu HMM con datos reales de misiones espaciales, utiliza estos repositorios públicos:

1. **NASA Planetary Data System (PDS):**
   - URL: `https://pds.nasa.gov/`
   - Contiene telemetría de ingeniería histórica de sondas como Voyager, Cassini, Mars Odyssey y Curiosity.
2. **NASA SMAP Telemetry Dataset:**
   - Registros horarios de anomalías de telemetría de satélites en órbita terrestre baja (LEO).
3. **NASA Lessons Learned Information System (LLIS):**
   - Registros de fallos en vuelo con tiempos de diagnóstico y secuencias de eventos de Safe-Mode.
4. **ESA Planetary Science Archive (PSA):**
   - Telemetría de misiones como Mars Express, Rosetta y BepiColombo.
