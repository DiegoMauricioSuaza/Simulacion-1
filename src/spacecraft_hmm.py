"""
================================================================================
SPACE TELEMETRY HIDDEN MARKOV MODEL (HMM) & MISSION CONTROL MONITOR
Simulador de Monitoreo de Salud de Naves Espaciales basado en Telemetría Terrestre
================================================================================

Este módulo implementa:
1. Modelo Oculto de Markov (HMM) completo desde cero (NumPy).
2. Algoritmo de Viterbi en espacio logarítmico (robusto ante underflow numérico).
3. Algoritmo Forward para estimación de probabilidades a posteriori en tiempo real.
4. Generador de telemetría y modulación de señal portadora (Heartbeat).
5. Sistema de contingencias automáticas y análisis de confiabilidad/riesgo.
6. Dashboard de telemetría estilo Sala de Control de Misión (NASA/ESA aesthetic).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Any

# =====================================================================
# 1. CLASE HMM PARA MONITOREO DE TELEMETRÍA ESPACIAL
# =====================================================================

class SpacecraftHMM:
    """
    Modelo Oculto de Markov para monitoreo y diagnóstico de salud de naves espaciales.
    
    Estados Ocultos (Salud de la Nave):
        0: Operación Nominal
        1: Degradación de Sistemas
        2: Anomalía Crítica
        
    Observaciones (Calidad de Señal de Telemetría):
        0: Señal Estable
        1: Señal Intermitente
        2: Señal Perdida
    """
    
    STATE_NAMES = ['Operación Nominal', 'Degradación de Sistemas', 'Anomalía Crítica']
    STATE_COLORS = ['#00E676', '#FFD600', '#FF1744']  # Verde, Amarillo, Rojo
    OBS_NAMES = ['Señal Estable', 'Señal Intermitente', 'Señal Perdida']
    OBS_MARKERS = ['o', '^', 'x']

    def __init__(self, 
                 transition_matrix: np.ndarray = None, 
                 emission_matrix: np.ndarray = None, 
                 initial_distribution: np.ndarray = None):
        """
        Inicializa las matrices del modelo según especificaciones de misión.
        """
        # Matriz de Transición A (3x3): P(S_t | S_{t-1})
        if transition_matrix is not None:
            self.A = np.array(transition_matrix, dtype=np.float64)
        else:
            self.A = np.array([
                [0.90, 0.08, 0.02],  # Nominal -> [Nominal, Degradación, Crítica]
                [0.40, 0.45, 0.15],  # Degradación -> [Nominal, Degradación, Crítica]
                [0.05, 0.15, 0.80]   # Crítica -> [Nominal, Degradación, Crítica]
            ], dtype=np.float64)

        # Matriz de Emisión B (3x3): P(O_t | S_t)
        if emission_matrix is not None:
            self.B = np.array(emission_matrix, dtype=np.float64)
        else:
            self.B = np.array([
                [0.85, 0.12, 0.03],  # Nominal -> [Estable, Intermitente, Perdida]
                [0.30, 0.50, 0.20],  # Degradación -> [Estable, Intermitente, Perdida]
                [0.05, 0.15, 0.80]   # Crítica -> [Estable, Intermitente, Perdida]
            ], dtype=np.float64)

        # Distribución Inicial Pi: P(S_0)
        if initial_distribution is not None:
            self.pi = np.array(initial_distribution, dtype=np.float64)
        else:
            self.pi = np.array([0.85, 0.10, 0.05], dtype=np.float64)

        self.n_states = len(self.STATE_NAMES)
        self.n_obs = len(self.OBS_NAMES)
        
        # Validar consistencia estocástica (filas deben sumar 1.0)
        assert np.allclose(self.A.sum(axis=1), 1.0), "La matriz A debe ser estocástica por filas"
        assert np.allclose(self.B.sum(axis=1), 1.0), "La matriz B debe ser estocástica por filas"
        assert np.isclose(self.pi.sum(), 1.0), "El vector Pi debe sumar 1.0"

    # -----------------------------------------------------------------
    # SIMULACIÓN DE TELEMETRÍA (72 HORAS DE MISIÓN)
    # -----------------------------------------------------------------
    def simulate(self, n_hours: int = 72, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula una trayectoria de misión de n_hours mediciones horarias.
        
        Retorna:
            hidden_states: Secuencia real de estados de salud (0, 1, 2)
            observations: Secuencia de observaciones de señal recibida (0, 1, 2)
        """
        rng = np.random.default_rng(seed)
        
        states = np.zeros(n_hours, dtype=int)
        observations = np.zeros(n_hours, dtype=int)

        # Estado inicial t=0
        states[0] = rng.choice(self.n_states, p=self.pi)
        observations[0] = rng.choice(self.n_obs, p=self.B[states[0]])

        # Evolución temporal t=1 hasta n_hours-1
        for t in range(1, n_hours):
            # Transición de estado oculta de Markov
            states[t] = rng.choice(self.n_states, p=self.A[states[t - 1]])
            # Emisión de señal observada
            observations[t] = rng.choice(self.n_obs, p=self.B[states[t]])

        return states, observations

    # -----------------------------------------------------------------
    # ALGORITMO DE VITERBI (INFERENCIA ÓPTIMA EN LOG-DOMAIN)
    # -----------------------------------------------------------------
    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Implementa el algoritmo de Viterbi para encontrar la secuencia más probable
        de estados de salud oculta dada la telemetría observada.
        
        Se ejecuta en escala logarítmica (log-probabilities) para garantizar máxima
        estabilidad numérica y evitar 'underflow' aritmético en secuencias largas.
        
        Retorna:
            best_path: Secuencia óptima de estados estimados [0, 1, 2]
            best_log_prob: Log-probabilidad de la ruta conjunta más probable
        """
        T = len(observations)
        K = self.n_states

        # Evitar log(0) con epsilon muy pequeño
        eps = 1e-16
        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)

        # Tablas de programación dinámica
        # V[t, k]: Máxima log-probabilidad de una ruta que termina en el estado k en el tiempo t
        V = np.zeros((T, K), dtype=np.float64)
        # backpointer[t, k]: Guarda el estado predecesor óptimo en t-1 para el estado k en t
        backpointer = np.zeros((T, K), dtype=int)

        # Paso 1: Inicialización (t = 0)
        first_obs = observations[0]
        V[0, :] = log_pi + log_B[:, first_obs]

        # Paso 2: Recursión (t = 1 .. T-1)
        for t in range(1, T):
            obs = observations[t]
            for j in range(K):
                # V[t-1, i] + log(A_ij) para todos los posibles estados previos i
                prev_probs = V[t - 1, :] + log_A[:, j]
                best_prev = np.argmax(prev_probs)
                
                backpointer[t, j] = best_prev
                V[t, j] = prev_probs[best_prev] + log_B[j, obs]

        # Paso 3: Terminación
        best_last_state = np.argmax(V[T - 1, :])
        best_log_prob = V[T - 1, best_last_state]

        # Paso 4: Backtracking (Reconstrucción del camino óptimo)
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = best_last_state

        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path, float(best_log_prob)

    # -----------------------------------------------------------------
    # ALGORITMO FORWARD (PROBABILIDADES POSTERIORES EN TIEMPO REAL)
    # -----------------------------------------------------------------
    def forward_filtering(self, observations: np.ndarray) -> np.ndarray:
        """
        Calcula la creencia filtrada en tiempo real P(S_t = k | O_1:t) usando 
        el algoritmo Forward con factores de escalado c_t.
        Útil para telemetría en vivo y display de confianza del operador de vuelo.
        """
        T = len(observations)
        K = self.n_states
        alpha = np.zeros((T, K), dtype=np.float64)

        # Inicialización t=0
        alpha_raw = self.pi * self.B[:, observations[0]]
        scale = alpha_raw.sum()
        alpha[0, :] = alpha_raw / (scale if scale > 0 else 1.0)

        # Paso recursivo
        for t in range(1, T):
            obs = observations[t]
            prior = alpha[t - 1, :] @ self.A
            alpha_raw = prior * self.B[:, obs]
            scale = alpha_raw.sum()
            alpha[t, :] = alpha_raw / (scale if scale > 0 else 1.0)

        return alpha

    # -----------------------------------------------------------------
    # MATRIZ ESTACIONARIA Y ANÁLISIS DE RIESGO A LARGO PLAZO
    # -----------------------------------------------------------------
    def compute_steady_state(self) -> np.ndarray:
        """
        Calcula la distribución estacionaria pi_inf tal que: pi_inf * A = pi_inf
        Representa la fracción de tiempo asintótica en cada estado bajo operación continua.
        """
        # (A.T - I) * pi = 0 con restricción sum(pi) = 1
        K = self.n_states
        M = self.A.T - np.eye(K)
        M[-1, :] = 1.0  # Reemplazar última ecuación por sum(pi) = 1
        rhs = np.zeros(K)
        rhs[-1] = 1.0
        try:
            pi_inf = np.linalg.solve(M, rhs)
            return np.clip(pi_inf, 0, 1)
        except np.linalg.LinAlgError:
            # Aproximación por potencias si singular
            Ak = np.linalg.matrix_power(self.A, 1000)
            return Ak[0, :]

    def risk_profile_projection(self, n_steps: int = 72) -> np.ndarray:
        """
        Proyecta la evolución temporal de la probabilidad de cada estado
        iniciando desde pi: P(S_t) = pi * (A^t).
        """
        risk_evolution = np.zeros((n_steps, self.n_states))
        current_p = self.pi.copy()
        for t in range(n_steps):
            risk_evolution[t, :] = current_p
            current_p = current_p @ self.A
        return risk_evolution


# =====================================================================
# 2. PROTOCOLOS DE CONTINGENCIA Y ACCIÓN OPERATIVA
# =====================================================================

CONTINGENCY_PROTOCOLS = {
    0: {
        "estado": "Operación Nominal",
        "codigo": "STATUS-GREEN // NORMAL OPERATIONS",
        "color": "#00E676",
        "semaforo": "VERDE",
        "accion_primaria": "Continuar misión según plan de vuelo primario.",
        "procedimientos": [
            "Mantener cargas útiles científicas encendidas al 100%.",
            "Descarga de datos de ciencia y telemetría en banda X autorizada.",
            "Consumo energético en ciclo nominal de baterías y paneles.",
            "Comando de rutina para control de actitud mediante ruedas de reacción."
        ],
        "seguridad": "Nivel 0 (Seguro)"
    },
    1: {
        "estado": "Degradación de Sistemas",
        "codigo": "STATUS-AMBER // SYSTEM DEGRADATION DETECTED",
        "color": "#FFD600",
        "semaforo": "AMARILLO",
        "accion_primaria": "Alerta Nivel 2: Monitoreo intensivo y aislamiento de subsistemas.",
        "procedimientos": [
            "Interrumpir observaciones científicas secundarias no críticas.",
            "Ejecutar rutina BITE (Built-In Test Equipment) en subsistema de RF y AOCS.",
            "Aumentar frecuencia de sondeo de baliza (heartbeat) a 2x.",
            "Verificar temperatura de amplificadores TWT y estado de celdas solares.",
            "Preparar secuencias de conmutación a buses de alimentación redundantes."
        ],
        "seguridad": "Nivel 2 (Precaución)"
    },
    2: {
        "estado": "Anomalía Crítica",
        "codigo": "STATUS-RED // CRITICAL FAULT - SAFE HOLD MODE",
        "color": "#FF1744",
        "semaforo": "ROJO",
        "accion_primaria": "Alerta Nivel 1: ACTIVAR MODO SEGURO (SAFE-HOLD MODE) INMEDIATO.",
        "procedimientos": [
            "APAGADO DE EMERGENCIA de todos los instrumentos científicos.",
            "Orientar paneles solares hacia el vector Sol para preservar energía.",
            "Conmutar transmisor principal a Antena de Baja Ganancia (LGA) omnidireccional.",
            "Estabilizar actitud con propulsores de control a reacción en modo 'Sun-Pointing'.",
            "Emitir baliza de emergencia continua a la Red del Espacio Profundo (DSN).",
            "Convocar de inmediato al Comité de Resolución de Anomalías de Vuelo."
        ],
        "seguridad": "Nivel 1 (Crítico / Supervivencia)"
    }
}


# =====================================================================
# 3. CÁLCULO DE MÉTRICAS AVANZADAS DE MISIÓN
# =====================================================================

def calculate_mission_metrics(real_states: np.ndarray, 
                              observations: np.ndarray, 
                              viterbi_states: np.ndarray) -> Dict[str, Any]:
    """
    Calcula exhaustivamente todas las métricas de misión requeridas:
    - Disponibilidad total (% horas señal estable)
    - Horas y porcentaje en cada estado real e inferido
    - Retraso en detección de anomalías (latencia en horas)
    - Duración y episodios de pérdida de señal
    - Matriz de confusión y rendimiento del algoritmo Viterbi
    """
    total_hours = len(real_states)
    
    # 1. Disponibilidad de Señal
    stable_hours = int(np.sum(observations == 0))
    intermittent_hours = int(np.sum(observations == 1))
    lost_hours = int(np.sum(observations == 2))
    availability_pct = (stable_hours / total_hours) * 100.0

    # 2. Distribución de Estados Reales
    real_counts = [int(np.sum(real_states == s)) for s in range(3)]
    real_pcts = [(cnt / total_hours) * 100.0 for cnt in real_counts]

    # 3. Distribución de Estados Inferidos por Viterbi
    inferred_counts = [int(np.sum(viterbi_states == s)) for s in range(3)]
    inferred_pcts = [(cnt / total_hours) * 100.0 for cnt in inferred_counts]

    # 4. Duración de los Períodos de Pérdida de Señal (Loss of Signal - LOS)
    los_periods = []
    current_los = 0
    for obs in observations:
        if obs == 2:  # Señal Perdida
            current_los += 1
        else:
            if current_los > 0:
                los_periods.append(current_los)
                current_los = 0
    if current_los > 0:
        los_periods.append(current_los)

    max_los_hours = max(los_periods) if los_periods else 0
    mean_los_hours = float(np.mean(los_periods)) if los_periods else 0.0
    total_los_events = len(los_periods)

    # 5. Retraso en la Detección de Anomalías (Latency)
    # Identificar bloques continuos de anomalía real (Degradación o Crítica: estado != 0)
    detection_delays = []
    in_anomaly = False
    anomaly_start = -1

    for t in range(total_hours):
        is_real_anomaly = (real_states[t] in [1, 2])
        if is_real_anomaly and not in_anomaly:
            in_anomaly = True
            anomaly_start = t
            # Buscar el primer instante t_detect >= anomaly_start donde Viterbi detectó anomalía
            detected_at = None
            for t_eval in range(anomaly_start, total_hours):
                if viterbi_states[t_eval] in [1, 2]:
                    detected_at = t_eval
                    break
            if detected_at is not None:
                delay = detected_at - anomaly_start
                detection_delays.append(delay)
            else:
                # No detectada dentro de la ventana de simulación
                detection_delays.append(total_hours - anomaly_start)
        elif not is_real_anomaly and in_anomaly:
            in_anomaly = False

    avg_detection_delay = float(np.mean(detection_delays)) if detection_delays else 0.0
    max_detection_delay = int(np.max(detection_delays)) if detection_delays else 0

    # 6. Matriz de Confusión (Filas: Real, Columnas: Inferido)
    conf_matrix = np.zeros((3, 3), dtype=int)
    for r, p in zip(real_states, viterbi_states):
        conf_matrix[r, p] += 1

    accuracy = float(np.trace(conf_matrix) / total_hours) * 100.0

    # Precision, Recall y F1 por clase
    precision = np.zeros(3)
    recall = np.zeros(3)
    f1 = np.zeros(3)
    for i in range(3):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i] = (2 * precision[i] * recall[i] / (precision[i] + recall[i])) if (precision[i] + recall[i]) > 0 else 0.0

    return {
        "total_hours": total_hours,
        "availability_pct": availability_pct,
        "stable_hours": stable_hours,
        "intermittent_hours": intermittent_hours,
        "lost_hours": lost_hours,
        "real_counts": real_counts,
        "real_pcts": real_pcts,
        "inferred_counts": inferred_counts,
        "inferred_pcts": inferred_pcts,
        "los_periods": los_periods,
        "total_los_events": total_los_events,
        "max_los_hours": max_los_hours,
        "mean_los_hours": mean_los_hours,
        "detection_delays": detection_delays,
        "avg_detection_delay": avg_detection_delay,
        "max_detection_delay": max_detection_delay,
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =====================================================================
# 4. GENERACIÓN DE SEÑAL DE "LATIDOS" (HEARTBEAT TELEMETRY)
# =====================================================================

def generate_heartbeat_waveform(observations: np.ndarray, samples_per_hour: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una forma de onda analógica modulada del latido de telemetría (Heartbeat carrier)
    a lo largo de las 72 horas para simular la señal de RF recibida por las antenas de tierra.
    
    - Señal Estable: Pulso periódico limpio con portadora continua (amplitud 1.0, ruido bajo)
    - Señal Intermitente: Portadora con atenuación fluctuante y jitter (amplitud 0.4 - 0.7)
    - Señal Perdida: Portadora ausente, solo ruido blanco térmico de fondo (amplitud ~0.05)
    """
    total_hours = len(observations)
    total_samples = total_hours * samples_per_hour
    t_wave = np.linspace(0, total_hours, total_samples, endpoint=False)
    waveform = np.zeros(total_samples)

    rng = np.random.default_rng(101)

    for h_idx, obs in enumerate(observations):
        start = h_idx * samples_per_hour
        end = (h_idx + 1) * samples_per_hour
        t_local = t_wave[start:end] - h_idx  # 0 a 1 dentro de la hora

        # Frecuencia de latido base (pulso de sincronismo)
        carrier = np.sin(2 * np.pi * 4 * t_local)
        
        # Pulso tipo latido cardíaco / telemetría periódica
        pulse = np.exp(-((t_local - 0.25) ** 2) / 0.01) * 1.5 - np.exp(-((t_local - 0.35) ** 2) / 0.005) * 0.8

        if obs == 0:  # Señal Estable
            amp = 1.0
            noise = rng.normal(0, 0.04, samples_per_hour)
            segment = amp * (0.3 * carrier + pulse) + noise
        elif obs == 1:  # Señal Intermitente
            amp = rng.uniform(0.3, 0.7, samples_per_hour)
            dropout = rng.choice([0.0, 1.0], size=samples_per_hour, p=[0.25, 0.75])
            noise = rng.normal(0, 0.12, samples_per_hour)
            segment = amp * dropout * (0.3 * carrier + pulse) + noise
        else:  # Señal Perdida
            noise = rng.normal(0, 0.08, samples_per_hour)
            segment = noise  # Solo ruido térmico receptor

        waveform[start:end] = segment

    return t_wave, waveform


# =====================================================================
# 5. VISUALIZACIÓN 1: TIMELINE DE TELEMETRÍA, HEARTBEAT Y ALERTAS
# =====================================================================

def plot_mission_timeline(real_states: np.ndarray, 
                          observations: np.ndarray, 
                          viterbi_states: np.ndarray, 
                          alpha_probs: np.ndarray,
                          save_path: str = None):
    """
    Genera la figura de la línea de tiempo de la misión con estética aeroespacial:
    1. Timeline de la señal con franjas coloreadas del estado real y alertas.
    2. Comparación Estado Real vs Inferido por Viterbi.
    3. Probabilidades a posteriori en tiempo real (Forward Algorithm).
    4. Señal de latidos (Heartbeat RF carrier wave).
    """
    total_hours = len(real_states)
    hours = np.arange(total_hours)

    # Configuración de estilo Dark Aerospace
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 14), facecolor='#0B0F19')
    gs = GridSpec(4, 1, height_ratios=[2.2, 1.3, 1.5, 1.4], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])  # Observaciones + Fondo de Salud Real + Alertas
    ax1 = fig.add_subplot(gs[1], sharex=ax0)  # Estado Real vs Viterbi
    ax2 = fig.add_subplot(gs[2], sharex=ax0)  # Probabilidades Posteriores (Forward)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)  # Latido / RF Carrier

    state_colors = SpacecraftHMM.STATE_COLORS
    state_names = SpacecraftHMM.STATE_NAMES
    obs_names = SpacecraftHMM.OBS_NAMES

    # -------------------------------------------------------------
    # SUBPLOT 0: SEÑAL DE TELEMETRÍA CON FONDO DE SALUD REAL Y ALERTAS
    # -------------------------------------------------------------
    ax0.set_facecolor('#101626')
    ax0.set_title("✦ LÍNEA DE TIEMPO DE TELEMETRÍA ESPACIAL Y SALUD REAL DE LA NAVE (72 HORAS DE MISIÓN) ✦", 
                  fontsize=13, fontweight='bold', color='#E0E6ED', pad=12)

    # Dibujar fondo coloreado según estado de salud real en cada hora
    for t in range(total_hours):
        st = real_states[t]
        ax0.axvspan(t - 0.5, t + 0.5, color=state_colors[st], alpha=0.18, zorder=0)

    # Mapeo numérico para graficar observaciones: 2: Estable, 1: Intermitente, 0: Perdida
    obs_plot_map = {0: 2, 1: 1, 2: 0}
    obs_y = np.array([obs_plot_map[o] for o in observations])

    # Trazo escalonado de la señal recibida
    ax0.step(hours, obs_y, where='mid', color='#00E5FF', linewidth=2.0, label='Calidad de Señal Recibida (Tierra)', zorder=3)
    
    # Marcadores específicos para cada tipo de observación
    for o_val, y_val, marker, color, label in [
        (0, 2, 'o', '#00E676', 'Señal Estable'),
        (1, 1, '^', '#FFD600', 'Señal Intermitente'),
        (2, 0, 'X', '#FF1744', 'Señal Perdida')
    ]:
        mask = (observations == o_val)
        ax0.scatter(hours[mask], obs_y[mask], marker=marker, color=color, s=70, edgecolor='#0B0F19', 
                    linewidth=1.2, zorder=4, label=f'Evento: {label}')

    # Indicadores de Alerta Crítica (cuando la nave entra en anomalía crítica)
    for t in range(total_hours):
        if real_states[t] == 2:  # Anomalía Crítica
            ax0.scatter(t, 2.3, marker='v', color='#FF1744', s=120, zorder=5)
            if t == 0 or real_states[t - 1] != 2:
                ax0.annotate(f"¡ALERTA T+{t:02d}h!\nAnomalía Crítica",
                             xy=(t, 2.3), xytext=(t, 2.7),
                             arrowprops=dict(facecolor='#FF1744', edgecolor='#FFFFFF', arrowstyle="->", lw=1.2),
                             fontsize=8, fontweight='bold', color='#FF5252', ha='center',
                             bbox=dict(boxstyle="round,pad=0.2", fc="#260C14", ec="#FF1744", lw=1.2))

    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels(['Perdida', 'Intermitente', 'Estable'], fontsize=10, fontweight='bold')
    ax0.set_ylabel("Calidad de Señal", fontsize=10, color='#90CAF9')
    ax0.set_ylim(-0.6, 3.2)
    ax0.grid(True, linestyle=':', alpha=0.3, color='#455A64')

    # Leyenda de fondos de salud real
    patch_nom = mpatches.Patch(color=state_colors[0], alpha=0.4, label='Fondo: Operación Nominal')
    patch_deg = mpatches.Patch(color=state_colors[1], alpha=0.4, label='Fondo: Degradación')
    patch_crt = mpatches.Patch(color=state_colors[2], alpha=0.4, label='Fondo: Anomalía Crítica')
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles=handles + [patch_nom, patch_deg, patch_crt], loc='upper right', 
               ncol=3, fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # -------------------------------------------------------------
    # SUBPLOT 1: ESTADO REAL VS INFERIDO POR ALGORITMO DE VITERBI
    # -------------------------------------------------------------
    ax1.set_facecolor('#101626')
    ax1.set_title("DECODIFICACIÓN DE ESTADO OCULTO: REAL (GROUND TRUTH) VS VITERBI (INFERIDO)", 
                  fontsize=10, fontweight='bold', color='#90CAF9', loc='left')

    ax1.plot(hours, real_states, color='#76FF03', lw=2.5, label='Salud Real (Oculta en la nave)', zorder=3)
    ax1.step(hours, viterbi_states, where='mid', color='#FF4081', lw=2.0, linestyle='--', 
             label='Viterbi Inferido (Tierra)', zorder=4)

    # Sombrear discrepancias (errores de inferencia)
    discrepancies = (real_states != viterbi_states)
    for t in range(total_hours):
        if discrepancies[t]:
            ax1.axvspan(t - 0.5, t + 0.5, color='#FF5252', alpha=0.35, hatch='//')

    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Nominal', 'Degradación', 'Crítica'], fontsize=9)
    ax1.set_ylabel("Estado de Salud", fontsize=10, color='#90CAF9')
    ax1.set_ylim(-0.4, 2.4)
    ax1.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax1.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # -------------------------------------------------------------
    # SUBPLOT 2: PROBABILIDADES FILTRADAS EN TIEMPO REAL (FORWARD)
    # -------------------------------------------------------------
    ax2.set_facecolor('#101626')
    ax2.set_title("CREENCIA POSTERIOR EN VIVO: P(S_t | O_1:t) - ESTIMADOR FORWARD DE CONTROL DE VUELO", 
                  fontsize=10, fontweight='bold', color='#90CAF9', loc='left')

    ax2.plot(hours, alpha_probs[:, 0], color=state_colors[0], lw=1.8, label='P(Nominal)')
    ax2.plot(hours, alpha_probs[:, 1], color=state_colors[1], lw=1.8, label='P(Degradación)')
    ax2.plot(hours, alpha_probs[:, 2], color=state_colors[2], lw=2.2, label='P(Crítica)')
    ax2.fill_between(hours, 0, alpha_probs[:, 2], color=state_colors[2], alpha=0.25)

    ax2.set_ylabel("Probabilidad", fontsize=10, color='#90CAF9')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # -------------------------------------------------------------
    # SUBPLOT 3: FORMA DE ONDA DE LATIDOS (HEARTBEAT TELEMETRY)
    # -------------------------------------------------------------
    ax3.set_facecolor('#0A0E1A')
    ax3.set_title("TELEMETRY HEARTBEAT WAVEFORM (SEÑAL DE BALIZA RF / PORTADORA RECEPCIÓN TIERRA)", 
                  fontsize=10, fontweight='bold', color='#00E5FF', loc='left')

    t_wave, waveform = generate_heartbeat_waveform(observations, samples_per_hour=40)
    ax3.plot(t_wave, waveform, color='#00E5FF', lw=0.9, alpha=0.85, label='Latido Portadora RF')
    
    # Línea de umbral de squelch / pérdida de señal
    ax3.axhline(0.25, color='#FF5252', linestyle=':', lw=1.0, alpha=0.7, label='Umbral Detección Portadora')
    ax3.axhline(-0.25, color='#FF5252', linestyle=':', lw=1.0, alpha=0.7)

    ax3.set_ylabel("Voltaje RF (uV)", fontsize=10, color='#90CAF9')
    ax3.set_xlabel("Tiempo Transcurrido de Misión (Horas MET)", fontsize=11, fontweight='bold', color='#E0E6ED')
    ax3.set_xlim(-0.5, total_hours - 0.5)
    ax3.set_xticks(np.arange(0, total_hours + 1, 6))
    ax3.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax3.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[✓] Gráfico de línea de tiempo guardado en: {save_path}")
    plt.show()


# =====================================================================
# 6. VISUALIZACIÓN 2: DASHBOARD DE CONTROL DE MISIÓN (C2 DASHBOARD)
# =====================================================================

def plot_mission_control_dashboard(metrics: Dict[str, Any], 
                                  last_detected_state: int, 
                                  steady_state_pi: np.ndarray,
                                  risk_evolution: np.ndarray,
                                  save_path: str = None):
    """
    Genera el Dashboard de Control de Misión con:
    1. Semáforo Central de Estado (Verde / Amarillo / Rojo) y Panel de Protocolo Activo.
    2. Matriz de Confusión (Real vs Inferido Viterbi).
    3. Distribución de Estados de Salud (Real vs Inferido).
    4. Medidores KPI de Disponibilidad, Latencia y Pérdida de Enlace.
    5. Curva de Análisis de Riesgo Acumulado a lo largo de la Misión.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 11), facecolor='#080C14')
    gs = GridSpec(2, 3, width_ratios=[1.1, 1.0, 1.2], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    # -------------------------------------------------------------
    # PANEL 1: SEMÁFORO DE CONTROL DE MISIÓN (ESTADO ACTUAL Y PROTOCOLO)
    # -------------------------------------------------------------
    ax_traffic = fig.add_subplot(gs[0, 0])
    ax_traffic.set_facecolor('#101626')
    ax_traffic.set_title("✦ ESTADO DE VUELO ACTUAL (SEMÁFORO) ✦", fontsize=11, fontweight='bold', color='#90CAF9')

    proto = CONTINGENCY_PROTOCOLS[last_detected_state]
    semaforo = proto["semaforo"]

    # Coordenadas de las tres luces del semáforo
    lamp_y = [2.2, 1.2, 0.2]  # Verde, Amarillo, Rojo
    lamp_colors = ['#00E676', '#FFD600', '#FF1744']
    lamp_labels = ['NOMINAL', 'DEGRADACIÓN', 'CRÍTICO']

    # Dibujar cuerpo del semáforo
    rect = mpatches.FancyBboxPatch((0.15, -0.2), 0.7, 2.8, boxstyle="round,pad=0.1",
                                   ec="#37474F", fc="#0A0E17", lw=2.0)
    ax_traffic.add_patch(rect)

    for i in range(3):
        is_active = (i == last_detected_state)
        fill_color = lamp_colors[i] if is_active else '#1E293B'
        alpha_val = 1.0 if is_active else 0.25
        edge_col = lamp_colors[i] if is_active else '#334155'
        
        # Halo de brillo para la luz activa
        if is_active:
            glow = plt.Circle((0.5, lamp_y[i]), 0.42, color=lamp_colors[i], alpha=0.22, zorder=2)
            ax_traffic.add_patch(glow)

        circle = plt.Circle((0.5, lamp_y[i]), 0.30, facecolor=fill_color, edgecolor=edge_col, 
                            lw=2.5, alpha=alpha_val, zorder=3)
        ax_traffic.add_patch(circle)

        # Etiqueta
        font_weight = 'bold' if is_active else 'normal'
        font_col = lamp_colors[i] if is_active else '#64748B'
        ax_traffic.text(0.5, lamp_y[i], lamp_labels[i], ha='center', va='center', 
                        color='#000000' if is_active else font_col,
                        fontsize=9, fontweight=font_weight, zorder=4)

    ax_traffic.set_xlim(-0.1, 1.1)
    ax_traffic.set_ylim(-0.4, 2.9)
    ax_traffic.axis('off')

    # Texto inferior con estado detectado
    ax_traffic.text(0.5, -0.35, f"STATUS: {proto['semaforo']} ({proto['seguridad']})", 
                    ha='center', va='center', color=proto['color'], fontsize=11, fontweight='bold')

    # -------------------------------------------------------------
    # PANEL 2: MATRIZ DE CONFUSIÓN (REAL VS INFERIDO VITERBI)
    # -------------------------------------------------------------
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_cm.set_facecolor('#101626')
    ax_cm.set_title("MATRIZ DE CONFUSIÓN (REAL VS VITERBI)", fontsize=11, fontweight='bold', color='#90CAF9')

    cm = metrics["confusion_matrix"]
    cax = ax_cm.imshow(cm, cmap='Blues', interpolation='nearest', alpha=0.85)

    # Anotar valores en la matriz
    classes = ['Nominal', 'Degrad.', 'Crítica']
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            color_text = '#FFFFFF' if val > cm.max() / 2 else '#90CAF9'
            ax_cm.text(j, i, f"{val}\n({val/metrics['total_hours']*100:.1f}%)",
                       ha="center", va="center", color=color_text, fontsize=10, fontweight='bold')

    ax_cm.set_xticks(range(3))
    ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(classes, fontsize=9)
    ax_cm.set_yticklabels(classes, fontsize=9)
    ax_cm.set_xlabel("Estado Inferido (Viterbi)", fontsize=10, color='#90CAF9')
    ax_cm.set_ylabel("Estado Real (Nave)", fontsize=10, color='#90CAF9')
    
    # Barra de color
    cb = fig.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors='#90CAF9')
    cb.set_label('Número de Horas', color='#90CAF9', fontsize=9)

    # -------------------------------------------------------------
    # PANEL 3: PROTOCOLO DE ACCIONES DE CONTINGENCIA DETALLADO
    # -------------------------------------------------------------
    ax_action = fig.add_subplot(gs[0, 2])
    ax_action.set_facecolor('#101626')
    ax_action.set_title(f"PROTOCOLO DE CONTINGENCIA ACTIVO // {proto['codigo']}", 
                        fontsize=10, fontweight='bold', color=proto['color'])

    action_box = [
        f"• DIRECTIVA PRINCIPAL:",
        f"  {proto['accion_primaria']}",
        "",
        f"• PROCEDIMIENTOS OPERACIONALES ESTÁNDAR (SOP):"
    ]
    for step in proto["procedimientos"]:
        action_box.append(f"  [✓] {step}")

    action_text = "\n".join(action_box)
    ax_action.text(0.04, 0.92, action_text, color='#E2E8F0', fontsize=9.2, verticalalignment='top',
                   fontfamily='monospace', linespacing=1.4,
                   bbox=dict(boxstyle="round,pad=0.7", fc="#0B0F19", ec=proto['color'], lw=1.5))
    ax_action.axis('off')

    # -------------------------------------------------------------
    # PANEL 4: DISTRIBUCIÓN DE ESTADOS (REAL VS VITERBI)
    # -------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.set_facecolor('#101626')
    ax_dist.set_title("DISTRIBUCIÓN DE TIEMPO EN ESTADO", fontsize=11, fontweight='bold', color='#90CAF9')

    bar_x = np.arange(3)
    bar_width = 0.35

    bars_real = ax_dist.bar(bar_x - bar_width/2, metrics["real_pcts"], bar_width, 
                            label='Real', color=['#00C853', '#FFD600', '#D50000'], alpha=0.85)
    bars_inf = ax_dist.bar(bar_x + bar_width/2, metrics["inferred_pcts"], bar_width, 
                           label='Viterbi', color=['#69F0AE', '#FFE57F', '#FF80AB'], alpha=0.85, hatch='//')

    for b in bars_real:
        h = b.get_height()
        ax_dist.text(b.get_x() + b.get_width()/2., h + 1.0, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, color='#E2E8F0')

    for b in bars_inf:
        h = b.get_height()
        ax_dist.text(b.get_x() + b.get_width()/2., h + 1.0, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, color='#FF80AB')

    ax_dist.set_xticks(bar_x)
    ax_dist.set_xticklabels(['Nominal', 'Degradación', 'Crítica'], fontsize=9)
    ax_dist.set_ylabel("% de Misión (Horas)", fontsize=10, color='#90CAF9')
    ax_dist.set_ylim(0, max(max(metrics["real_pcts"]), max(metrics["inferred_pcts"])) + 15)
    ax_dist.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax_dist.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # -------------------------------------------------------------
    # PANEL 5: MÉTRICAS CLAVE DE TELEMETRÍA Y DISPONIBILIDAD (KPIs)
    # -------------------------------------------------------------
    ax_kpi = fig.add_subplot(gs[1, 1])
    ax_kpi.set_facecolor('#101626')
    ax_kpi.set_title("TELEMETRY KEY PERFORMANCE INDICATORS", fontsize=11, fontweight='bold', color='#90CAF9')

    kpi_lines = [
        f"✦ DISPONIBILIDAD TOTAL:      {metrics['availability_pct']:.2f}% ({metrics['stable_hours']}h / {metrics['total_hours']}h)",
        f"✦ SEÑAL INTERMITENTE:        {metrics['intermittent_hours']} h ({metrics['intermittent_hours']/metrics['total_hours']*100:.1f}%)",
        f"✦ SEÑAL PERDIDA (LOS):       {metrics['lost_hours']} h ({metrics['lost_hours']/metrics['total_hours']*100:.1f}%)",
        "--------------------------------------------------",
        f"✦ EPISODIOS PÉRDIDA ENLACE:  {metrics['total_los_events']} eventos",
        f"✦ MÁXIMA DURACIÓN LOS:       {metrics['max_los_hours']} horas continuas",
        f"✦ PROMEDIO DURACIÓN LOS:     {metrics['mean_los_hours']:.2f} horas",
        "--------------------------------------------------",
        f"✦ PRECISIÓN GLOBAL VITERBI:  {metrics['accuracy']:.2f}%",
        f"✦ RETRASO DETECCIÓN ANOMALÍA: {metrics['avg_detection_delay']:.2f} h (Máx: {metrics['max_detection_delay']} h)",
        f"✦ F1-SCORE CRÍTICO:          {metrics['f1'][2]:.3f}"
    ]

    ax_kpi.text(0.05, 0.92, "\n".join(kpi_lines), color='#00E5FF', fontsize=9.2, verticalalignment='top',
                fontfamily='monospace', linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.7", fc="#0B0F19", ec="#00E5FF", lw=1.2))
    ax_kpi.axis('off')

    # -------------------------------------------------------------
    # PANEL 6: ANÁLISIS DE RIESGO DE MISIÓN Y ASÍNTOTA ESTACIONARIA
    # -------------------------------------------------------------
    ax_risk = fig.add_subplot(gs[1, 2])
    ax_risk.set_facecolor('#101626')
    ax_risk.set_title("ANÁLISIS DE RIESGO Y CONFIABILIDAD TEMPORAL", fontsize=11, fontweight='bold', color='#90CAF9')

    hours_risk = np.arange(len(risk_evolution))
    ax_risk.plot(hours_risk, risk_evolution[:, 0], color='#00E676', lw=2.0, label='P(Nominal)')
    ax_risk.plot(hours_risk, risk_evolution[:, 1], color='#FFD600', lw=2.0, label='P(Degradación)')
    ax_risk.plot(hours_risk, risk_evolution[:, 2], color='#FF1744', lw=2.5, label='P(Anomalía Crítica)')

    # Líneas punteadas de estado estacionario asintótico
    ax_risk.axhline(steady_state_pi[2], color='#FF5252', linestyle='--', lw=1.2, 
                    label=f'Riesgo Estacionario Crítico ({steady_state_pi[2]*100:.1f}%)')
    ax_risk.fill_between(hours_risk, 0, risk_evolution[:, 2], color='#FF1744', alpha=0.2)

    ax_risk.set_xlabel("Horas de Misión Proyectadas", fontsize=10, color='#90CAF9')
    ax_risk.set_ylabel("Probabilidad Acumulada", fontsize=10, color='#90CAF9')
    ax_risk.set_ylim(-0.02, 1.02)
    ax_risk.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax_risk.legend(loc='center right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[✓] Dashboard de Control de Misión guardado en: {save_path}")
    plt.show()


# =====================================================================
# 7. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =====================================================================

def run_spacecraft_mission_simulation(n_hours: int = 72, seed: int = 42, display_plots: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el ciclo de vida completo de la simulación de misión espacial de 72 horas:
    1. Instanciación del modelo HMM.
    2. Simulación estocástica de estados y telemetría.
    3. Inferencia de estado oculta mediante Viterbi.
    4. Filtrado posterior en vivo con algoritmo Forward.
    5. Métricas operacionales completas.
    6. Determinación de contingencia actual.
    7. Análisis asintótico y de riesgo de pérdida.
    8. Generación de dashboards visuales.
    """
    print("\n" + "=" * 80)
    print(" INICIANDO SIMULADOR HMM DE TELEMETRÍA ESPACIAL // MISIÓN ORBITAL 72H ")
    print("=" * 80)

    # 1. Crear modelo HMM con las especificaciones del usuario
    hmm = SpacecraftHMM()
    print("[1/6] Modelo HMM inicializado con éxito:")
    print("      • Estados Ocultos:", hmm.STATE_NAMES)
    print("      • Observaciones:   ", hmm.OBS_NAMES)

    # 2. Simular 72 horas de misión
    real_states, observations = hmm.simulate(n_hours=n_hours, seed=seed)
    print(f"[2/6] Simulación completada: {n_hours} horas de mediciones de telemetría.")

    # 3. Decodificar secuencia de estados con Viterbi
    viterbi_states, best_log_prob = hmm.viterbi(observations)
    print(f"[3/6] Algoritmo de Viterbi ejecutado. Log-probabilidad de la ruta óptima: {best_log_prob:.4f}")

    # 4. Inferencia probabilística en tiempo real (Forward filtering)
    alpha_probs = hmm.forward_filtering(observations)
    print("[4/6] Probabilidades filtradas Forward calculadas para cada hora.")

    # 5. Cálculo de métricas
    metrics = calculate_mission_metrics(real_states, observations, viterbi_states)
    steady_state_pi = hmm.compute_steady_state()
    risk_evolution = hmm.risk_profile_projection(n_steps=n_hours)
    last_detected_state = viterbi_states[-1]
    last_proto = CONTINGENCY_PROTOCOLS[last_detected_state]
    print("[5/6] Métricas de telemetría y análisis de confiabilidad procesados.")

    # Imprimir reporte en consola
    print("\n" + "-" * 80)
    print("                      REPORTE EJECUTIVO DE TELEMETRÍA DE MISIÓN")
    print("-" * 80)
    print(f" • Disponibilidad de Señal de Telemetría: {metrics['availability_pct']:.2f}%")
    print(f" • Distribución de Señal:  Estable={metrics['stable_hours']}h | Intermitente={metrics['intermittent_hours']}h | Perdida={metrics['lost_hours']}h")
    print(f" • Eventos de Pérdida (LOS): {metrics['total_los_events']} eventos (Duración máxima: {metrics['max_los_hours']}h, promedio: {metrics['mean_los_hours']:.2f}h)")
    print(f" • Tiempo Real en Estado:   Nominal={metrics['real_counts'][0]}h ({metrics['real_pcts'][0]:.1f}%) | Degradación={metrics['real_counts'][1]}h ({metrics['real_pcts'][1]:.1f}%) | Crítica={metrics['real_counts'][2]}h ({metrics['real_pcts'][2]:.1f}%)")
    print(f" • Tiempo Viterbi Inferido: Nominal={metrics['inferred_counts'][0]}h ({metrics['inferred_pcts'][0]:.1f}%) | Degradación={metrics['inferred_counts'][1]}h ({metrics['inferred_pcts'][1]:.1f}%) | Crítica={metrics['inferred_counts'][2]}h ({metrics['inferred_pcts'][2]:.1f}%)")
    print(f" • Retraso Promedio Detección Anomalías: {metrics['avg_detection_delay']:.2f} horas (Máximo: {metrics['max_detection_delay']}h)")
    print(f" • Precisión Global Viterbi: {metrics['accuracy']:.2f}%")
    print(f" • Distribución Estacionaria Asintótica (Largo Plazo):")
    print(f"     Nominal: {steady_state_pi[0]*100:.2f}% | Degradación: {steady_state_pi[1]*100:.2f}% | Crítica: {steady_state_pi[2]*100:.2f}%")
    print(f" • Estado Detectado al Final de la Misión (T+72h): {last_proto['estado'].upper()} [{last_proto['semaforo']}]")
    print(f" • Directiva Activa: {last_proto['accion_primaria']}")
    print("-" * 80)

    # 6. Visualizaciones
    if display_plots:
        print("[6/6] Renderizando visualizaciones y dashboard de control de misión...")
        plot_mission_timeline(real_states, observations, viterbi_states, alpha_probs)
        plot_mission_control_dashboard(metrics, last_detected_state, steady_state_pi, risk_evolution)

    return {
        "hmm": hmm,
        "real_states": real_states,
        "observations": observations,
        "viterbi_states": viterbi_states,
        "alpha_probs": alpha_probs,
        "metrics": metrics,
        "steady_state_pi": steady_state_pi,
        "risk_evolution": risk_evolution,
        "last_contingency": last_proto
    }

if __name__ == "__main__":
    run_spacecraft_mission_simulation(n_hours=72, seed=42)
