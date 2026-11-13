"""
================================================================================
MODELO OCULTO DE MARKOV (HMM) PARA MONITOREO DE SALUD DE NAVES ESPACIALES
Simulación de Telemetría Terrestre, Algoritmo de Viterbi y Dashboard de Control
================================================================================
Autores: Desarrollado para Ingenieros Aeroespaciales y Aficionados a la Exploración Espacial
Compatible al 100% con Google Colab, Jupyter Notebooks y Python 3.8+
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Any

# =====================================================================
# 1. IMPLEMENTACIÓN DE LA CLASE HMM PARA TELEMETRÍA ESPACIAL
# =====================================================================

class SpacecraftHMM:
    """
    Modelo Oculto de Markov (HMM) para el diagnóstico y monitoreo en tiempo real
    de la salud de subsistemas críticos de una nave espacial en órbita.
    
    ESTADOS OCULTOS (Health Status):
      0: 'Operación Nominal'     (Todos los subsistemas en rango operativo)
      1: 'Degradación de Sistemas' (Anomalías térmicas, caída de potencia o jitter en actitud)
      2: 'Anomalía Crítica'      (Falla mayor, desestabilización o pérdida inminente)
      
    OBSERVACIONES (Ground Telemetry Signals):
      0: 'Señal Estable'       (SNR alto, margen de enlace > 12 dB, sin pérdida de paquetes)
      1: 'Señal Intermitente'  (Pérdida de paquetes 10-40%, fluctuaciones de fase/potencia)
      2: 'Señal Perdida'       (Loss of Signal - LOS, receptor en squelch térmico)
    """

    STATE_NAMES = ['Operación Nominal', 'Degradación de Sistemas', 'Anomalía Crítica']
    STATE_COLORS = ['#00E676', '#FFD600', '#FF1744']  # Verde, Amarillo, Rojo
    OBS_NAMES = ['Señal Estable', 'Señal Intermitente', 'Señal Perdida']

    def __init__(self, 
                 transition_matrix: np.ndarray = None, 
                 emission_matrix: np.ndarray = None, 
                 initial_distribution: np.ndarray = None):
        """
        Inicializa las matrices estocásticas de acuerdo con los parámetros de la misión.
        """
        # Matriz de Transición A (3x3): P(S_t | S_{t-1})
        # Fila i: probabilidades de transicionar desde el estado i a [Nominal, Degradación, Crítica]
        if transition_matrix is not None:
            self.A = np.array(transition_matrix, dtype=np.float64)
        else:
            self.A = np.array([
                [0.90, 0.08, 0.02],  # Nominal
                [0.40, 0.45, 0.15],  # Degradación
                [0.05, 0.15, 0.80]   # Anomalía Crítica
            ], dtype=np.float64)

        # Matriz de Emisión B (3x3): P(O_t | S_t)
        # Fila i: probabilidades de observar [Estable, Intermitente, Perdida] en el estado i
        if emission_matrix is not None:
            self.B = np.array(emission_matrix, dtype=np.float64)
        else:
            self.B = np.array([
                [0.85, 0.12, 0.03],  # Nominal
                [0.30, 0.50, 0.20],  # Degradación
                [0.05, 0.15, 0.80]   # Anomalía Crítica
            ], dtype=np.float64)

        # Distribución Inicial Pi: P(S_0)
        if initial_distribution is not None:
            self.pi = np.array(initial_distribution, dtype=np.float64)
        else:
            self.pi = np.array([0.85, 0.10, 0.05], dtype=np.float64)

        self.n_states = len(self.STATE_NAMES)
        self.n_obs = len(self.OBS_NAMES)

        # Validación formal de propiedades de probabilidad
        assert np.allclose(self.A.sum(axis=1), 1.0), "Cada fila de la matriz A debe sumar 1.0"
        assert np.allclose(self.B.sum(axis=1), 1.0), "Cada fila de la matriz B debe sumar 1.0"
        assert np.isclose(self.pi.sum(), 1.0), "La distribución inicial Pi debe sumar 1.0"

    # -----------------------------------------------------------------
    # SIMULACIÓN ESTOCÁSTICA DE MISIÓN (72 HORAS)
    # -----------------------------------------------------------------
    def simulate(self, n_hours: int = 72, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera una trayectoria sintética de 72 horas para la salud oculta de la nave
        y las señales de telemetría captadas por el segmento terreno.
        """
        rng = np.random.default_rng(seed)
        states = np.zeros(n_hours, dtype=int)
        observations = np.zeros(n_hours, dtype=int)

        # Inicialización t=0
        states[0] = rng.choice(self.n_states, p=self.pi)
        observations[0] = rng.choice(self.n_obs, p=self.B[states[0]])

        # Transición en el tiempo t=1..n_hours-1
        for t in range(1, n_hours):
            states[t] = rng.choice(self.n_states, p=self.A[states[t - 1]])
            observations[t] = rng.choice(self.n_obs, p=self.B[states[t]])

        return states, observations

    # -----------------------------------------------------------------
    # ALGORITMO DE VITERBI (INFERENCIA ÓPTIMA EN ESPACIO LOGARÍTMICO)
    # -----------------------------------------------------------------
    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Decodifica la secuencia oculta más probable de salud de la nave mediante
        programación dinámica en el dominio logarítmico (evita underflow numérico).
        
        Ecuación de Bellman / Recursión de Viterbi:
          V_t(j) = max_i [ V_{t-1}(i) + ln(A_{i,j}) ] + ln(B_{j, y_t})
        """
        T = len(observations)
        K = self.n_states
        eps = 1e-16

        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)

        # Tablas de programación dinámica
        V = np.zeros((T, K), dtype=np.float64)
        backpointer = np.zeros((T, K), dtype=int)

        # Paso 1: Inicialización (t = 0)
        V[0, :] = log_pi + log_B[:, observations[0]]

        # Paso 2: Inducción / Recursión hacia adelante (t = 1 .. T-1)
        for t in range(1, T):
            obs = observations[t]
            for j in range(K):
                candidates = V[t - 1, :] + log_A[:, j]
                best_prev = np.argmax(candidates)
                backpointer[t, j] = best_prev
                V[t, j] = candidates[best_prev] + log_B[j, obs]

        # Paso 3: Terminación
        best_last_state = np.argmax(V[T - 1, :])
        best_log_prob = V[T - 1, best_last_state]

        # Paso 4: Reconstrucción hacia atrás (Backtracking)
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = best_last_state
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path, float(best_log_prob)

    # -----------------------------------------------------------------
    # FILTRADO FORWARD (CREENCIA PROBABILÍSTICA EN TIEMPO REAL)
    # -----------------------------------------------------------------
    def forward_filtering(self, observations: np.ndarray) -> np.ndarray:
        """
        Calcula la probabilidad a posteriori en tiempo real P(S_t = k | O_{1:t})
        utilizando el algoritmo Forward con normalización en cada paso horario.
        """
        T = len(observations)
        K = self.n_states
        alpha = np.zeros((T, K), dtype=np.float64)

        # t = 0
        raw_alpha = self.pi * self.B[:, observations[0]]
        alpha[0, :] = raw_alpha / np.sum(raw_alpha)

        # t = 1 .. T-1
        for t in range(1, T):
            prior = alpha[t - 1, :] @ self.A
            raw_alpha = prior * self.B[:, observations[t]]
            scale = np.sum(raw_alpha)
            alpha[t, :] = raw_alpha / (scale if scale > 0 else 1.0)

        return alpha

    # -----------------------------------------------------------------
    # ANÁLISIS DE ESTADO ESTACIONARIO Y RIESGO DE MISIÓN
    # -----------------------------------------------------------------
    def compute_steady_state(self) -> np.ndarray:
        """
        Determina la distribución de equilibrio a largo plazo (pi_inf * A = pi_inf).
        """
        K = self.n_states
        M = self.A.T - np.eye(K)
        M[-1, :] = 1.0
        rhs = np.zeros(K)
        rhs[-1] = 1.0
        try:
            return np.clip(np.linalg.solve(M, rhs), 0, 1)
        except np.linalg.LinAlgError:
            Ak = np.linalg.matrix_power(self.A, 500)
            return Ak[0, :]

    def risk_profile_projection(self, n_steps: int = 72) -> np.ndarray:
        """
        Proyecta la evolución de la probabilidad de estado a lo largo del tiempo:
        P(S_t) = Pi * A^t
        """
        evolution = np.zeros((n_steps, self.n_states))
        current = self.pi.copy()
        for t in range(n_steps):
            evolution[t, :] = current
            current = current @ self.A
        return evolution


# =====================================================================
# 2. ACCIONES DE CONTINGENCIA SEGÚN ESTADO DE DETECCIÓN
# =====================================================================

CONTINGENCY_ACTIONS = {
    0: {
        "estado": "Operación Nominal",
        "semaforo": "VERDE",
        "codigo": "STATUS-GREEN // NOMINAL FLIGHT",
        "color": "#00E676",
        "directiva": "CONTINUAR MISIÓN NOMINAL",
        "procedimientos": [
            "Mantener cronograma de ciencia y telemetría de alta velocidad (Banda X).",
            "Cargas útiles científicas al 100% de potencia nominal.",
            "Monitoreo estándar de ciclo térmico y orientación de paneles solares.",
            "Descarga de buffers de memoria de a bordo programada en el próximo pase."
        ]
    },
    1: {
        "estado": "Degradación de Sistemas",
        "semaforo": "AMARILLO",
        "codigo": "STATUS-AMBER // SYSTEM DEGRADATION",
        "color": "#FFD600",
        "directiva": "MONITOREO INTENSIVO Y DIAGNÓSTICO",
        "procedimientos": [
            "Activar telemetría de ingeniería de alta resolución (sub-rate 1 Hz).",
            "Suspender maniobras orbitales y experimentos científicos no críticos.",
            "Aislar bus de carga útil secundaria; verificar márgenes de potencia.",
            "Ejecutar diagnóstico térmico de tubos amplificadores de potencia (TWTA).",
            "Preparar comandos de conmutación a cadena de aviónica redundante (Canal B)."
        ]
    },
    2: {
        "estado": "Anomalía Crítica",
        "semaforo": "ROJO",
        "codigo": "STATUS-RED // CRITICAL FAULT - SAFE HOLD",
        "color": "#FF1744",
        "directiva": "ACTIVAR MODO SEGURO (SAFE-HOLD MODE)",
        "procedimientos": [
            "ORDEN INMEDIATA: Disparar rutina de aborto y modo de supervivencia.",
            "Apagado total de todas las cargas útiles e instrumentos científicos.",
            "Orientar eje primario de paneles solares directamente hacia el vector Sol.",
            "Conmutar transmisor a Antena de Baja Ganancia (LGA) omnidireccional en Banda S.",
            "Amortiguamiento de giros de actitud con ruedas de reacción y propulsores RCS.",
            "Notificar a la Red del Espacio Profundo (DSN) para enlace de emergencia continuo."
        ]
    }
}


# =====================================================================
# 3. CÁLCULO DE MÉTRICAS DE TELEMETRÍA Y RENDIMIENTO
# =====================================================================

def compute_mission_kpis(real_states: np.ndarray, 
                         observations: np.ndarray, 
                         viterbi_states: np.ndarray) -> Dict[str, Any]:
    """
    Calcula los indicadores clave de desempeño (KPIs) de la misión:
    - Disponibilidad de señal
    - Tiempo en cada estado de salud
    - Latencia en detección de fallas
    - Períodos de pérdida de señal (Loss of Signal)
    - Matriz de confusión y métricas de clasificación
    """
    T = len(real_states)

    # 1. Disponibilidad de Señal (% horas de señal estable)
    stable_h = int(np.sum(observations == 0))
    intermittent_h = int(np.sum(observations == 1))
    lost_h = int(np.sum(observations == 2))
    availability = (stable_h / T) * 100.0

    # 2. Tiempos en estados
    real_counts = [int(np.sum(real_states == s)) for s in range(3)]
    viterbi_counts = [int(np.sum(viterbi_states == s)) for s in range(3)]

    # 3. Análisis de Períodos de Pérdida de Señal (LOS)
    los_runs = []
    current_run = 0
    for obs in observations:
        if obs == 2:
            current_run += 1
        else:
            if current_run > 0:
                los_runs.append(current_run)
                current_run = 0
    if current_run > 0:
        los_runs.append(current_run)

    max_los = max(los_runs) if los_runs else 0
    mean_los = float(np.mean(los_runs)) if los_runs else 0.0

    # 4. Retraso en la Detección de Anomalías (Detection Delay)
    delays = []
    in_fault = False
    start_t = 0
    for t in range(T):
        fault_active = (real_states[t] in [1, 2])
        if fault_active and not in_fault:
            in_fault = True
            start_t = t
            # Buscar instante de detección
            detected = None
            for scan in range(start_t, T):
                if viterbi_states[scan] in [1, 2]:
                    detected = scan
                    break
            delays.append(detected - start_t if detected is not None else (T - start_t))
        elif not fault_active and in_fault:
            in_fault = False

    avg_delay = float(np.mean(delays)) if delays else 0.0
    max_delay = int(np.max(delays)) if delays else 0

    # 5. Matriz de Confusión
    conf = np.zeros((3, 3), dtype=int)
    for r, v in zip(real_states, viterbi_states):
        conf[r, v] += 1

    accuracy = float(np.trace(conf) / T) * 100.0

    # Recall y F1 para anomalía crítica
    tp_crit = conf[2, 2]
    fp_crit = np.sum(conf[:, 2]) - tp_crit
    fn_crit = np.sum(conf[2, :]) - tp_crit
    prec_crit = tp_crit / (tp_crit + fp_crit) if (tp_crit + fp_crit) > 0 else 0.0
    rec_crit = tp_crit / (tp_crit + fn_crit) if (tp_crit + fn_crit) > 0 else 0.0
    f1_crit = (2 * prec_crit * rec_crit / (prec_crit + rec_crit)) if (prec_crit + rec_crit) > 0 else 0.0

    return {
        "total_hours": T,
        "availability_pct": availability,
        "stable_hours": stable_h,
        "intermittent_hours": intermittent_h,
        "lost_hours": lost_h,
        "real_counts": real_counts,
        "viterbi_counts": viterbi_counts,
        "los_runs": los_runs,
        "max_los_hours": max_los,
        "mean_los_hours": mean_los,
        "detection_delays": delays,
        "avg_detection_delay": avg_delay,
        "max_detection_delay": max_delay,
        "confusion_matrix": conf,
        "accuracy": accuracy,
        "f1_critical": f1_crit,
        "recall_critical": rec_crit
    }


# =====================================================================
# 4. FORMA DE ONDA DE LATIDOS DE TELEMETRÍA (HEARTBEAT)
# =====================================================================

def generate_heartbeat_signal(observations: np.ndarray, samples_per_hour: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera la señal de RF continua del 'Heartbeat' de la nave espacial,
    modulando pulsos periódicos y ruido térmico según la calidad de recepción.
    """
    T = len(observations)
    n_samples = T * samples_per_hour
    t_axis = np.linspace(0, T, n_samples, endpoint=False)
    signal = np.zeros(n_samples)
    rng = np.random.default_rng(777)

    for h in range(T):
        idx0 = h * samples_per_hour
        idx1 = (h + 1) * samples_per_hour
        tau = t_axis[idx0:idx1] - h  # 0 a 1 dentro de la hora

        # Pulso biométrico / telemetría de baliza
        pulse = (np.exp(-((tau - 0.25) ** 2) / 0.008) * 1.4 - 
                 np.exp(-((tau - 0.35) ** 2) / 0.004) * 0.7 +
                 0.2 * np.sin(2 * np.pi * 3 * tau))

        obs = observations[h]
        if obs == 0:  # Señal Estable
            carrier = 0.25 * np.cos(2 * np.pi * 12 * tau)
            signal[idx0:idx1] = pulse + carrier + rng.normal(0, 0.04, samples_per_hour)
        elif obs == 1:  # Señal Intermitente
            drop = rng.choice([0.0, 1.0], size=samples_per_hour, p=[0.3, 0.7])
            amp = rng.uniform(0.3, 0.6, samples_per_hour)
            signal[idx0:idx1] = (pulse * drop * amp) + rng.normal(0, 0.12, samples_per_hour)
        else:  # Señal Perdida (solo ruido térmico)
            signal[idx0:idx1] = rng.normal(0, 0.06, samples_per_hour)

    return t_axis, signal


# =====================================================================
# 5. VISUALIZACIÓN A: LÍNEA DE TIEMPO, HEARTBEAT Y ALERTAS
# =====================================================================

def plot_timeline_and_heartbeat(real_states: np.ndarray, 
                                observations: np.ndarray, 
                                viterbi_states: np.ndarray, 
                                alpha_probs: np.ndarray):
    """
    Crea la figura de alta resolución de la telemetría temporal de la nave:
    - Señal observada con fondo coloreado según estado real y banderolas de alerta.
    - Decodificación Viterbi vs Real.
    - Probabilidad en tiempo real (Forward).
    - Señal de latidos (Heartbeat RF).
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 13), facecolor='#0B0F19')
    gs = GridSpec(4, 1, height_ratios=[2.2, 1.3, 1.4, 1.3], hspace=0.35)

    hours = np.arange(len(real_states))
    colors = SpacecraftHMM.STATE_COLORS

    # 1. Observación y Fondo Real
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor('#101626')
    ax0.set_title("✦ TELEMETRÍA DE TIERRA, ESTADOS DE SALUD REAL Y ALERTAS CRÍTICAS (72 HORAS MET) ✦", 
                  fontsize=12, fontweight='bold', color='#E0E6ED', pad=12)

    # Franjas de fondo según estado de salud real
    for t in range(len(real_states)):
        ax0.axvspan(t - 0.5, t + 0.5, color=colors[real_states[t]], alpha=0.18, zorder=0)

    # Mapeo de observación: 2=Estable, 1=Intermitente, 0=Perdida
    obs_y = np.array([{0: 2, 1: 1, 2: 0}[o] for o in observations])
    ax0.step(hours, obs_y, where='mid', color='#00E5FF', lw=2.0, label='Calidad de Señal Recibida (Tierra)', zorder=2)

    # Marcadores
    ax0.scatter(hours[observations == 0], obs_y[observations == 0], marker='o', color='#00E676', s=60, label='Señal Estable', zorder=3)
    ax0.scatter(hours[observations == 1], obs_y[observations == 1], marker='^', color='#FFD600', s=70, label='Señal Intermitente', zorder=3)
    ax0.scatter(hours[observations == 2], obs_y[observations == 2], marker='X', color='#FF1744', s=80, label='Señal Perdida (LOS)', zorder=3)

    # Marcadores de Alerta cuando ocurre anomalía crítica
    for t in range(len(real_states)):
        if real_states[t] == 2:
            ax0.scatter(t, 2.3, marker='v', color='#FF1744', s=120, zorder=5)
            if t == 0 or real_states[t - 1] != 2:
                ax0.annotate(f"¡ANOMALÍA!\nT+{t:02d}h", xy=(t, 2.3), xytext=(t, 2.7),
                             arrowprops=dict(facecolor='#FF1744', edgecolor='#FFFFFF', arrowstyle="->", lw=1.2),
                             fontsize=8, fontweight='bold', color='#FF5252', ha='center',
                             bbox=dict(boxstyle="round,pad=0.2", fc="#2A0B14", ec="#FF1744", lw=1.2))

    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels(['Perdida', 'Intermitente', 'Estable'], fontsize=10, fontweight='bold')
    ax0.set_ylabel("Enlace de Telemetría", fontsize=10, color='#90CAF9')
    ax0.set_ylim(-0.5, 3.1)
    ax0.grid(True, linestyle=':', alpha=0.3, color='#455A64')

    patch_n = mpatches.Patch(color=colors[0], alpha=0.4, label='Salud Real: Nominal')
    patch_d = mpatches.Patch(color=colors[1], alpha=0.4, label='Salud Real: Degradación')
    patch_c = mpatches.Patch(color=colors[2], alpha=0.4, label='Salud Real: Crítica')
    h, l = ax0.get_legend_handles_labels()
    ax0.legend(handles=h + [patch_n, patch_d, patch_c], loc='upper right', ncol=3, fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # 2. Decodificación Real vs Viterbi
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.set_facecolor('#101626')
    ax1.set_title("DIAGNÓSTICO INFERIDO: ESTADO REAL (NAVE) VS ALGORITMO DE VITERBI (TIERRA)", 
                  fontsize=10, fontweight='bold', color='#90CAF9', loc='left')
    ax1.plot(hours, real_states, color='#76FF03', lw=2.4, label='Salud Real (Ground Truth)')
    ax1.step(hours, viterbi_states, where='mid', color='#FF4081', lw=2.0, linestyle='--', label='Inferencia Viterbi')
    
    # Resaltar discordancias
    disc = (real_states != viterbi_states)
    for t in range(len(real_states)):
        if disc[t]:
            ax1.axvspan(t - 0.5, t + 0.5, color='#FF1744', alpha=0.25, hatch='//')

    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Nominal', 'Degradación', 'Crítica'], fontsize=9)
    ax1.set_ylabel("Estado de Salud", fontsize=10, color='#90CAF9')
    ax1.set_ylim(-0.4, 2.4)
    ax1.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax1.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # 3. Probabilidades en Tiempo Real (Forward)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.set_facecolor('#101626')
    ax2.set_title("PROBABILIDAD POSTERIOR EN TIEMPO REAL: P(S_t | O_1:t) - ESTIMACIÓN DE CONTROL DE VUELO", 
                  fontsize=10, fontweight='bold', color='#90CAF9', loc='left')
    ax2.plot(hours, alpha_probs[:, 0], color=colors[0], lw=1.8, label='P(Nominal)')
    ax2.plot(hours, alpha_probs[:, 1], color=colors[1], lw=1.8, label='P(Degradación)')
    ax2.plot(hours, alpha_probs[:, 2], color=colors[2], lw=2.2, label='P(Anomalía Crítica)')
    ax2.fill_between(hours, 0, alpha_probs[:, 2], color=colors[2], alpha=0.22)
    ax2.set_ylabel("Probabilidad", fontsize=10, color='#90CAF9')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # 4. Latidos de Telemetría (Heartbeat RF Waveform)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.set_facecolor('#0A0E1A')
    ax3.set_title("FORMA DE ONDA DE LATIDOS (HEARTBEAT CARRIER WAVEFORM - BALIZA DE RF)", 
                  fontsize=10, fontweight='bold', color='#00E5FF', loc='left')
    t_wave, waveform = generate_heartbeat_signal(observations)
    ax3.plot(t_wave, waveform, color='#00E5FF', lw=0.9, alpha=0.9, label='Portadora RF de Baliza')
    ax3.set_ylabel("Amplitud RF (uV)", fontsize=10, color='#90CAF9')
    ax3.set_xlabel("Tiempo Transcurrido de Misión (Horas MET)", fontsize=11, fontweight='bold', color='#E0E6ED')
    ax3.set_xlim(-0.5, len(real_states) - 0.5)
    ax3.set_xticks(np.arange(0, len(real_states) + 1, 6))
    ax3.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax3.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    plt.tight_layout()
    plt.show()


# =====================================================================
# 6. VISUALIZACIÓN B: DASHBOARD DE CONTROL DE MISIÓN (SEMÁFORO Y KPIs)
# =====================================================================

def plot_mission_dashboard(metrics: Dict[str, Any], 
                           active_state: int, 
                           steady_state: np.ndarray, 
                           risk_curve: np.ndarray):
    """
    Construye el Centro de Control de Operaciones Espaciales:
    - Semáforo iluminado según estado actual (Verde / Amarillo / Rojo).
    - Protocolo de contingencia y directivas de vuelo activas.
    - Matriz de confusión interactiva.
    - Desglose de tiempo de salud vs inferencia.
    - Panel de telemetría y fiabilidad a largo plazo.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 11), facecolor='#080C14')
    gs = GridSpec(2, 3, width_ratios=[1.1, 1.0, 1.2], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    proto = CONTINGENCY_ACTIONS[active_state]

    # 1. Semáforo Central de Control
    ax_light = fig.add_subplot(gs[0, 0])
    ax_light.set_facecolor('#101626')
    ax_light.set_title("✦ SEMÁFORO DE SALUD DE VUELO (ESTADO ACTUAL) ✦", fontsize=11, fontweight='bold', color='#90CAF9')

    # Cuerpo del semáforo
    cuerpo = mpatches.FancyBboxPatch((0.15, -0.2), 0.7, 2.8, boxstyle="round,pad=0.1", ec="#37474F", fc="#0A0E17", lw=2.0)
    ax_light.add_patch(cuerpo)

    lamp_y = [2.2, 1.2, 0.2]  # Verde, Amarillo, Rojo
    lamp_colors = ['#00E676', '#FFD600', '#FF1744']
    lamp_names = ['NOMINAL', 'DEGRADACIÓN', 'CRÍTICO']

    for i in range(3):
        active = (i == active_state)
        fc = lamp_colors[i] if active else '#1E293B'
        ec = lamp_colors[i] if active else '#334155'
        
        if active:
            glow = plt.Circle((0.5, lamp_y[i]), 0.42, color=lamp_colors[i], alpha=0.25, zorder=2)
            ax_light.add_patch(glow)

        circ = plt.Circle((0.5, lamp_y[i]), 0.30, facecolor=fc, edgecolor=ec, lw=2.5, alpha=1.0 if active else 0.3, zorder=3)
        ax_light.add_patch(circ)
        ax_light.text(0.5, lamp_y[i], lamp_names[i], ha='center', va='center', 
                      color='#000000' if active else '#64748B', fontsize=9, fontweight='bold' if active else 'normal', zorder=4)

    ax_light.set_xlim(-0.1, 1.1)
    ax_light.set_ylim(-0.4, 2.9)
    ax_light.axis('off')
    ax_light.text(0.5, -0.35, f"STATUS ACTUAL: {proto['semaforo']}", ha='center', va='center', color=proto['color'], fontsize=11, fontweight='bold')

    # 2. Matriz de Confusión
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_cm.set_facecolor('#101626')
    ax_cm.set_title("MATRIZ DE CONFUSIÓN (REAL VS VITERBI)", fontsize=11, fontweight='bold', color='#90CAF9')
    cm = metrics["confusion_matrix"]
    cax = ax_cm.imshow(cm, cmap='Blues', alpha=0.85)
    
    classes = ['Nominal', 'Degrad.', 'Crítica']
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            col = '#FFFFFF' if val > cm.max() / 2 else '#90CAF9'
            ax_cm.text(j, i, f"{val}\n({val/metrics['total_hours']*100:.1f}%)", ha="center", va="center", color=col, fontsize=10, fontweight='bold')

    ax_cm.set_xticks(range(3))
    ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(classes, fontsize=9)
    ax_cm.set_yticklabels(classes, fontsize=9)
    ax_cm.set_xlabel("Inferencia (Viterbi)", fontsize=10, color='#90CAF9')
    ax_cm.set_ylabel("Estado Real (Nave)", fontsize=10, color='#90CAF9')
    cb = fig.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)
    cb.set_label('Horas', color='#90CAF9', fontsize=9)

    # 3. Protocolo de Contingencia
    ax_proto = fig.add_subplot(gs[0, 2])
    ax_proto.set_facecolor('#101626')
    ax_proto.set_title(f"DIRECTIVAS Y PLAN DE CONTINGENCIA // {proto['codigo']}", fontsize=10, fontweight='bold', color=proto['color'])

    lines = [
        f"• ACCIÓN INMEDIATA:",
        f"  {proto['directiva']}",
        "",
        f"• PROCEDIMIENTOS ESTÁNDAR (SOP):"
    ]
    for step in proto["procedimientos"]:
        lines.append(f"  [>] {step}")

    ax_proto.text(0.04, 0.92, "\n".join(lines), color='#E2E8F0', fontsize=9.2, verticalalignment='top',
                  fontfamily='monospace', linespacing=1.35,
                  bbox=dict(boxstyle="round,pad=0.7", fc="#0B0F19", ec=proto['color'], lw=1.5))
    ax_proto.axis('off')

    # 4. Distribución de Estados
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.set_facecolor('#101626')
    ax_dist.set_title("DISTRIBUCIÓN TEMPORAL DE SALUD", fontsize=11, fontweight='bold', color='#90CAF9')
    x_bar = np.arange(3)
    w = 0.35
    r_pct = [c / metrics['total_hours'] * 100 for c in metrics['real_counts']]
    v_pct = [c / metrics['total_hours'] * 100 for c in metrics['viterbi_counts']]

    b1 = ax_dist.bar(x_bar - w/2, r_pct, w, label='Real', color=['#00C853', '#FFD600', '#D50000'], alpha=0.85)
    b2 = ax_dist.bar(x_bar + w/2, v_pct, w, label='Viterbi', color=['#69F0AE', '#FFE57F', '#FF80AB'], alpha=0.85, hatch='//')

    for b in b1:
        ax_dist.text(b.get_x() + b.get_width()/2., b.get_height() + 1, f"{b.get_height():.1f}%", ha='center', va='bottom', fontsize=8, color='#E2E8F0')
    for b in b2:
        ax_dist.text(b.get_x() + b.get_width()/2., b.get_height() + 1, f"{b.get_height():.1f}%", ha='center', va='bottom', fontsize=8, color='#FF80AB')

    ax_dist.set_xticks(x_bar)
    ax_dist.set_xticklabels(classes, fontsize=9)
    ax_dist.set_ylabel("% de Misión", fontsize=10, color='#90CAF9')
    ax_dist.set_ylim(0, max(max(r_pct), max(v_pct)) + 15)
    ax_dist.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax_dist.legend(loc='upper right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    # 5. Métricas Clave (KPIs)
    ax_kpi = fig.add_subplot(gs[1, 1])
    ax_kpi.set_facecolor('#101626')
    ax_kpi.set_title("MÉTRICAS CLAVE DE TELEMETRÍA (KPIS)", fontsize=11, fontweight='bold', color='#90CAF9')

    kpi_text = [
        f"✦ DISPONIBILIDAD TOTAL:      {metrics['availability_pct']:.2f}% ({metrics['stable_hours']}/{metrics['total_hours']}h)",
        f"✦ SEÑAL INTERMITENTE:        {metrics['intermittent_hours']} h ({metrics['intermittent_hours']/metrics['total_hours']*100:.1f}%)",
        f"✦ SEÑAL PERDIDA (LOS):       {metrics['lost_hours']} h ({metrics['lost_hours']/metrics['total_hours']*100:.1f}%)",
        "--------------------------------------------------",
        f"✦ EPISODIOS PÉRDIDA ENLACE:  {len(metrics['los_runs'])} eventos",
        f"✦ DURACIÓN MÁXIMA DE LOS:    {metrics['max_los_hours']} horas seguidas",
        f"✦ DURACIÓN MEDIA DE LOS:     {metrics['mean_los_hours']:.2f} horas",
        "--------------------------------------------------",
        f"✦ PRECISIÓN GLOBAL VITERBI:  {metrics['accuracy']:.2f}%",
        f"✦ RETRASO DETECCIÓN ANOMALÍA: {metrics['avg_detection_delay']:.2f} h (Máx: {metrics['max_detection_delay']}h)",
        f"✦ F1-SCORE ANOMALÍA CRÍTICA: {metrics['f1_critical']:.3f}"
    ]

    ax_kpi.text(0.04, 0.92, "\n".join(kpi_text), color='#00E5FF', fontsize=9.2, verticalalignment='top',
                fontfamily='monospace', linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.7", fc="#0B0F19", ec="#00E5FF", lw=1.2))
    ax_kpi.axis('off')

    # 6. Confiabilidad y Análisis de Riesgo
    ax_risk = fig.add_subplot(gs[1, 2])
    ax_risk.set_facecolor('#101626')
    ax_risk.set_title("ANÁLISIS DE RIESGO DE MISIÓN Y ASÍNTOTA", fontsize=11, fontweight='bold', color='#90CAF9')
    h_risk = np.arange(len(risk_curve))
    ax_risk.plot(h_risk, risk_curve[:, 0], color='#00E676', lw=2.0, label='P(Nominal)')
    ax_risk.plot(h_risk, risk_curve[:, 1], color='#FFD600', lw=2.0, label='P(Degradación)')
    ax_risk.plot(h_risk, risk_curve[:, 2], color='#FF1744', lw=2.4, label='P(Crítica)')
    ax_risk.axhline(steady_state[2], color='#FF5252', linestyle='--', lw=1.2, label=f'Riesgo Crítico Asintótico ({steady_state[2]*100:.1f}%)')
    ax_risk.fill_between(h_risk, 0, risk_curve[:, 2], color='#FF1744', alpha=0.2)
    ax_risk.set_xlabel("Horas de Misión", fontsize=10, color='#90CAF9')
    ax_risk.set_ylabel("Probabilidad", fontsize=10, color='#90CAF9')
    ax_risk.set_ylim(-0.02, 1.02)
    ax_risk.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax_risk.legend(loc='center right', fontsize=8, facecolor='#161F33', edgecolor='#37474F')

    plt.tight_layout()
    plt.show()


# =====================================================================
# 7. EJECUCIÓN PRINCIPAL
# =====================================================================

def main():
    print("=" * 80)
    print(" SISTEMA DE CONTROL DE TELEMETRÍA ESPACIAL MEDIANTE HMM (72 HORAS MET) ")
    print("=" * 80)

    # Inicializar HMM con especificaciones de vuelo
    hmm = SpacecraftHMM()

    # Simular 72 horas de telemetría orbital
    real_states, observations = hmm.simulate(n_hours=72, seed=42)

    # Inferencia con Algoritmo de Viterbi
    viterbi_states, log_prob = hmm.viterbi(observations)

    # Inferencia en tiempo real (Forward)
    alpha_probs = hmm.forward_filtering(observations)

    # Métricas y KPIs
    metrics = compute_mission_kpis(real_states, observations, viterbi_states)
    steady_state = hmm.compute_steady_state()
    risk_curve = hmm.risk_profile_projection(72)

    # Estado actual detectado
    current_state = viterbi_states[-1]
    active_proto = CONTINGENCY_ACTIONS[current_state]

    # Resumen en consola
    print(f"\n[+] DISPONIBILIDAD TOTAL DE TELEMETRÍA: {metrics['availability_pct']:.2f}%")
    print(f"[+] PRECISIÓN GLOBAL DE VITERBI:        {metrics['accuracy']:.2f}%")
    print(f"[+] RETRASO PROMEDIO DETECCIÓN FALLAS:  {metrics['avg_detection_delay']:.2f} horas")
    print(f"[+] DURACIÓN MÁXIMA PÉRDIDA SEÑAL (LOS):{metrics['max_los_hours']} horas")
    print(f"[+] ESTADO DETECTADO AL FINAL (T+72h):  {active_proto['estado'].upper()} [{active_proto['semaforo']}]")
    print(f"[+] ACCIÓN DE CONTINGENCIA INMEDIATA:   {active_proto['directiva']}")
    print(f"[+] PROBABILIDAD ESTACIONARIA CRÍTICA:  {steady_state[2]*100:.2f}%\n")

    # Renderizar gráficos
    plot_timeline_and_heartbeat(real_states, observations, viterbi_states, alpha_probs)
    plot_mission_dashboard(metrics, current_state, steady_state, risk_curve)


if __name__ == "__main__":
    main()
