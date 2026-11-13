"""
================================================================================
SISTEMA DE MONITOREO DE SALUD DE NAVE ESPACIAL MEDIANTE MODELOS OCULTOS DE MARKOV (HMM)
================================================================================
Misión: Simulación y Decodificación de Telemetría Orbital en Tiempo Real
Autor: Antigravity Aerospace Division
Ejecutable en: Google Colab, Jupyter Notebook o entorno local de Python.
Dependencias: numpy, matplotlib, pandas, seaborn
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Tuple, Dict

# Configuración visual para Google Colab y entornos locales
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'


# ==============================================================================
# 1. IMPLEMENTACIÓN DE LA CLASE HMM PARA TELEMETRÍA ESPACIAL
# ==============================================================================

class SpacecraftTelemetryHMM:
    """
    Modelo Oculto de Markov (HMM) discreto para el diagnóstico del estado
    de subsistemas de una nave espacial a partir de la calidad de señal de enlace descendente.
    """

    def __init__(
        self,
        hidden_states: List[str],
        observations: List[str],
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_distribution: np.ndarray
    ):
        self.states = hidden_states
        self.observations = observations
        self.A = np.array(transition_matrix, dtype=np.float64)
        self.B = np.array(emission_matrix, dtype=np.float64)
        self.pi = np.array(initial_distribution, dtype=np.float64)

        # Mapeos rápidos índice <-> nombre
        self.state_to_idx = {name: i for i, name in enumerate(self.states)}
        self.obs_to_idx = {name: i for i, name in enumerate(self.observations)}
        self.idx_to_state = {i: name for i, name in enumerate(self.states)}
        self.idx_to_obs = {i: name for i, name in enumerate(self.observations)}

        self._validate_matrices()

    def _validate_matrices(self):
        """Valida que las matrices sean estocásticas (sumen 1 por fila)."""
        assert np.allclose(self.A.sum(axis=1), 1.0), "Filas de matriz A deben sumar 1"
        assert np.allclose(self.B.sum(axis=1), 1.0), "Filas de matriz B deben sumar 1"
        assert np.isclose(self.pi.sum(), 1.0), "Vector pi debe sumar 1"

    def simulate(self, T: int, seed: int = None) -> Tuple[List[int], List[int]]:
        """
        Simula T horas de misión generando los estados de salud reales (ocultos)
        y las señales recibidas en las estaciones terrestres (observadas).
        """
        if seed is not None:
            np.random.seed(seed)

        states_seq = []
        obs_seq = []

        # Estado y observación en t = 0
        current_state = np.random.choice(len(self.states), p=self.pi)
        current_obs = np.random.choice(len(self.observations), p=self.B[current_state])
        states_seq.append(current_state)
        obs_seq.append(current_obs)

        # Generación de la serie de tiempo para t = 1 ... T-1
        for _ in range(1, T):
            current_state = np.random.choice(len(self.states), p=self.A[current_state])
            current_obs = np.random.choice(len(self.observations), p=self.B[current_state])
            states_seq.append(current_state)
            obs_seq.append(current_obs)

        return states_seq, obs_seq

    def viterbi(self, obs_seq: List[int]) -> List[int]:
        """
        Algoritmo de Viterbi implementado en espacio logarítmico para evitar
        underflow numérico en secuencias largas.
        Encuentra la secuencia de estados ocultos más probable: argmax P(Q | O).
        """
        T = len(obs_seq)
        N = len(self.states)
        eps = 1e-15  # Prevenir log(0)

        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)

        viterbi_table = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)

        # Inicialización en t = 0
        viterbi_table[0] = log_pi + log_B[:, obs_seq[0]]

        # Recursión forward
        for t in range(1, T):
            o_t = obs_seq[t]
            for j in range(N):
                trans_probs = viterbi_table[t - 1] + log_A[:, j]
                best_prev = np.argmax(trans_probs)
                viterbi_table[t, j] = trans_probs[best_prev] + log_B[j, o_t]
                backpointer[t, j] = best_prev

        # Terminación y reconstrucción backtrack del camino óptimo
        best_path = [int(np.argmax(viterbi_table[T - 1]))]
        for t in range(T - 1, 0, -1):
            best_prev = backpointer[t, best_path[-1]]
            best_path.append(best_prev)

        best_path.reverse()
        return best_path

    def compute_steady_state(self) -> np.ndarray:
        """
        Calcula la distribución estacionaria a largo plazo (pi * A = pi).
        Resuelve el sistema lineal (A.T - I) * pi = 0 con sum(pi) = 1.
        """
        N = len(self.states)
        Q = self.A.T - np.eye(N)
        Q[-1] = np.ones(N)
        rhs = np.zeros(N)
        rhs[-1] = 1.0
        return np.linalg.solve(Q, rhs)

    def compute_mttf_critical(self) -> float:
        """
        Análisis de Confiabilidad: Tiempo medio hasta la primera Anomalía Crítica (MTTF).
        Modela el sistema como una cadena absorbente con el estado Crítico como sumidero.
        """
        # Estados transitorios: Nominal (0), Degradación (1)
        Q = self.A[:2, :2]
        I = np.eye(2)
        # Matriz fundamental N = (I - Q)^(-1)
        N_fund = np.linalg.inv(I - Q)
        # Vector de tiempos esperados de absorción iniciando en Nominal
        time_to_absorb = N_fund.sum(axis=1)
        return float(time_to_absorb[0])


# ==============================================================================
# 2. DEFINICIÓN DE CONTINGENCIAS Y ANÁLISIS DE TELEMETRÍA
# ==============================================================================

CONTINGENCY_ACTIONS = {
    'Operación Nominal': {
        'protocolo': 'CRUCERO_ESTÁNDAR',
        'accion': 'Continuar cronograma de misión nominal. Telemetría estándar a 1 Hz.',
        'color': '#2ecc71',  # Verde
        'nivel_alerta': 'VERDE'
    },
    'Degradación de Sistemas': {
        'protocolo': 'MONITOREO_INTENSIVO',
        'accion': 'Incrementar tasa de enlace a 10 Hz. Activar diagnósticos térmicos y bus de potencia.',
        'color': '#f1c40f',  # Amarillo
        'nivel_alerta': 'AMARILLO'
    },
    'Anomalía Crítica': {
        'protocolo': 'MODO_SEGURO_SAFE_MODE',
        'accion': 'ABORTAR operaciones de carga útil. Orientación solar de emergencia y enlace omnidireccional Banda S.',
        'color': '#e74c3c',  # Rojo
        'nivel_alerta': 'ROJO'
    }
}


def analyze_mission_metrics(
    real_states: List[int],
    obs_signals: List[int],
    inferred_states: List[int],
    model: SpacecraftTelemetryHMM
) -> Dict:
    """Calcula todas las métricas operacionales requeridas por el control de vuelo."""
    T = len(real_states)

    # 1. Disponibilidad total de telemetría (Señal Estable)
    stable_idx = model.obs_to_idx['Señal Estable']
    stable_hours = sum(1 for o in obs_signals if o == stable_idx)
    availability_pct = (stable_hours / T) * 100.0

    # 2. Tiempo en cada estado de salud real
    time_per_state = {}
    for idx, name in model.idx_to_state.items():
        hours = sum(1 for s in real_states if s == idx)
        time_per_state[name] = {
            'horas': hours,
            'porcentaje': (hours / T) * 100.0
        }

    # 3. Retraso en la detección de anomalías críticas
    crit_idx = model.state_to_idx['Anomalía Crítica']
    detection_latencies = []
    in_anomaly = False
    start_time = None

    for t in range(T):
        if real_states[t] == crit_idx:
            if not in_anomaly:
                in_anomaly = True
                start_time = t
            # Verificar si Viterbi ya detectó la anomalía en este episodio
            if inferred_states[t] == crit_idx and start_time is not None:
                detection_latencies.append(t - start_time)
                start_time = None  # Marcado como detectado
        else:
            in_anomaly = False
            start_time = None

    mean_detection_delay = (
        float(np.mean(detection_latencies)) if detection_latencies else 0.0
    )

    # 4. Duración de los períodos de pérdida de señal (run-length encoding)
    lost_idx = model.obs_to_idx['Señal Perdida']
    loss_durations = []
    current_run = 0

    for o in obs_signals:
        if o == lost_idx:
            current_run += 1
        else:
            if current_run > 0:
                loss_durations.append(current_run)
                current_run = 0
    if current_run > 0:
        loss_durations.append(current_run)

    mean_loss_duration = float(np.mean(loss_durations)) if loss_durations else 0.0
    max_loss_duration = int(np.max(loss_durations)) if loss_durations else 0

    # 5. Exactitud y matriz de confusión Real vs Inferido
    confusion = np.zeros((len(model.states), len(model.states)), dtype=int)
    for r, inf in zip(real_states, inferred_states):
        confusion[r, inf] += 1

    accuracy = np.trace(confusion) / T * 100.0

    return {
        'total_hours': T,
        'availability_pct': availability_pct,
        'stable_hours': stable_hours,
        'time_per_state': time_per_state,
        'detection_latencies': detection_latencies,
        'mean_detection_delay': mean_detection_delay,
        'loss_episodes_count': len(loss_durations),
        'mean_loss_duration': mean_loss_duration,
        'max_loss_duration': max_loss_duration,
        'confusion_matrix': confusion,
        'accuracy': accuracy
    }


# ==============================================================================
# 3. DASHBOARD Y VISUALIZACIONES EN MATPLOTLIB
# ==============================================================================

def plot_spacecraft_dashboard(
    real_states: List[int],
    obs_signals: List[int],
    inferred_states: List[int],
    model: SpacecraftTelemetryHMM,
    metrics: Dict,
    save_path: str = "spacecraft_telemetry_dashboard.png"
):
    """
    Genera el panel completo de telemetría y diagnóstico aeroespacial.
    """
    T = len(real_states)
    hours = np.arange(T)

    fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.0, 0.9], hspace=0.38, wspace=0.28)

    # --------------------------------------------------------------------------
    # Panel 1: Monitor de "Latidos" y Calidad de Señal con Fondo de Salud Real
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#161b22')

    # Señal mapeada a nivel numérico de "latido" (2: Estable, 1: Intermitente, 0: Pérdida)
    # Convertimos los índices [0, 1, 2] a niveles físicos [2, 1, 0] para la gráfica
    signal_levels = [2 - o for o in obs_signals]

    # Sombrado de fondo según estado real de salud
    state_colors = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c'}
    state_alphas = {0: 0.15, 1: 0.25, 2: 0.40}

    for t in range(T):
        ax1.axvspan(
            t - 0.5, t + 0.5,
            color=state_colors[real_states[t]],
            alpha=state_alphas[real_states[t]],
            edgecolor=None
        )

    # Pulso estilo "ECG / Heartbeat" aeroespacial
    pulse_x, pulse_y = [], []
    for t, val in enumerate(signal_levels):
        pulse_x.extend([t - 0.4, t - 0.1, t, t + 0.1, t + 0.4])
        # Pequeña modulación de pulso en el latido de telemetría
        pulse_y.extend([val, val, val + (0.25 if val > 0 else 0.1), val - 0.1, val])

    ax1.plot(pulse_x, pulse_y, color='#58a6ff', lw=1.6, label='Latido de Telemetría (Downlink)')
    ax1.scatter(hours, signal_levels, color='#79c0ff', s=24, zorder=4)

    # Marcadores de Alerta en Anomalías Críticas
    crit_idx = model.state_to_idx['Anomalía Crítica']
    alert_hours = [t for t in range(T) if real_states[t] == crit_idx]
    if alert_hours:
        ax1.scatter(
            alert_hours, [signal_levels[t] for t in alert_hours],
            color='#ff4d4d', s=90, marker='X', zorder=5, label='Alerta de Anomalía Crítica'
        )

    ax1.set_yticks([2, 1, 0])
    ax1.set_yticklabels(['Estable [OK]', 'Intermitente [WARN]', 'Perdida [LOS]'], fontsize=10, fontweight='bold')
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.set_xlabel('Tiempo Transcurrido de Misión (MET - Horas)', fontsize=11, color='#c9d1d9')
    ax1.set_title('🛰️ TELEMETRÍA DOWNLINK & ESTADO DE SALUD REAL DE LA NAVE (72 HORAS)', fontsize=13, fontweight='bold', color='#58a6ff')
    ax1.grid(True, linestyle='--', alpha=0.3, color='#30363d')

    legend_patches = [
        mpatches.Patch(color='#2ecc71', alpha=0.3, label='Salud: Nominal'),
        mpatches.Patch(color='#f1c40f', alpha=0.4, label='Salud: Degradación'),
        mpatches.Patch(color='#e74c3c', alpha=0.5, label='Salud: Anomalía Crítica'),
    ]
    ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0], loc='upper right', facecolor='#161b22', edgecolor='#30363d')

    # --------------------------------------------------------------------------
    # Panel 2: Comparativa de Estado Real vs Estado Inferido (Viterbi)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor('#161b22')

    ax2.step(hours, real_states, where='mid', label='Estado Real (Ground Truth)', color='#3fb950', lw=2.5, alpha=0.85)
    ax2.step(hours, inferred_states, where='mid', label='Estado Inferido (Viterbi HMM)', color='#f0883e', lw=2.0, linestyle='--')

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Nominal', 'Degradación', 'Crítica'], fontsize=10)
    ax2.set_xlim(-0.5, T - 0.5)
    ax2.set_xlabel('Tiempo Transcurrido de Misión (MET - Horas)', fontsize=11, color='#c9d1d9')
    ax2.set_ylabel('Nivel de Estado', fontsize=11, color='#c9d1d9')
    ax2.set_title('🔍 ESTIMACIÓN DE SALUD MEDIANTE ALGORITMO DE VITERBI', fontsize=12, fontweight='bold', color='#f0883e')
    ax2.grid(True, linestyle=':', alpha=0.3, color='#30363d')
    ax2.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d')

    # --------------------------------------------------------------------------
    # Panel 3: Matriz de Confusión (Real vs Inferido)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor('#161b22')

    labels_short = ['Nominal', 'Degrad.', 'Crítica']
    sns.heatmap(
        metrics['confusion_matrix'], annot=True, fmt='d', cmap='YlGnBu',
        xticklabels=labels_short, yticklabels=labels_short, cbar=False, ax=ax3,
        annot_kws={'fontsize': 12, 'fontweight': 'bold'}
    )
    ax3.set_title(f'🎯 MATRIZ DE CONFUSIÓN\nPrecisión Global: {metrics["accuracy"]:.1f}%', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax3.set_xlabel('Predicción Viterbi', fontsize=10, color='#8b949e')
    ax3.set_ylabel('Estado Verdadero', fontsize=10, color='#8b949e')

    # --------------------------------------------------------------------------
    # Panel 4: Distribución a Largo Plazo y Análisis de Riesgo
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor('#161b22')

    steady = model.compute_steady_state()
    bar_colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    bars = ax4.bar(labels_short, steady * 100, color=bar_colors, alpha=0.85, edgecolor='#ffffff', lw=0.5)
    ax4.set_ylabel('Probabilidad Estacionaria (%)', fontsize=9, color='#8b949e')
    ax4.set_title('📈 DISTRIBUCIÓN ESTACIONARIA\n(Régimen Permanente)', fontsize=10, fontweight='bold', color='#c9d1d9')
    ax4.grid(True, axis='y', linestyle=':', alpha=0.3, color='#30363d')

    for bar, val in zip(bars, steady):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f'{val * 100:.1f}%', ha='center', color='#ffffff', fontsize=9, fontweight='bold')
    ax4.set_ylim(0, max(steady * 100) + 15)

    # --------------------------------------------------------------------------
    # Panel 5: Semáforo y Tarjeta de Estado Actual (Dashboard de Control)
    # --------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.set_facecolor('#161b22')
    ax5.axis('off')

    # Estado final detectado en hora 72
    final_detected_idx = inferred_states[-1]
    final_state_name = model.idx_to_state[final_detected_idx]
    contingency = CONTINGENCY_ACTIONS[final_state_name]

    # Dibujo de Luces de Semáforo
    lamp_colors = {
        'VERDE': '#2ecc71' if contingency['nivel_alerta'] == 'VERDE' else '#143d22',
        'AMARILLO': '#f1c40f' if contingency['nivel_alerta'] == 'AMARILLO' else '#423709',
        'ROJO': '#e74c3c' if contingency['nivel_alerta'] == 'ROJO' else '#3d1614',
    }

    # Caja del semáforo
    ax5.add_patch(mpatches.FancyBboxPatch((0.03, 0.20), 0.16, 0.65, boxstyle="round,pad=0.03", fc="#0d1117", ec="#30363d", lw=2))
    ax5.add_patch(plt.Circle((0.11, 0.70), 0.05, color=lamp_colors['VERDE']))
    ax5.add_patch(plt.Circle((0.11, 0.52), 0.05, color=lamp_colors['AMARILLO']))
    ax5.add_patch(plt.Circle((0.11, 0.35), 0.05, color=lamp_colors['ROJO']))

    # Texto del Semáforo y Protocolo Activo
    ax5.text(0.24, 0.82, f"CONTROL DE MISIÓN: SEMÁFORO DE VUELO", fontsize=12, fontweight='bold', color='#58a6ff')
    ax5.text(0.24, 0.66, f"ESTADO DETECTADO: {final_state_name.upper()}", fontsize=11, fontweight='bold', color=contingency['color'])
    ax5.text(0.24, 0.52, f"PROTOCOLO ACTIVO: {contingency['protocolo']}", fontsize=10, fontweight='bold', color='#c9d1d9')

    # Acción de contingencia envuelta
    ax5.text(0.24, 0.30, f"ORDEN OPERACIONAL:\n{contingency['accion']}", fontsize=9, color='#8b949e', wrap=True)

    # Métricas clave en miniatura
    summary_text = (
        f"• Disponibilidad Señal: {metrics['availability_pct']:.1f}%\n"
        f"• Retraso Detección: {metrics['mean_detection_delay']:.1f} h\n"
        f"• Pérdida de Enlace Máx: {metrics['max_loss_duration']} h"
    )
    ax5.text(0.72, 0.60, summary_text, fontsize=9.5, color='#e6edf3', family='monospace', bbox=dict(boxstyle='square,pad=0.6', fc='#0d1117', ec='#30363d'))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Dashboard gráfico guardado exitosamente en: {save_path}")


# ==============================================================================
# 4. REPORTE DE INGENIERÍA Y CONTROL DE MISIÓN (CONSOLA)
# ==============================================================================

def print_mission_report(
    metrics: Dict,
    model: SpacecraftTelemetryHMM,
    inferred_states: List[int]
):
    """Genera un reporte técnico de telemetría y confiabilidad en consola."""
    sep = "=" * 80
    print("\n" + sep)
    print("🛰️  INFORME DE TELEMETRÍA Y CONFIABILIDAD DE VUELO - MISIÓN 72H")
    print(sep)

    print(f"\n[1] DISPONIBILIDAD DE ENLACE DE TELEMETRÍA:")
    print(f"    • Horas con Señal Estable : {metrics['stable_hours']} / {metrics['total_hours']} h")
    print(f"    • Disponibilidad Global   : {metrics['availability_pct']:.2f}%")
    print(f"    • Episodios de Pérdida    : {metrics['loss_episodes_count']} eventos")
    print(f"    • Duración Media Pérdida  : {metrics['mean_loss_duration']:.2f} horas")
    print(f"    • Duración Máxima Pérdida : {metrics['max_loss_duration']} horas consecutivas")

    print(f"\n[2] TIEMPO EN CADA ESTADO DE SALUD REAL (GROUND TRUTH):")
    for state, data in metrics['time_per_state'].items():
        print(f"    • {state:<22}: {data['horas']:>2} h ({data['porcentaje']:>5.1f}%)")

    print(f"\n[3] RENDIMIENTO DEL ALGORITMO DE VITERBI:")
    print(f"    • Exactitud de Clasificación  : {metrics['accuracy']:.2f}%")
    print(f"    • Retraso Medio en Detección  : {metrics['mean_detection_delay']:.2f} horas")
    if metrics['detection_latencies']:
        print(f"    • Latencias registradas (h)   : {metrics['detection_latencies']}")
    else:
        print("    • Latencias registradas       : Sin eventos de anomalía crítica o detección instantánea")

    print(f"\n[4] ANÁLISIS DE RIESGO Y CONFIABILIDAD ESPACIAL:")
    steady = model.compute_steady_state()
    print("    • Distribución de Estado Estacionario (Largo Plazo):")
    for i, s_name in enumerate(model.states):
        print(f"        - {s_name:<22}: {steady[i] * 100:.2f}%")

    mttf = model.compute_mttf_critical()
    print(f"    • Tiempo Medio hasta Fallo Crítico (MTTF desde Nominal) : {mttf:.1f} horas de vuelo")

    last_state = model.idx_to_state[inferred_states[-1]]
    contingency = CONTINGENCY_ACTIONS[last_state]
    print(f"\n[5] SEMÁFORO DE CONTROL Y ACCIÓN DE CONTINGENCIA FINAL (T=72h):")
    print(f"    • Nivel de Alerta       : [{contingency['nivel_alerta']}]")
    print(f"    • Estado Actual Inferido: {last_state}")
    print(f"    • Protocolo Mandatorio  : {contingency['protocolo']}")
    print(f"    • Instrucción Terrestre : {contingency['accion']}")
    print(sep + "\n")


# ==============================================================================
# 5. BLOQUE DE EJECUCIÓN PRINCIPAL (MAIN)
# ==============================================================================

if __name__ == "__main__":
    # Definición de especificaciones
    hidden_states = ['Operación Nominal', 'Degradación de Sistemas', 'Anomalía Crítica']
    observations = ['Señal Estable', 'Señal Intermitente', 'Señal Perdida']

    # Matriz de transición entre estados de salud de la nave
    A = [
        [0.90, 0.08, 0.02],  # Nominal -> [Nominal, Degradación, Crítica]
        [0.40, 0.45, 0.15],  # Degradación -> [Nominal, Degradación, Crítica]
        [0.05, 0.15, 0.80],  # Crítica -> [Nominal, Degradación, Crítica]
    ]

    # Matriz de emisión (calidad de señal de telemetría observada)
    B = [
        [0.85, 0.12, 0.03],  # Nominal -> [Estable, Intermitente, Perdida]
        [0.30, 0.50, 0.20],  # Degradación -> [Estable, Intermitente, Perdida]
        [0.05, 0.15, 0.80],  # Crítica -> [Estable, Intermitente, Perdida]
    ]

    # Probabilidades iniciales al inicio del enlace
    pi = [0.85, 0.10, 0.05]

    # Instanciación del Modelo Oculto de Markov
    spacecraft_hmm = SpacecraftTelemetryHMM(
        hidden_states=hidden_states,
        observations=observations,
        transition_matrix=A,
        emission_matrix=B,
        initial_distribution=pi
    )

    # Simulación de 72 horas de misión (seed=42 para reproducibilidad en Colab)
    T_MISSION = 72
    real_states, observed_signals = spacecraft_hmm.simulate(T=T_MISSION, seed=42)

    # Inferencia de estados óptimos con algoritmo de Viterbi
    inferred_states = spacecraft_hmm.viterbi(observed_signals)

    # Cálculo integral de métricas
    metrics = analyze_mission_metrics(
        real_states=real_states,
        obs_signals=observed_signals,
        inferred_states=inferred_states,
        model=spacecraft_hmm
    )

    # Impresión de reporte de misión en consola
    print_mission_report(metrics, spacecraft_hmm, inferred_states)

    # Renderizado y guardado del Dashboard de Vuelo
    plot_spacecraft_dashboard(
        real_states=real_states,
        obs_signals=observed_signals,
        inferred_states=inferred_states,
        model=spacecraft_hmm,
        metrics=metrics,
        save_path="spacecraft_telemetry_dashboard.png"
    )
