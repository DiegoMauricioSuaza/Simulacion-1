"""
================================================================================
MONITOR DE TELEMETRÍA EN STREAMING Y ALARMAS DE SALA DE CONTROL
================================================================================
Procesador en tiempo real que ingesta paquetes de telemetría hora a hora,
mantiene la creencia filtrada P(S_t | O_1:t) y dispara alertas operativas.
"""

import numpy as np
from typing import Dict, Any, List

class TelemetryStreamMonitor:
    """
    Monitor de flujo de telemetría en línea.
    """

    STATE_NAMES = ['Operación Nominal', 'Degradación de Sistemas', 'Anomalía Crítica']
    STATE_COLORS = ['VERDE', 'AMARILLO', 'ROJO']
    OBS_NAMES = ['Señal Estable', 'Señal Intermitente', 'Señal Perdida']

    def __init__(self, A: np.ndarray = None, B: np.ndarray = None, pi: np.ndarray = None):
        self.A = np.array(A if A is not None else [
            [0.90, 0.08, 0.02],
            [0.40, 0.45, 0.15],
            [0.05, 0.15, 0.80]
        ], dtype=np.float64)

        self.B = np.array(B if B is not None else [
            [0.85, 0.12, 0.03],
            [0.30, 0.50, 0.20],
            [0.05, 0.15, 0.80]
        ], dtype=np.float64)

        self.pi = np.array(pi if pi is not None else [0.85, 0.10, 0.05], dtype=np.float64)
        
        self.current_hour = 0
        self.belief = self.pi.copy()
        self.history_obs = []
        self.history_beliefs = []
        self.history_states = []

    def ingest_packet(self, obs: int) -> Dict[str, Any]:
        """
        Ingesta una nueva medición de señal (0: Estable, 1: Intermitente, 2: Perdida).
        Actualiza la creencia filtrada online y genera dictamen de vuelo.
        """
        assert obs in [0, 1, 2], "Observación inválida (debe ser 0, 1 o 2)"

        if self.current_hour == 0:
            raw_belief = self.pi * self.B[:, obs]
        else:
            prior = self.belief @ self.A
            raw_belief = prior * self.B[:, obs]

        scale = np.sum(raw_belief)
        self.belief = raw_belief / (scale if scale > 0 else 1.0)

        # Estado más probable según creencia posterior actual
        most_likely_state = int(np.argmax(self.belief))

        self.history_obs.append(obs)
        self.history_beliefs.append(self.belief.copy())
        self.history_states.append(most_likely_state)

        # Generar alerta operativa
        status_color = self.STATE_COLORS[most_likely_state]
        status_name = self.STATE_NAMES[most_likely_state]

        if most_likely_state == 0:
            alert_level = "INFO"
            action = "Continuar operaciones nominales de ciencia y descarga."
        elif most_likely_state == 1:
            alert_level = "WARNING"
            action = "ALERTA AMARILLA: Monitoreo intensivo a 1Hz. Suspender ciencia secundaria."
        else:
            alert_level = "CRITICAL"
            action = "¡EMERGENCIA ROJA! ACTIVAR MODO SEGURO (SAFE-HOLD MODE) INMEDIATO."

        report = {
            "hour": self.current_hour,
            "obs_received": self.OBS_NAMES[obs],
            "estimated_state": status_name,
            "status_color": status_color,
            "alert_level": alert_level,
            "confidence": float(self.belief[most_likely_state]) * 100.0,
            "p_nominal": float(self.belief[0]) * 100.0,
            "p_degrad": float(self.belief[1]) * 100.0,
            "p_critical": float(self.belief[2]) * 100.0,
            "action": action
        }

        self.current_hour += 1
        return report

    def reset(self):
        """Reinicia el monitor para una nueva misión."""
        self.current_hour = 0
        self.belief = self.pi.copy()
        self.history_obs = []
        self.history_beliefs = []
        self.history_states = []

if __name__ == "__main__":
    monitor = TelemetryStreamMonitor()
    # Simular una degradación súbita
    secuencia_prueba = [0, 0, 1, 1, 2, 2, 2]
    print("Ingestando telemetría en streaming:")
    for paquete in secuencia_prueba:
        rep = monitor.ingest_packet(paquete)
        print(f"[T+{rep['hour']:02d}h] Señal: {rep['obs_received']:<18} | Estado: {rep['status_color']:<8} ({rep['confidence']:.1f}%) | {rep['action']}")
