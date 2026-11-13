"""
================================================================================
SIMULADOR MONTE CARLO MASIVO PARA CONFIABILIDAD ESPACIAL (10,000 MISIONES)
================================================================================
Calcula empíricamente:
- Curva de Supervivencia de Misión R(t) = P(Sin anomalía crítica hasta t)
- Tiempo Medio hasta la Primera Falla Crítica (MTTF - Mean Time To Failure)
- Tasa de Falla Horaria λ(t)
- Disponibilidad del enlace de telemetría a escala de flota
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

def run_monte_carlo_fleet(n_missions: int = 10000, 
                          n_hours: int = 72, 
                          seed: int = 42) -> Dict[str, Any]:
    """
    Ejecuta una simulación vectorial ultra-rápida de n_missions naves simultáneas.
    """
    rng = np.random.default_rng(seed)

    # Matrices del sistema
    A = np.array([
        [0.90, 0.08, 0.02],
        [0.40, 0.45, 0.15],
        [0.05, 0.15, 0.80]
    ])
    pi = np.array([0.85, 0.10, 0.05])

    # Distribución acumulada para muestreo rápido
    A_cdf = np.cumsum(A, axis=1)
    pi_cdf = np.cumsum(pi)

    # Matriz de estados: (n_missions, n_hours)
    states = np.zeros((n_missions, n_hours), dtype=np.int8)

    # t = 0
    u0 = rng.uniform(0, 1, size=n_missions)
    states[:, 0] = np.digitize(u0, pi_cdf)

    # t = 1 .. n_hours-1
    for t in range(1, n_hours):
        prev = states[:, t - 1]
        u = rng.uniform(0, 1, size=n_missions)
        # vectorized choice based on previous state
        cdfs = A_cdf[prev]  # shape (n_missions, 3)
        states[:, t] = (u[:, None] > cdfs).sum(axis=1)

    # 1. Curva de Supervivencia R(t): % de misiones que NO han tocado estado 2 (Crítica) hasta t
    # Identificar el primer instante donde cada misión entró en estado Crítico (2)
    has_failed = (states == 2)
    first_failure_time = np.full(n_missions, n_hours + 1)

    for i in range(n_missions):
        crit_indices = np.where(has_failed[i])[0]
        if len(crit_indices) > 0:
            first_failure_time[i] = crit_indices[0]

    hours = np.arange(n_hours)
    survival_R = np.zeros(n_hours)
    for t in range(n_hours):
        survival_R[t] = np.mean(first_failure_time > t)

    # 2. Tiempo Medio hasta la Falla (MTTF) para las naves que fallaron dentro de las 72h
    failed_mask = first_failure_time <= n_hours
    mttf = float(np.mean(first_failure_time[failed_mask])) if np.any(failed_mask) else float(n_hours)
    fleet_loss_pct = float(np.mean(failed_mask)) * 100.0

    # 3. Fracción de tiempo en cada estado
    total_samples = n_missions * n_hours
    pct_nominal = float(np.sum(states == 0) / total_samples) * 100.0
    pct_degrad = float(np.sum(states == 1) / total_samples) * 100.0
    pct_crit = float(np.sum(states == 2) / total_samples) * 100.0

    return {
        "n_missions": n_missions,
        "n_hours": n_hours,
        "survival_R": survival_R,
        "first_failure_time": first_failure_time,
        "mttf": mttf,
        "fleet_loss_pct": fleet_loss_pct,
        "pct_nominal": pct_nominal,
        "pct_degrad": pct_degrad,
        "pct_crit": pct_crit
    }


def plot_monte_carlo_results(mc_data: Dict[str, Any], save_path: str = None) -> plt.Figure:
    """
    Visualiza la función de fiabilidad R(t) y el histograma de fallas de la flota.
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#0B0F19')
    ax1.set_facecolor('#101626')
    ax2.set_facecolor('#101626')

    hours = np.arange(mc_data["n_hours"])
    R = mc_data["survival_R"] * 100

    # Gráfico 1: Curva de Supervivencia R(t)
    ax1.plot(hours, R, color='#00E5FF', lw=2.5, label='Confiabilidad R(t)')
    ax1.fill_between(hours, 0, R, color='#00E5FF', alpha=0.15)
    ax1.axhline(50, color='#FFD600', linestyle=':', label='Mediana de Supervivencia (50%)')
    
    final_R = R[-1]
    ax1.scatter(hours[-1], final_R, color='#FF1744', s=80, zorder=5)
    ax1.annotate(f"Supervivencia T+72h:\n{final_R:.1f}%", xy=(hours[-1], final_R), xytext=(hours[-1]-18, final_R - 15),
                 arrowprops=dict(facecolor='#FF1744', arrowstyle="->", lw=1.2),
                 fontsize=9, fontweight='bold', color='#FF5252',
                 bbox=dict(boxstyle="round,pad=0.2", fc="#2A0B14", ec="#FF1744"))

    ax1.set_title(f"CURVA DE SUPERVIVENCIA DE MISIÓN R(t) (N = {mc_data['n_missions']:,} NAVES)",
                  fontsize=11, fontweight='bold', color='#E0E6ED')
    ax1.set_xlabel("Horas de Misión (MET)", fontsize=10, color='#90CAF9')
    ax1.set_ylabel("Probabilidad de No Falla (%)", fontsize=10, color='#90CAF9')
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax1.legend(loc='lower left', fontsize=8.5, facecolor='#161F33', edgecolor='#37474F')

    # Gráfico 2: Distribución de Tiempos a la Primera Falla Crítica
    fft = mc_data["first_failure_time"]
    failed = fft[fft <= mc_data["n_hours"]]
    
    n, bins, patches = ax2.hist(failed, bins=24, color='#FF1744', alpha=0.8, edgecolor='#0B0F19')
    ax2.axvline(mc_data["mttf"], color='#FFD600', lw=2.0, linestyle='--',
                label=f"MTTF Promedio = {mc_data['mttf']:.1f} h")

    ax2.set_title(f"HISTOGRAMA DE PRIMERA ANOMALÍA CRÍTICA (TASA DE FALLO)",
                  fontsize=11, fontweight='bold', color='#E0E6ED')
    ax2.set_xlabel("Hora de Primera Anomalía Crítica", fontsize=10, color='#90CAF9')
    ax2.set_ylabel("Número de Naves", fontsize=10, color='#90CAF9')
    ax2.grid(True, linestyle=':', alpha=0.3, color='#455A64')
    ax2.legend(loc='upper right', fontsize=8.5, facecolor='#161F33', edgecolor='#37474F')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig

if __name__ == "__main__":
    res = run_monte_carlo_fleet(n_missions=10000, n_hours=72)
    print(f"Simuladas {res['n_missions']} misiones.")
    print(f"Tasa de naves que experimentaron anomalía crítica: {res['fleet_loss_pct']:.2f}%")
    print(f"MTTF promedio: {res['mttf']:.2f} horas")
    plot_monte_carlo_results(res)
    plt.show()
