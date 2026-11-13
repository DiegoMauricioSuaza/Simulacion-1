"""
================================================================================
SUITE COMPLETA DE TELEMETRÍA Y MODELOS DE MARKOV PARA NAVES ESPACIALES
================================================================================
Ejecución integral de todas las capacidades:
1. Simulación Orbital HMM de 72 horas y Algoritmo de Viterbi
2. Diagrama de Transición de Estados NetworkX
3. Simulación Masiva Monte Carlo (10,000 Misiones) y Curva R(t)
4. Calibración no supervisada Baum-Welch (EM)
5. Monitor de Telemetría en Tiempo Real (Streaming & Alarmas)
================================================================================
"""

import sys
import os
import matplotlib.pyplot as plt

# Incluir src en path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from spacecraft_hmm import SpacecraftHMM, run_spacecraft_mission_simulation
from markov_graph import plot_markov_transition_graph, plot_emission_bipartite_graph
from monte_carlo import run_monte_carlo_fleet, plot_monte_carlo_results
from baum_welch import demonstrate_baum_welch
from telemetry_stream import TelemetryStreamMonitor

def main_suite():
    print("=" * 80)
    print(" INICIANDO SUITE COMPLETA DE TELEMETRÍA ESPACIAL // MARKOV HMM ")
    print("=" * 80)

    # 1. Simulación de Misión de 72 horas
    print("\n[FASE 1/5] Ejecutando Simulación de Misión Orbital de 72 horas...")
    sim_results = run_spacecraft_mission_simulation(n_hours=72, seed=42, display_plots=False)

    # 2. Grafo de Markov NetworkX
    print("\n[FASE 2/5] Generando Grafos de Arquitectura con NetworkX...")
    hmm = sim_results["hmm"]
    fig_graph = plot_markov_transition_graph(hmm.A)
    fig_bipartite = plot_emission_bipartite_graph(hmm.B)

    # 3. Monte Carlo Masivo (10,000 Naves)
    print("\n[FASE 3/5] Ejecutando Análisis Monte Carlo (N = 10,000 misiones)...")
    mc_results = run_monte_carlo_fleet(n_missions=10000, n_hours=72, seed=42)
    fig_mc = plot_monte_carlo_results(mc_results)
    print(f"      • Supervivencia proyectada a T+72h: {mc_results['survival_R'][-1]*100:.1f}%")
    print(f"      • Tiempo Medio a la Primera Falla (MTTF): {mc_results['mttf']:.2f} horas")

    # 4. Calibración Baum-Welch (EM)
    print("\n[FASE 4/5] Calibración de Parámetros No Supervisada (Baum-Welch)...")
    bw_results = demonstrate_baum_welch()
    print(f"      • Algoritmo EM convergido en {bw_results['iterations']} iteraciones.")

    # 5. Monitor de Telemetría en Streaming
    print("\n[FASE 5/5] Prueba de Ingesta en Streaming y Alertas de Vuelo...")
    stream_monitor = TelemetryStreamMonitor(A=hmm.A, B=hmm.B, pi=hmm.pi)
    secuencia_evento = [0, 0, 1, 1, 2, 2, 2]
    for p in secuencia_evento:
        rep = stream_monitor.ingest_packet(p)
        print(f"      • [T+{rep['hour']:02d}h] {rep['obs_received']:<18} -> Estado: [{rep['status_color']}] ({rep['confidence']:.1f}%) | {rep['action']}")

    print("\n" + "=" * 80)
    print(" ✓ TODAS LAS FASES COMPLETADAS EXITOSAMENTE. Mostrando gráficos...")
    print("=" * 80)

    # Mostrar gráficos generados
    from spacecraft_hmm import plot_mission_timeline, plot_mission_control_dashboard
    plot_mission_timeline(sim_results["real_states"], sim_results["observations"], 
                          sim_results["viterbi_states"], sim_results["alpha_probs"])
    plot_mission_control_dashboard(sim_results["metrics"], sim_results["viterbi_states"][-1], 
                                  sim_results["steady_state_pi"], sim_results["risk_evolution"])
    plt.show()

if __name__ == "__main__":
    main_suite()
