"""
================================================================================
VISUALIZADOR DE GRAFOS DE MARKOV PARA TELEMETRÍA ESPACIAL (NETWORKX + MATPLOTLIB)
================================================================================
Genera diagramas de arquitectura de estados de Markov:
1. Grafo de Transición de Estados Ocultos P(S_t | S_{t-1})
2. Grafo Bipartito de Emisión de Telemetría P(O_t | S_t)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Tuple

def plot_markov_transition_graph(A: np.ndarray, 
                                 state_names: list = None, 
                                 state_colors: list = None, 
                                 save_path: str = None) -> plt.Figure:
    """
    Dibuja el grafo dirigido de la Cadena de Markov con probabilidades de transición,
    bucles de persistencia y estilo aeroespacial oscuro.
    """
    if state_names is None:
        state_names = ['Nominal\n(S0)', 'Degradación\n(S1)', 'Crítica\n(S2)']
    if state_colors is None:
        state_colors = ['#00E676', '#FFD600', '#FF1744']

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Crear grafo dirigido
    G = nx.DiGraph()
    n_states = len(state_names)

    for i in range(n_states):
        G.add_node(i, label=state_names[i], color=state_colors[i])

    # Añadir aristas con pesos
    for i in range(n_states):
        for j in range(n_states):
            prob = A[i, j]
            if prob > 0.01:
                G.add_edge(i, j, weight=prob)

    # Posición fija en triángulo simétrico
    pos = {
        0: np.array([-1.0, -0.6]),  # Nominal (abajo a la izquierda)
        1: np.array([1.0, -0.6]),   # Degradación (abajo a la derecha)
        2: np.array([0.0, 1.0])     # Crítica (arriba en el vértice)
    }

    # Dibujar nodos
    node_colors = [state_colors[i] for i in range(n_states)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=3400,
                           edgecolors='#FFFFFF', linewidths=2.5, alpha=0.92)

    # Etiquetas de nodos
    labels = {i: state_names[i] for i in range(n_states)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold', 
                            font_color='#000000', font_family='sans-serif')

    # Dibujar aristas entre nodos diferentes con curvatura
    edge_styles = [
        (0, 1, 0.20, '#00E5FF'),
        (1, 0, 0.20, '#00E5FF'),
        (0, 2, 0.20, '#FF5252'),
        (2, 0, 0.20, '#69F0AE'),
        (1, 2, 0.20, '#FF1744'),
        (2, 1, 0.20, '#FFD600')
    ]

    for u, v, rad, col in edge_styles:
        if G.has_edge(u, v):
            p = A[u, v]
            ax.annotate("",
                        xy=pos[v], xycoords='data',
                        xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", 
                                        connectionstyle=f"arc3,rad={rad}",
                                        color=col, lw=2.2, alpha=0.85,
                                        mutation_scale=22))
            
            # Punto medio curvo para colocar etiqueta de probabilidad
            mid_x = (pos[u][0] + pos[v][0]) / 2.0
            mid_y = (pos[u][1] + pos[v][1]) / 2.0
            # Desplazamiento según curvatura
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            norm = np.hypot(dx, dy)
            off_x = -dy / norm * (rad * 0.9)
            off_y = dx / norm * (rad * 0.9)

            ax.text(mid_x + off_x, mid_y + off_y, f"P={p:.2f}",
                    color='#E0E6ED', fontsize=9.5, fontweight='bold', fontfamily='monospace',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.25", fc="#161F33", ec=col, lw=1.2, alpha=0.95))

    # Anotar bucles de persistencia (self-loops)
    loop_offsets = {
        0: (-0.35, -0.25, f"P={A[0,0]:.2f}\nPersistencia"),
        1: (0.35, -0.25, f"P={A[1,1]:.2f}\nPersistencia"),
        2: (0.0, 0.38, f"P={A[2,2]:.2f}\nPersistencia")
    }

    for i, (ox, oy, text) in loop_offsets.items():
        x, y = pos[i]
        ax.text(x + ox, y + oy, text,
                color=state_colors[i], fontsize=9, fontweight='bold', fontfamily='monospace',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="#0E1626", ec=state_colors[i], lw=1.5))

    ax.set_title("✦ ARQUITECTURA DEL MODELO DE MARKOV: MATRIZ DE TRANSICIÓN DE ESTADOS A ✦",
                 fontsize=12, fontweight='bold', color='#E0E6ED', pad=20)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.1, 1.6)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


def plot_emission_bipartite_graph(B: np.ndarray, 
                                  state_names: list = None, 
                                  obs_names: list = None, 
                                  save_path: str = None) -> plt.Figure:
    """
    Dibuja el grafo de emisión que mapea los Estados de Salud Ocultos
    a las Observaciones de Telemetría recibidas por las estaciones terrenas.
    """
    if state_names is None:
        state_names = ['Nominal', 'Degradación', 'Crítica']
    if obs_names is None:
        obs_names = ['Señal Estable', 'Señal Intermitente', 'Señal Perdida']

    state_colors = ['#00E676', '#FFD600', '#FF1744']
    obs_colors = ['#00E5FF', '#FFB300', '#FF5252']

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7), facecolor='#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Posiciones X: Estados a la izquierda (x=0), Observaciones a la derecha (x=1)
    y_pos = [0.8, 0.5, 0.2]

    # Dibujar nodos de Estados Ocultos
    for i in range(3):
        ax.scatter(0.1, y_pos[i], s=3200, color=state_colors[i], edgecolors='#FFFFFF', lw=2.0, zorder=4)
        ax.text(0.1, y_pos[i], f"Salud Oculta\n{state_names[i]}", ha='center', va='center',
                color='#000000', fontsize=9.5, fontweight='bold')

    # Dibujar nodos de Observaciones
    for j in range(3):
        ax.scatter(0.9, y_pos[j], s=3200, color=obs_colors[j], edgecolors='#FFFFFF', lw=2.0, zorder=4)
        ax.text(0.9, y_pos[j], f"Telemetría\n{obs_names[j]}", ha='center', va='center',
                color='#000000', fontsize=9.5, fontweight='bold')

    # Dibujar aristas de emisión ponderadas
    for i in range(3):
        for j in range(3):
            prob = B[i, j]
            lw = prob * 5.0 + 0.5
            alpha = max(0.2, prob * 0.9)
            ax.plot([0.1, 0.9], [y_pos[i], y_pos[j]], color=state_colors[i], lw=lw, alpha=alpha, zorder=2)
            
            # Etiqueta de probabilidad sobre la arista
            frac = 0.35 + 0.15 * (j - 1)
            tx = 0.1 + frac * 0.8
            ty = y_pos[i] + frac * (y_pos[j] - y_pos[i])
            if prob >= 0.15:
                ax.text(tx, ty, f"{prob*100:.0f}%", color='#E0E6ED', fontsize=8.5, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="circle,pad=0.2", fc="#101626", ec=state_colors[i], lw=1.0, alpha=0.9))

    ax.set_title("✦ MATRIZ DE EMISIÓN DE TELEMETRÍA B: P(OBSERVACIÓN | ESTADO OCULTO) ✦",
                 fontsize=12, fontweight='bold', color='#E0E6ED', pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.05, 0.95)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig

if __name__ == "__main__":
    A = np.array([
        [0.90, 0.08, 0.02],
        [0.40, 0.45, 0.15],
        [0.05, 0.15, 0.80]
    ])
    B = np.array([
        [0.85, 0.12, 0.03],
        [0.30, 0.50, 0.20],
        [0.05, 0.15, 0.80]
    ])
    plot_markov_transition_graph(A)
    plot_emission_bipartite_graph(B)
    plt.show()
