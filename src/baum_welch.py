"""
================================================================================
ALGORITMO BAUM-WELCH (EXPECTATION-MAXIMIZATION) PARA HMM ESPACIAL
================================================================================
Permite estimar y calibrar las matrices de transición A y de emisión B
a partir de secuencias de telemetría no etiquetadas (sin conocer el estado real).
Útil cuando la nave envejece en órbita y cambian las tasas de fallo reales.
"""

import numpy as np
from typing import Tuple, List, Dict, Any

class BaumWelchTrainer:
    """
    Entrenador no supervisado para Modelos Ocultos de Markov (HMM discretos)
    utilizando el algoritmo de Baum-Welch (EM) con escalado numérico.
    """

    def __init__(self, n_states: int = 3, n_obs: int = 3, max_iter: int = 50, tol: float = 1e-4):
        self.n_states = n_states
        self.n_obs = n_obs
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, observations_list: List[np.ndarray], 
            init_A: np.ndarray = None, 
            init_B: np.ndarray = None, 
            init_pi: np.ndarray = None) -> Dict[str, Any]:
        """
        Ajusta las matrices A, B y Pi a partir de una o múltiples secuencias de telemetría.
        """
        K = self.n_states
        M = self.n_obs

        # Inicialización de matrices (si no se proveen, inicializar con perturbación uniforme)
        rng = np.random.default_rng(123)
        if init_A is not None:
            A = np.copy(init_A)
        else:
            A = rng.uniform(0.1, 1.0, size=(K, K))
            A /= A.sum(axis=1, keepdims=True)

        if init_B is not None:
            B = np.copy(init_B)
        else:
            B = rng.uniform(0.1, 1.0, size=(K, M))
            B /= B.sum(axis=1, keepdims=True)

        if init_pi is not None:
            pi = np.copy(init_pi)
        else:
            pi = rng.uniform(0.1, 1.0, size=K)
            pi /= pi.sum()

        log_likelihood_history = []

        for iteration in range(self.max_iter):
            # Acumuladores para M-step
            exp_transitions = np.zeros((K, K))
            exp_emissions = np.zeros((K, M))
            exp_initial = np.zeros(K)
            total_log_lik = 0.0

            for obs in observations_list:
                T = len(obs)
                # 1. Forward Pass con factores de escalado c_t
                alpha = np.zeros((T, K))
                c = np.zeros(T)

                # t = 0
                alpha_raw = pi * B[:, obs[0]]
                c[0] = 1.0 / (np.sum(alpha_raw) + 1e-16)
                alpha[0, :] = alpha_raw * c[0]

                # t = 1 .. T-1
                for t in range(1, T):
                    alpha_raw = (alpha[t - 1, :] @ A) * B[:, obs[t]]
                    c[t] = 1.0 / (np.sum(alpha_raw) + 1e-16)
                    alpha[t, :] = alpha_raw * c[t]

                log_lik = -np.sum(np.log(c + 1e-16))
                total_log_lik += log_lik

                # 2. Backward Pass con los mismos factores c_t
                beta = np.zeros((T, K))
                beta[T - 1, :] = c[T - 1]

                for t in range(T - 2, -1, -1):
                    beta[t, :] = (A @ (B[:, obs[t + 1]] * beta[t + 1, :])) * c[t]

                # 3. Cálculo de Gamma (P(S_t = i | O)) y Xi (P(S_t = i, S_{t+1} = j | O))
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)

                xi = np.zeros((T - 1, K, K))
                for t in range(T - 1):
                    numerator = alpha[t, :, None] * A * B[None, :, obs[t + 1]] * beta[t + 1, None, :]
                    denom = np.sum(numerator)
                    xi[t, :, :] = numerator / (denom if denom > 0 else 1.0)

                # Acumulación
                exp_initial += gamma[0, :]
                exp_transitions += xi.sum(axis=0)
                for t in range(T):
                    exp_emissions[:, obs[t]] += gamma[t, :]

            # Paso M (Maximización): Re-estimar parámetros
            pi_new = exp_initial / np.sum(exp_initial)
            A_new = exp_transitions / (exp_transitions.sum(axis=1, keepdims=True) + 1e-16)
            B_new = exp_emissions / (exp_emissions.sum(axis=1, keepdims=True) + 1e-16)

            log_likelihood_history.append(total_log_lik)

            # Criterio de convergencia
            delta = np.max(np.abs(A - A_new)) + np.max(np.abs(B - B_new))
            A, B, pi = A_new, B_new, pi_new

            if delta < self.tol:
                break

        return {
            "A_learned": A,
            "B_learned": B,
            "pi_learned": pi,
            "iterations": iteration + 1,
            "log_likelihood_history": log_likelihood_history
        }


def demonstrate_baum_welch():
    """
    Simula una flota de naves espaciales y recalibra las matrices usando Baum-Welch.
    """
    from spacecraft_hmm import SpacecraftHMM
    true_hmm = SpacecraftHMM()

    # Generar 10 secuencias de 72 horas (720 horas de telemetría de vuelo)
    dataset = []
    for s in range(10):
        _, obs = true_hmm.simulate(n_hours=72, seed=s + 100)
        dataset.append(obs)

    # Entrenar HMM con conjetura inicial ruidosa
    trainer = BaumWelchTrainer(n_states=3, n_obs=3, max_iter=30)
    
    # Suposición inicial con cierta incertidumbre
    noisy_A = np.array([
        [0.80, 0.15, 0.05],
        [0.30, 0.50, 0.20],
        [0.10, 0.20, 0.70]
    ])
    result = trainer.fit(dataset, init_A=noisy_A)
    return result

if __name__ == "__main__":
    res = demonstrate_baum_welch()
    print("Baum-Welch convergido en", res["iterations"], "iteraciones.")
    print("Matriz A calibrada:\n", np.round(res["A_learned"], 3))
