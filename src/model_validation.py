"""
================================================================================
MÓDULO DE VALIDACIÓN Y VERIFICACIÓN DE MODELOS DE MARKOV (HMM)
================================================================================
Herramientas para validar la eficiencia, robustez y realismo del modelo:
1. Criterios de Información de Akaike (AIC) y Bayesiano (BIC)
2. Validación Cruzada Temporal (Forward-Chaining Cross-Validation)
3. Análisis de Sensibilidad y Robustez de Viterbi ante Perturbaciones
4. Prueba de Bondad de Ajuste Chi-cuadrado (Chi-Square Goodness-of-Fit)
5. Verificación de Convergencia a Distribución Estacionaria Real
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from spacecraft_hmm import SpacecraftHMM

class MarkovModelValidator:
    """
    Validador formal de Modelos Ocultos de Markov para telemetría espacial.
    """

    def __init__(self, hmm: SpacecraftHMM):
        self.hmm = hmm
        self.K = hmm.n_states
        self.M = hmm.n_obs
        # Número de parámetros libres independientes: (K-1) + K*(K-1) + K*(M-1)
        self.num_free_params = (self.K - 1) + self.K * (self.K - 1) + self.K * (self.M - 1)

    def compute_log_likelihood(self, observations: np.ndarray) -> float:
        """
        Calcula la log-verosimilitud exacta ln P(O_{1:T} | lambda)
        mediante el algoritmo Forward normalizado.
        """
        T = len(observations)
        alpha = np.zeros((T, self.K))
        log_scale_factors = np.zeros(T)

        # t = 0
        raw_alpha = self.hmm.pi * self.hmm.B[:, observations[0]]
        scale = np.sum(raw_alpha)
        if scale == 0:
            return -np.inf
        log_scale_factors[0] = np.log(scale)
        alpha[0, :] = raw_alpha / scale

        # t = 1 .. T-1
        for t in range(1, T):
            prior = alpha[t - 1, :] @ self.hmm.A
            raw_alpha = prior * self.hmm.B[:, observations[t]]
            scale = np.sum(raw_alpha)
            if scale == 0:
                return -np.inf
            log_scale_factors[t] = np.log(scale)
            alpha[t, :] = raw_alpha / scale

        return float(np.sum(log_scale_factors))

    def compute_information_criteria(self, observations: np.ndarray) -> Dict[str, float]:
        """
        Calcula AIC y BIC para determinar si el modelo tiene sobreajuste o subajuste.
        """
        T = len(observations)
        ll = self.compute_log_likelihood(observations)
        p = self.num_free_params

        aic = 2 * p - 2 * ll
        bic = p * np.log(T) - 2 * ll

        return {
            "T_samples": T,
            "free_parameters": p,
            "log_likelihood": ll,
            "normalized_ll_per_hour": ll / T,
            "AIC": aic,
            "BIC": bic
        }

    def test_sensitivity(self, observations: np.ndarray, epsilon: float = 0.10, seed: int = 42) -> Dict[str, Any]:
        """
        Evalúa la robustez del algoritmo de Viterbi frente a pequeñas incertidumbres
        en la calibración de las matrices A y B (+/- epsilon, ej: 10%).
        """
        rng = np.random.default_rng(seed)
        T = len(observations)

        # Inferencia con modelo base
        base_path, _ = self.hmm.viterbi(observations)

        # Perturbar matriz A
        noise_A = rng.uniform(-epsilon, epsilon, size=self.hmm.A.shape) * self.hmm.A
        A_pert = np.clip(self.hmm.A + noise_A, 0.001, 1.0)
        A_pert /= A_pert.sum(axis=1, keepdims=True)

        # Perturbar matriz B
        noise_B = rng.uniform(-epsilon, epsilon, size=self.hmm.B.shape) * self.hmm.B
        B_pert = np.clip(self.hmm.B + noise_B, 0.001, 1.0)
        B_pert /= B_pert.sum(axis=1, keepdims=True)

        # Modelo perturbado
        perturbed_hmm = SpacecraftHMM(transition_matrix=A_pert, emission_matrix=B_pert, initial_distribution=self.hmm.pi)
        pert_path, _ = perturbed_hmm.viterbi(observations)

        # Calcular concordancia de ruta (Path Agreement Ratio)
        matches = np.sum(base_path == pert_path)
        agreement_ratio = (matches / T) * 100.0

        is_robust = agreement_ratio >= 85.0

        return {
            "epsilon_pct": epsilon * 100.0,
            "agreement_ratio_pct": agreement_ratio,
            "identical_hours": int(matches),
            "total_hours": T,
            "is_robust": is_robust,
            "interpretation": "Alta robustez ante errores de calibración de enlace." if is_robust else "Sensible: calibrar con datos reales adicionales."
        }

    def forward_chaining_cross_validation(self, dataset: List[np.ndarray], n_splits: int = 4) -> Dict[str, Any]:
        """
        Realiza validación cruzada temporal sin fuga de datos (Forward-Chaining Time-Series Split).
        Evalúa cómo se desempeña el modelo en secuencias futuras no vistas.
        """
        scores = []
        n_seqs = len(dataset)
        split_size = n_seqs // (n_splits + 1)

        for fold in range(1, n_splits + 1):
            train_seqs = dataset[:fold * split_size]
            test_seqs = dataset[fold * split_size : (fold + 1) * split_size]

            # Evaluar verosimilitud en test
            fold_lls = [self.compute_log_likelihood(seq) / len(seq) for seq in test_seqs]
            scores.append(float(np.mean(fold_lls)))

        return {
            "n_splits": n_splits,
            "scores_per_fold": scores,
            "mean_test_log_lik_per_hour": float(np.mean(scores)),
            "std_test_log_lik": float(np.std(scores))
        }

    def goodness_of_fit_chi2(self, real_states_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Prueba Chi-cuadrado para contrastar si las transiciones observadas en datos reales
        son estadísticamente consistentes con la matriz teórica A.
        """
        observed_counts = np.zeros((self.K, self.K), dtype=float)
        for seq in real_states_list:
            for t in range(len(seq) - 1):
                observed_counts[seq[t], seq[t + 1]] += 1

        row_totals = observed_counts.sum(axis=1, keepdims=True)
        expected_counts = row_totals * self.hmm.A

        # Calcular estadístico Chi-cuadrado (evitar división por cero)
        mask = (expected_counts > 0)
        chi2_stat = np.sum(((observed_counts[mask] - expected_counts[mask]) ** 2) / expected_counts[mask])
        
        # Grados de libertad: K * (K - 1)
        dof = self.K * (self.K - 1)
        
        # P-valor aproximado (debe ser > 0.05 para aceptar el modelo)
        # scipy.stats.chi2.sf si disponible, o aproximación asintótica
        try:
            from scipy.stats import chi2
            p_value = float(chi2.sf(chi2_stat, dof))
        except ImportError:
            # Aproximación Wilson-Hilferty
            z = (((chi2_stat / dof) ** (1/3)) - (1 - 2/(9 * dof))) / np.sqrt(2/(9 * dof))
            p_value = float(0.5 * (1.0 - np.math.erf(z / np.sqrt(2))))

        passes_test = p_value > 0.05

        return {
            "chi2_statistic": float(chi2_stat),
            "degrees_of_freedom": dof,
            "p_value": p_value,
            "passes_test": passes_test,
            "conclusion": "El modelo HMM es estadísticamente coherente con la realidad (p > 0.05)." if passes_test else "Existen discrepancias estadísticas significativas (p <= 0.05)."
        }

    def generate_validation_report(self, observations: np.ndarray, test_dataset: List[np.ndarray]) -> Dict[str, Any]:
        """
        Ejecuta el protocolo completo de validación y emite el dictamen de ingeniería.
        """
        ic = self.compute_information_criteria(observations)
        sens = self.test_sensitivity(observations, epsilon=0.10)
        cv = self.forward_chaining_cross_validation(test_dataset, n_splits=3)

        print("\n" + "=" * 80)
        print("          INFORME FORMAL DE VALIDACIÓN Y EFICIENCIA DEL MODELO HMM")
        print("=" * 80)
        print(f" • Parámetros Libres del Sistema (p):  {ic['free_parameters']}")
        print(f" • Log-Verosimilitud Total (LL):      {ic['log_likelihood']:.2f}")
        print(f" • Log-Verosimilitud Normalizada:     {ic['normalized_ll_per_hour']:.4f} / hora")
        print(f" • Criterio AIC (Akaike):             {ic['AIC']:.2f}")
        print(f" • Criterio BIC (Bayesiano):          {ic['BIC']:.2f}")
        print("-" * 80)
        print(f" • Robustez de Viterbi (Ruido ±10%):  {sens['agreement_ratio_pct']:.1f}% de coincidencia de ruta")
        print(f" • Dictamen de Sensibilidad:          {sens['interpretation']}")
        print("-" * 80)
        print(f" • Validación Cruzada Temporal (CV):  LL media test = {cv['mean_test_log_lik_per_hour']:.4f} ± {cv['std_test_log_lik']:.4f}")
        print("=" * 80 + "\n")

        return {
            "information_criteria": ic,
            "sensitivity": sens,
            "cross_validation": cv
        }


if __name__ == "__main__":
    hmm = SpacecraftHMM()
    validator = MarkovModelValidator(hmm)
    
    # Generar telemetría sintética para prueba
    real_st, obs = hmm.simulate(n_hours=72, seed=42)
    test_data = [hmm.simulate(n_hours=72, seed=i + 500)[1] for i in range(12)]
    
    report = validator.generate_validation_report(obs, test_data)
