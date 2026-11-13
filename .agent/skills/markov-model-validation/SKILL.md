---
name: markov-model-validation
description: Comprehensive framework and procedures for validating Hidden Markov Models (HMM) and Markov Chains, assessing model accuracy, goodness-of-fit, information criteria (AIC/BIC), and time-series cross-validation.
---

# Markov Model Validation Skill

Esta skill define los protocolos matemáticos y estadísticos para validar si un Modelo Oculto de Markov (HMM) es eficiente, numéricamente robusto y describe con fidelidad la realidad física observada.

## 1. Criterios de Validación Estadística

### A. Criterios de Selección e Información del Modelo (AIC y BIC)
Evalúan si el número de estados ocultos ($K$) y la estructura de las matrices están sub-ajustados (*underfitting*) o sobre-ajustados (*overfitting*):

- **Log-Verosimilitud ($LL$):**
  $$LL = \ln P(O_{1:T} \mid \lambda) = \sum_{t=1}^T \ln c_t^{-1}$$
  calculada mediante el algoritmo Forward normalizado.

- **Criterio de Información de Akaike (AIC):**
  $$AIC = 2p - 2 LL$$

- **Criterio de Información Bayesiano (BIC):**
  $$BIC = p \ln(T) - 2 LL$$

*Donde $p$ es el número de parámetros libres independientes del modelo:*
$$p = \underbrace{(K - 1)}_{\pi} + \underbrace{K(K - 1)}_{A} + \underbrace{K(M - 1)}_{B}$$
Para $K=3$ estados y $M=3$ observaciones:
$$p = 2 + 3(2) + 3(2) = 14 \text{ parámetros libres}$$

*Regla de decisión:* Modelos con menor AIC/BIC ofrecen el mejor balance entre verosimilitud y parsimonia.

---

### B. Validación Cruzada Temporal (Time-Series Forward-Chaining CV)
En procesos estocásticos de Markov no se debe utilizar K-Fold aleatorio convencional para evitar fuga temporal de información (*data leakage*).
Se debe emplear **Forward-Chaining Split**:
- Split 1: Entrenar en $0 \dots T_1$, validar en $T_1 \dots T_2$.
- Split 2: Entrenar en $0 \dots T_2$, validar en $T_2 \dots T_3$.
- Métrica: Log-verosimilitud media por paso temporal en el conjunto de validación $LL_{\text{val}} / T_{\text{val}}$.

---

### C. Análisis de Sensibilidad y Robustez de Viterbi
Evalúa la estabilidad de la inferencia ante perturbaciones estocásticas:
1. Perturbar cada elemento de $A$ y $B$ en un $\pm \epsilon$ (típicamente $\pm 10\%$), renormalizando por filas.
2. Re-ejecutar el algoritmo de Viterbi sobre la misma secuencia de observaciones.
3. Medir la **Tasa de Coincidencia de Trayectoria (Path Agreement Ratio)**:
   $$J(\hat{S}, \hat{S}_{\text{perturbada}}) = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\hat{S}_t == \hat{S}_{\text{perturbada}, t})$$
   *Si $J > 0.90$, el modelo es robusto ante errores de calibración de sensores.*

---

### D. Prueba de Bondad de Ajuste de Frecuencias de Transición ($\chi^2$)
Compara las frecuencias empíricas de transición $N_{ij}$ observadas en misiones de larga duración con las probabilidades teóricas esperadas $E_{ij} = N_i A_{ij}$:

$$\chi^2 = \sum_{i=1}^K \sum_{j=1}^K \frac{(N_{ij} - E_{ij})^2}{E_{ij}}$$
Con $(K-1)^2$ grados de libertad.

---

## 2. Procedimiento de Ejecución en el Proyecto

Utiliza el validador automático implementado en `src/model_validation.py`:

```python
from src.model_validation import MarkovModelValidator
from src.spacecraft_hmm import SpacecraftHMM

hmm = SpacecraftHMM()
validator = MarkovModelValidator(hmm)

# 1. Calcular AIC, BIC y Verosimilitud
ic_report = validator.compute_information_criteria(observations)

# 2. Validación cruzada temporal
cv_scores = validator.cross_validate(dataset_sequences, n_splits=4)

# 3. Test de robustez y sensibilidad
robustness = validator.test_sensitivity(observations, epsilon=0.10)
```
