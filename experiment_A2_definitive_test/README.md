# Experiment A2: The Definitive Coordinate Independence Test

## Credits and References

**Darwin's Cage Theory:**
- **Theory Creator**: Gideon Samid
- **Reference**: Samid, G. (2025). Negotiating Darwin's Barrier: Evolution Limits Our View of Reality, AI Breaks Through. *Applied Physics Research*, 17(2), 102. https://doi.org/10.5539/apr.v17n2p102
- **Publication**: Applied Physics Research; Vol. 17, No. 2; 2025. ISSN 1916-9639 E-ISSN 1916-9647. Published by Canadian Center of Science and Education
- **Available at**: https://www.researchgate.net/publication/396377476_Negotiating_Darwin's_Barrier_Evolution_Limits_Our_View_of_Reality_AI_Breaks_Through

**Experiments, AI Models, Architectures, and Reports:**
- **Author**: Francisco Angulo de Lafuente
- **Responsibilities**: Experimental design, AI model creation, architecture development, results analysis, and report writing

---

## Abstract
This experiment corrects the architectural flaw in A1 by using **LSTM** (proper temporal architecture) instead of Reservoir Computing. We test if LSTM can learn chaotic dynamics in twisted coordinates as well as in standard coordinates, while polynomial regression fails.

## Why A2 is Definitive

### What A1 Got Wrong
- Used **static** architecture (Reservoir) for **temporal** task
- Impossible to distinguish "cage locked" from "wrong architecture"

### What A2 Gets Right
- Uses **LSTM** (recurrent, temporal) for temporal prediction
- Fair comparison: Both models are appropriate for the task
- Clear test: Does LSTM maintain performance in twisted coordinates?

## The Test

### System
Double Pendulum with 4 state variables: $(\theta_1, \theta_2, \omega_1, \omega_2)$

### Coordinate Twist
$$u_1 = \theta_1 + 0.5 \sin(\theta_2)$$
$$u_2 = \theta_2 + 0.5 \cos(\theta_1)$$  
$$v_1 = \omega_1 + 0.5 \tanh(\omega_2)$$
$$v_2 = \omega_2 + 0.2 \theta_1 \theta_2$$

### Models
1. **Polynomial (Darwinian):** Degree-3 polynomial for 1-step prediction
2. **LSTM (AI):** 2-layer LSTM (128 units) for multi-step prediction

### Prediction Task
- **Short-term:** Predict 1 step ahead
- **Long-term:** Predict 10 steps ahead (rollout)

## Hypothesis

**If Darwin's Cage is Real:**
- Polynomial: High R² in standard, low R² in twisted (gap > 0.3)
- LSTM: Similar R² in both (gap < 0.1)

**If Darwin's Cage is False:**
- Both models show similar gaps
- Or LSTM also fails in twisted coordinates

## Expected Result
Based on theory, LSTM should be more robust to coordinate changes than polynomial regression, as it learns temporal patterns rather than explicit functional forms.
