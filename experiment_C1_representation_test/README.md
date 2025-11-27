# Experiment C1: The Representation Test
## Direct Falsification of Darwin's Cage Theory

**Author:** Francisco Angulo de Lafuente  
**Date:** 2024  
**Status:** Implementation Complete

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

Experiment C1 is a **direct falsification test** of Darwin's Cage theory. Unlike previous experiments that compared different physical problems or model architectures, C1 uses **rigorous experimental control** to test if input representation alone affects "cage status."

**Key Innovation:** Same physics, same model, same hyperparameters - **ONLY the input representation differs**.

---

## Core Hypothesis (Falsifiable)

### If Darwin's Cage Theory is Correct:
A non-anthropomorphic representation (raw coordinates without human concepts) should result in **LOWER correlation** with human variables in internal features, compared to an anthropomorphic representation (velocity, angle) - even when solving the **SAME physical problem** with the **SAME model architecture**.

### If the Theory is False:
Both representations should show the **SAME correlation patterns**, indicating that representation doesn't affect cage status.

---

## Experimental Design

### Physical Problem
**Projectile Motion** (same as Experiment 1)
- Ground truth: $R = \frac{v_0^2 \sin(2\theta)}{g}$
- Same physics, same difficulty, same learnability
- This ensures we're testing representation, not problem difficulty

### Two Representations (ONLY Difference)

#### 1. Anthropomorphic Representation (Human Variables)
- **Input:** $[v_0, \theta]$ (velocity, angle)
- This is how humans naturally think about the problem
- **Expected:** High correlation with internal features (cage locked)

#### 2. Non-Anthropomorphic Representation (Raw Coordinates)
- **Input:** $[x_0, y_0, v_x, v_y]$ (initial position and velocity components)
- Contains the **SAME information** but without human concepts
- **Expected (if theory correct):** Lower correlation with $v_0$ and $\theta$ (cage broken)

### Critical Controls

✅ **Same model architecture** (OpticalChaosMachine)  
✅ **Same hyperparameters** (n_features=4096, brightness=0.001)  
✅ **Same random seeds** for reproducibility  
✅ **Same dataset size** and distribution  
✅ **Same train/test split**  
✅ **Same evaluation metrics**  
✅ **Same cage analysis methodology** (check ALL features)

**Only variable that changes:** Input representation

---

## Methodology

### 1. Data Generation
- Generate projectile trajectories using same physics simulator
- Convert to both representations
- **Verify information equivalence** (both contain same physical information)

### 2. Model Training
- Train identical models on both representations
- Use same hyperparameters
- Same random seeds for optical matrix

### 3. Evaluation
- **Standard R² score** (both should achieve similar accuracy)
- **Extrapolation test** (generalization)
- **Noise robustness** (stability)
- **Cage Analysis:** Correlation of internal features with:
  - $v_0$ (velocity)
  - $\theta$ (angle)
  - $v_0^2$ (velocity squared)
  - $\sin(2\theta)$ (angle function)

### 4. Statistical Test
- Compare correlation distributions between representations
- Use statistical significance testing (t-test, Mann-Whitney U)
- Report confidence intervals (bootstrap)
- Calculate effect sizes (Cohen's d)

---

## Success Criteria (Falsification Test)

### Theory SUPPORTED if:
- Non-anthropomorphic representation shows **significantly LOWER** max correlation with human variables
- Both achieve similar R² (proving they learned the same physics)
- Statistical test shows **significant difference** (p < 0.05)
- Effect size is meaningful (Cohen's d > 0.5)

### Theory FALSIFIED if:
- Both representations show **similar correlation patterns**
- No significant difference in cage analysis
- Representation **doesn't affect** cage status

**Either outcome is valuable** - we seek truth, not confirmation.

---

## Bias Prevention

1. **No Selection Bias:** Same dataset, just different representation
2. **No Hyperparameter Bias:** Identical hyperparameters
3. **No Architecture Bias:** Same model architecture
4. **No Interpretation Bias:** Pre-defined success criteria
5. **No Confirmation Bias:** Designed to falsify, not confirm
6. **Statistical Rigor:** Proper significance testing

---

## Files Structure

```
experiment_C1_representation_test/
├── experiment_C1_representation_test.py  # Main experiment
├── benchmark_experiment_C1.py            # Rigorous validation
├── README.md                             # This file
├── STATISTICAL_ANALYSIS.md               # Statistical methods
└── RESULTS.md                            # Results and interpretation
```

---

## Usage

### Run Main Experiment
```bash
cd experiment_C1_representation_test
python experiment_C1_representation_test.py
```

### Run Benchmark Validation
```bash
python benchmark_experiment_C1.py
```

---

## Expected Outcomes

### Best Case (Theory Supported):
- Anthropomorphic: Max correlation > 0.9 (cage locked)
- Non-anthropomorphic: Max correlation < 0.3 (cage broken)
- Both: R² > 0.99 (same physics learned)
- Statistical test: p < 0.001
- Effect size: Cohen's d > 0.8 (large)

### Worst Case (Theory Falsified):
- Both: Similar correlation patterns
- No significant difference (p > 0.05)
- Representation doesn't matter
- Theory is falsified

### Intermediate Case:
- Some difference but not statistically significant
- Effect size is small
- Inconclusive - may need more data or different analysis

---

## Scientific Rigor

- ✅ **Pre-registered hypothesis** (this document)
- ✅ **Falsifiable predictions** (clear success/failure criteria)
- ✅ **Controlled variables** (only representation differs)
- ✅ **Statistical testing** (proper significance tests)
- ✅ **Honest reporting** (regardless of outcome)
- ✅ **Reproducible** (all seeds documented)

---

## Key Differences from Previous Experiments

1. **Rigorous Control:** Only representation varies, everything else identical
2. **Direct Falsification:** Clear criteria for supporting/falsifying theory
3. **Statistical Rigor:** Proper significance testing and effect sizes
4. **Information Equivalence:** Verified that representations contain same information
5. **Pre-registered:** Hypothesis and criteria defined before running

---

## References

- Experiment 1: The Chaotic Reservoir (baseline for comparison)
- Comprehensive Experimental Review: Identified need for controlled experiment
- Darwin's Cage Theory: Gideon Samid's original hypothesis

---

## Contact

**Author:** Francisco Angulo (Agnuxo1)  
**Email:** lareliquia.angulo@gmail.com

---

**Status:** Implementation complete. Ready for execution and analysis.

