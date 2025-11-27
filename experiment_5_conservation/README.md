# Experiment 5: Conservation Laws Discovery (The Hidden Symmetry)

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

This experiment investigates whether a chaotic optical AI system can discover physical conservation laws (energy and momentum conservation) without explicit knowledge of these concepts. We test whether the system reconstructs human variables (energy, momentum) or finds distributed representations that capture conservation without explicit variable reconstruction.

## Objective

To determine if a chaos-based optical reservoir can:
1. Learn to predict collision outcomes accurately
2. Discover that certain quantities are conserved (momentum always, energy only in elastic collisions)
3. Do so without reconstructing the human variables of "energy" and "momentum"

## Methodology

### 1. Physical Simulator

**Domain**: 1D Collisions (Elastic and Inelastic)

**Elastic Collisions** ($e = 1.0$):
- **Conservation Laws**: Both momentum and kinetic energy are conserved
- **Formula**: 
  - Momentum: $m_1v_1 + m_2v_2 = m_1v'_1 + m_2v'_2$
  - Energy: $\frac{1}{2}m_1v_1^2 + \frac{1}{2}m_2v_2^2 = \frac{1}{2}m_1v'^2_1 + \frac{1}{2}m_2v'^2_2$
- **Solution**: 
  $$v'_1 = \frac{(m_1 - m_2)v_1 + 2m_2v_2}{m_1 + m_2}$$
  $$v'_2 = \frac{(m_2 - m_1)v_2 + 2m_1v_1}{m_1 + m_2}$$

**Inelastic Collisions** ($e \in [0, 0.9]$):
- **Conservation Laws**: Only momentum is conserved, energy is dissipated
- **Coefficient of Restitution**: $e = \frac{v'_2 - v'_1}{v_1 - v_2}$
- **Solution**:
  $$v'_1 = \frac{m_1v_1 + m_2v_2 - m_2e(v_2 - v_1)}{m_1 + m_2}$$
  $$v'_2 = \frac{m_1v_1 + m_2v_2 + m_1e(v_2 - v_1)}{m_1 + m_2}$$

### 2. Datasets

- **Elastic Dataset**: 3,000 samples, $e = 1.0$
- **Inelastic Dataset**: 2,000 samples, $e \in [0, 0.9]$
- **Mixed Dataset**: 1,000 samples, $e \in [0, 1.0]$ (for transfer testing)

**Parameter Ranges**:
- Masses: $m_1, m_2 \in [0.1, 10.0]$ kg
- Velocities: $v_1, v_2 \in [-50, 50]$ m/s

### 3. Models

**Baseline (Darwinian)**:
- Polynomial Features (degree 4)
- Ridge Regression
- Expected to learn explicit formulas

**Optical Chaos Model**:
- Input: $[m_1, m_2, v_1, v_2, e]$
- Random projection (4096 features)
- FFT mixing
- Ridge readout
- No explicit feature engineering

### 4. Evaluation Metrics

- **R² Score**: Prediction accuracy
- **Conservation Errors**: 
  - Momentum error: $|p_{final} - p_{initial}|$
  - Energy error: $|E_{final} - E_{initial}|$
- **Cage Analysis**: Correlation of internal features with:
  - Total energy: $E = \frac{1}{2}m_1v_1^2 + \frac{1}{2}m_2v_2^2$
  - Total momentum: $p = m_1v_1 + m_2v_2$

## Results

### Standard Performance

**Within-Domain (Elastic Collisions)**:
| Model | R² Score |
|-------|----------|
| Darwinian Baseline | 0.9976 |
| **Optical Chaos Model** | **0.2781** |

**Transfer (Elastic → Inelastic)**:
| Model | R² Score |
|-------|----------|
| Darwinian Baseline | -0.1198 |
| **Optical Chaos Model** | **-0.2607** |

### Critical Audit (`benchmark_experiment_5.py`)

#### 1. Extrapolation (Mass Range)
- **Test**: Train on masses < 10 kg, test on masses ≥ 10 kg
- **Result**: R² = 0.0469 ❌
- **Analysis**: Model fails to generalize to unseen mass ranges, indicating overfitting

#### 2. Transfer Learning
- **Test**: Train on elastic, test on inelastic
- **Result**: R² = -0.3476 ❌
- **Analysis**: Model is domain-specific and fails to transfer knowledge

#### 3. Conservation Law Verification
- **Momentum Error**: Mean = 289.79, Max = 1315.53
- **Energy Error**: Mean = 4869.37, Max = 55580.73
- **Status**: ❌ **FAIL** - Model violates conservation laws significantly

#### 4. Cage Analysis
- **Max Correlation with Energy**: 0.7818
- **Max Correlation with Momentum**: 0.8266
- **Mean Correlation with Energy**: 0.2593
- **Mean Correlation with Momentum**: 0.2946
- **Status**: 🔒 **CAGE LOCKED** - Model reconstructed human variables

## Discussion

### Key Findings

1. **Poor Performance**: The chaos model achieves only R² = 0.28 on elastic collisions, significantly worse than the baseline (R² = 0.998). This suggests the model struggles with this particular problem.

2. **No Transfer**: Both models fail to transfer from elastic to inelastic collisions, indicating domain-specific learning.

3. **Conservation Violations**: The chaos model's predictions violate conservation laws significantly, with momentum errors averaging ~290 units and energy errors averaging ~4,870 units.

4. **Cage Status**: The model shows high correlation with momentum (0.83) and moderate-high correlation with energy (0.78), indicating it reconstructed these human variables rather than finding distributed representations.

### Critical Review and Validation

**CRITICAL VALIDATION PERFORMED**: We conducted extensive testing to ensure results are genuine, not experimental artifacts.

1. **Output Scaling Test**: Applied StandardScaler to outputs → R² = 0.2799 (no improvement)
2. **Hyperparameter Tuning**: Tested brightness [0.0001, 0.001, 0.01, 0.1, 1.0] → brightness=0.001 is optimal
3. **Baseline Comparison**: Polynomial baseline achieves R² = 0.9949 ✅ (problem IS learnable)
4. **Data Validation**: Physics simulator verified - conservation errors < 1e-12 (perfect)
5. **More Data Test**: Increased to 1600 samples → R² = 0.2726 (no improvement)

**CONCLUSION**: The low R² = 0.28 is a **genuine model limitation**, not a design flaw. The problem is learnable (baseline succeeds), but the chaos model fails.

### Limitations

1. **Division Operations**: The collision formula involves division: $v'_1 = \frac{(m_1-m_2)v_1 + 2m_2v_2}{m_1+m_2}$. The chaos model may struggle with division compared to multiplication (which it handles well in Experiments 1-2).

2. **Feature Expressiveness**: The FFT transformation may not naturally encode division operations that are central to collision physics.

3. **Architecture Mismatch**: While the chaos model excels at multiplicative relationships (Experiments 1-2), it fails at division-based relationships (this experiment).

4. **Baseline Success**: The polynomial baseline (R² = 0.99) proves the problem is learnable, just not by this architecture.

### Comparison with Other Experiments

| Experiment | Domain | Chaos R² | Cage Status |
|------------|--------|----------|-------------|
| 1. Newtonian | Ballistics | 0.9999 | 🔒 Locked |
| 2. Relativity | Time Dilation | 1.0000 | 🔓 Broken |
| 3. Absolute Frame | Hidden Variables | 0.9998 | 🔓 Broken* |
| 4. Transfer | Cross-Domain | -0.51 to -247 | ❌ Failed |
| **5. Conservation** | **Collisions** | **0.2781** | **🔒 Locked** |

*Only within training distribution

### Implications

1. **Not All Physics is Equal**: The chaos model excels at some problems (Experiments 1-3) but fails at others (Experiments 4-5). This suggests that the effectiveness of chaos-based approaches depends on the specific problem structure.

2. **Conservation Laws are Hard**: Discovering conservation laws may require explicit architectural biases or different learning paradigms than pure pattern matching.

3. **Variable Reconstruction**: When the model does learn (even partially), it tends to reconstruct human variables rather than finding novel distributed representations.

## Conclusion

This experiment demonstrates that discovering conservation laws through chaos-based learning is challenging. The model:
- **Fails to learn effectively** (R² = 0.28) - validated as genuine limitation, not experimental artifact
- **Violates conservation laws** in its predictions
- **Reconstructs human variables** (energy, momentum) rather than finding distributed representations
- **Does not transfer** knowledge between elastic and inelastic collisions

**Key Finding**: The polynomial baseline achieves R² = 0.99, proving the problem is learnable. The chaos model's failure is a genuine architectural limitation, specifically with division operations.

**Comparison with Other Experiments**:
- Experiment 1 (multiplicative: v²): R² = 0.9999 ✅
- Experiment 2 (multiplicative: √): R² = 1.0000 ✅
- Experiment 5 (divisive: /): R² = 0.2799 ❌

This suggests the chaos model excels at multiplicative relationships but struggles with division.

These results suggest that conservation laws may require:
1. **Explicit architectural constraints** to enforce conservation
2. **Different learning paradigms** beyond pure pattern matching
3. **Hybrid approaches** combining chaos with explicit physical constraints
4. **Feature engineering** to help with division operations

The experiment provides valuable negative results, showing the limitations of pure chaos-based approaches for division-based physical problems. **The results are validated and genuine** - not experimental artifacts.

## Files

- `experiment_5_conservation.py`: Main experiment code
- `benchmark_experiment_5.py`: Critical audit and cage analysis
- `experiment_5_conservation.png`: Performance visualizations
- `benchmark_5_results.png`: Benchmark results

## Reproduction

```bash
python experiment_5_conservation.py
python benchmark_experiment_5.py
```

## References

This experiment is part of the "Darwin's Cage" series investigating whether AI systems can discover physical laws without human conceptual biases. The results demonstrate both the potential and limitations of chaos-based approaches.

