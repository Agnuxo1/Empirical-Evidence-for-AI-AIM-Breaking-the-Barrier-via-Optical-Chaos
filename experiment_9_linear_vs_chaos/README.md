# Experiment 9: Linear vs Nonlinear (Chaos)
## Testing Complexity Hypothesis: Predictable vs Chaotic Systems

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

This experiment compares cage status between:
- **Simple Physics**: Linear RLC circuit (predictable, analytical solution)
- **Complex Physics**: Lorenz attractor (chaotic, sensitive to initial conditions)

**Hypothesis**: Chaotic system (complex) should break the cage, while linear system (simple) should lock it.

## Objective

Test whether the complexity of physics (linear vs chaotic) affects cage status:
- Simple physics (linear) → Cage Locked (reconstructs human variables)
- Complex physics (chaotic) → Cage Broken (distributed solution)

## Methodology

### Part A: Linear RLC Circuit (Simple)

**Physics**: $Q(t) = Q_0 e^{-\gamma t} \cos(\omega_d t + \phi)$

**Parameters**:
- $Q_0$: Initial charge [0.1, 10.0] C
- $\gamma$: Damping coefficient [0.1, 2.0] s⁻¹
- $\omega_d$: Damped frequency [0.5, 5.0] rad/s
- $\phi$: Phase [0, $2\pi$] rad
- $t$: Time [0, 10] s

**Input**: $[Q_0, \gamma, \omega_d, \phi, t]$
**Output**: Charge $Q(t)$

**Key Simplicity**: 
- Linear differential equation
- Analytical solution exists
- Predictable behavior

**Expected**: 🔒 CAGE LOCKED (intuitive physics, evolution prepared us)

### Part B: Lorenz Attractor (Complex)

**Physics**: Lorenz system (chaotic differential equations):
- $\dot{x} = \sigma(y - x)$
- $\dot{y} = x(\rho - z) - y$
- $\dot{z} = xy - \beta z$

**Parameters**:
- $\sigma = 10$ (Prandtl number)
- $\rho = 28$ (Rayleigh number)
- $\beta = 8/3$ (geometric factor)

**Initial Conditions**:
- $x_0 \in [-20, 20]$
- $y_0 \in [-20, 20]$
- $z_0 \in [0, 50]$
- $t \in [0, 20]$ s

**Input**: $[x_0, y_0, z_0, t]$
**Output**: $x(t)$ coordinate

**Key Complexity**:
- Nonlinear, coupled equations
- Chaotic behavior (sensitive to initial conditions)
- No analytical solution
- Strange attractor

**Expected**: 🔓 CAGE BROKEN (chaotic, non-linear, evolution didn't prepare us)

### Models

- **Baseline**: Polynomial Regression (degree 4)
- **Chaos Model**: Optical Chaos (4096 features, brightness optimized)

## Results

### Standard Performance

**Part A: Linear RLC Circuit**:
| Model | R² Score |
|-------|----------|
| Darwinian Baseline | -0.2427 |
| **Chaos Model** | **-0.1978** |

**Part B: Lorenz Attractor**:
| Model | R² Score |
|-------|----------|
| Darwinian Baseline | 0.0745 |
| **Chaos Model** | **0.0634** |

### Critical Finding: Low Performance

Both models struggle with these problems:
- **Linear RLC**: R² = -0.20 (fails completely)
- **Lorenz**: R² = 0.06 (very low, but positive)

**Note**: The linear RLC problem also contains trigonometric functions (cos), which may contribute to the difficulty.

### Cage Analysis

**Part A: Linear RLC**:
- Max correlation with **Q0**: 0.9494
- Max correlation with **Gamma**: 0.9687
- Max correlation with **Omega_d**: 0.9751
- Max correlation with **Phase**: 0.9524
- Max correlation with **Time**: 0.9862
- Mean correlation: 0.3184
- **Cage Status**: 🔒 **LOCKED**

**Part B: Lorenz**:
- Max correlation with **x0**: 0.9770
- Max correlation with **y0**: 0.9719
- Max correlation with **z0**: 0.9753
- Max correlation with **Time**: 0.9806
- Mean correlation: 0.3645
- **Cage Status**: 🔒 **LOCKED**

### Extrapolation Tests

**Linear RLC** (Train on t < 5, Test on t ≥ 5):
- Darwinian R²: -4268.46 ❌
- Chaos R²: -3.82 ❌

**Lorenz** (Train on t < 10, Test on t ≥ 10):
- Darwinian R²: -30.65 ❌
- Chaos R²: -0.35 ❌

Both models fail completely at extrapolation.

### Sensitivity Test (Lorenz)

**Test**: Small perturbation (Δx₀ = 0.01) in initial conditions
- Base trajectory: x(5) = -6.5123
- Perturbed trajectory: x(5) = -6.5136
- Divergence: 0.0014
- Amplification factor: 0.14x

**Analysis**: The divergence is weak, possibly because:
1. The time horizon (t=5) may not be long enough for exponential divergence
2. The perturbation may be too small relative to numerical precision
3. The specific initial conditions may not be in a highly sensitive region

### Noise Robustness

- **Linear RLC** (5% noise): R² = -0.22 ❌
- **Lorenz** (5% noise): R² = 0.07 ⚠️

## Discussion

### Key Findings

1. **Both systems have Cage Locked**: Contrary to hypothesis, both linear and chaotic systems show high correlation with human variables (> 0.97).

2. **Low performance**: Both models struggle:
   - Linear RLC: R² = -0.20 (fails completely)
   - Lorenz: R² = 0.06 (very low)

3. **Extrapolation failure**: Both models fail completely when extrapolating beyond training ranges.

4. **Sensitivity test inconclusive**: The Lorenz sensitivity test shows weak divergence, possibly due to test design rather than lack of chaos.

### Why Both Are Locked

**Possible explanations**:

1. **Input reconstruction**: The models may be reconstructing input variables directly rather than learning the physics.

2. **Low performance = unreliable cage analysis**: When models fail to learn (R² < 0 or very low), cage analysis may be less meaningful because the models aren't learning the underlying physics.

3. **Trigonometric limitation (Linear RLC)**: The linear RLC problem contains trigonometric functions (cos), which may make it difficult to learn, similar to Experiment 8.

4. **Chaos difficulty (Lorenz)**: The Lorenz system is genuinely difficult to predict, but the model may still be reconstructing inputs rather than learning the chaotic dynamics.

### Hypothesis Test

**Hypothesis**: Linear system locks cage, chaotic system breaks it.

**Result**: ❌ **HYPOTHESIS NOT CONFIRMED**

- Both systems show **Cage Locked** status
- No clear difference between linear and chaotic systems
- Both systems are difficult to learn (low R²)

### Limitations

1. **Low performance**: The models may not be learning the physics at all, making cage analysis less meaningful.

2. **Trigonometric functions**: The linear RLC problem contains trigonometric functions, which may contribute to learning difficulty.

3. **Extrapolation failure**: Complete failure at extrapolation suggests the models are not learning true physical laws.

4. **Sensitivity test**: The sensitivity test may not be strong enough to demonstrate chaos properties.

## Conclusion

This experiment does **not confirm** the complexity hypothesis. Both linear and chaotic systems show:
- 🔒 **Cage Locked** status (high correlation with human variables)
- Low R² scores (models struggle to learn)
- Complete failure at extrapolation

**Key Insight**: The problems may be too difficult for the models to learn, causing them to fall back to reconstructing input variables rather than learning the physics. This makes cage analysis less meaningful when the model isn't learning the underlying physics.

**Comparison with Experiment 8**: Similar pattern - both simple and complex systems show locked cage status when models fail to learn.

**Future Work**: 
- Test with problems that don't require trigonometric functions
- Verify that models are actually learning physics before cage analysis
- Improve sensitivity tests for chaotic systems

## Files

- `experiment_9_linear_vs_chaos.py`: Main experiment code
- `benchmark_experiment_9.py`: Comprehensive benchmark tests
- `experiment_9_linear_vs_chaos.png`: Results visualization

## Reproduction

```bash
cd experiment_9_linear_vs_chaos
python experiment_9_linear_vs_chaos.py
python benchmark_experiment_9.py
```

