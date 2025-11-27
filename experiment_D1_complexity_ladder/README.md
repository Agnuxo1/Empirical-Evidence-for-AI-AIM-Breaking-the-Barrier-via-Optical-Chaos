# Experiment D1: Complexity Phase Transition

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

## Objective

**Systematically map the boundary where cage-breaking begins**

This experiment implements a 5-level complexity ladder to empirically determine the exact dimensionality and complexity threshold at which the optical chaos model transitions from reconstructing human variables (LOCKED) to discovering emergent representations (BROKEN).

---

## Strategic Context

### Why This Experiment?

After 11 experiments (1-10 + B1), we have identified **3 confirmed cases of cage-breaking**:
1. **Experiment 2 (Relativity)**: R²=1.0, max_corr=0.01 (geometric encoding)
2. **Experiment 3 (Phase)**: R²=0.9998 (phase interference)
3. **Experiment 10 (N-body)**: max_corr=0.13 at 36D (dimensionality forcing)

We also have **5 confirmed cases of locked cage**:
- Low-dimensional systems (2-3D): Perfect reconstruction
- Architectural failures (division, trig): Fallback to reconstruction

**Critical Unknown**: What is the exact boundary? When does the transition occur?

### Hypothesis

> **The cage-breaking threshold occurs at ~6-18 dimensions for chaotic dynamical systems**

Based on:
- 2-3D: Consistently LOCKED (Exp 1, 10 2-body)
- 36D: Consistently BROKEN (Exp 10 N-body), but performance fails
- 40D: B1 failed (exceeded capacity)

**D1 tests the untested intermediate range** to find the precise transition point.

---

## Experimental Design

### The Complexity Ladder

Five progressive levels in orbital/dynamical mechanics:

| Level | System | Dim | Chaotic? | Analytical Solution? | Expected Status |
|-------|--------|-----|----------|---------------------|-----------------|
| 1 | Harmonic Oscillator | 4D | No | Yes (x = A cos(ωt+φ)) | 🔒 LOCKED |
| 2 | Kepler 2-Body | 3D | No | Yes (r = a(1-e²)/(1+e cos θ)) | 🔒 LOCKED |
| 3 | Restricted 3-Body | 6D | Partial | No | 🔄 TRANSITION |
| 4 | Unrestricted 3-Body | 18D | Yes | No | 🔓 BROKEN |
| 5 | N-Body (N=7) | 42D | Strongly | No | 🔓 STRONGLY BROKEN |

---

## Level Descriptions

### Level 1: Harmonic Oscillator (4D)

**Physics**: Simple harmonic motion
```
x(t) = A * cos(ω*t + φ)
```

**Inputs**: [ω, A, φ, t] (4 dimensions)
**Output**: x(t) (displacement)

**Characteristics**:
- Fully analytical
- Linear system
- No chaos
- Lowest complexity

**Expected**: LOCKED (model reconstructs ω, A, φ, t)

**Rationale**: Baseline test to confirm low-D behavior

---

### Level 2: Kepler 2-Body (3D)

**Physics**: Planetary orbit (Kepler's equation)
```
r(θ) = a(1 - e²) / (1 + e cos(θ))
```

**Inputs**: [a, e, θ] (3 dimensions)
- a: semi-major axis
- e: eccentricity
- θ: true anomaly

**Output**: r (orbital radius)

**Characteristics**:
- Integrable system
- Conservation laws (energy, angular momentum)
- Known from Exp 10: R²=0.98, max_corr=0.98

**Expected**: LOCKED

**Rationale**: Validates previous results, establishes low-D baseline

---

### Level 3: Restricted 3-Body (6D)

**Physics**: Circular Restricted 3-Body Problem (CR3BP)

Two massive bodies orbit barycenter, test particle moves in their gravitational field.

**Inputs**: [x₀, y₀, vx₀, vy₀, μ, t] (6 dimensions)
- (x₀, y₀): Initial position
- (vx₀, vy₀): Initial velocity
- μ: Mass parameter
- t: Time

**Output**: x(t) (position at time t)

**Characteristics**:
- Some chaotic regions (near Lagrange points)
- No general analytical solution
- Famous for chaos (horseshoe orbits)
- **Critical test**: First potentially chaotic system

**Expected**: TRANSITION (max_corr ≈ 0.5-0.7)

**Rationale**: **THIS IS THE KEY LEVEL** - likely where cage begins to break

---

### Level 4: Unrestricted 3-Body (18D)

**Physics**: Full 3-body problem, all masses free

**Inputs**: [m₁, m₂, m₃, x₁, y₁, x₂, y₂, x₃, y₃, vx₁, vy₁, vx₂, vy₂, vx₃, vy₃, G, t, target] (18 dimensions)

**Output**: x_target(t) (position of target body)

**Characteristics**:
- Fully chaotic
- No general analytical solution (only special cases)
- Sensitive to initial conditions
- **High dimensionality** (18D)

**Expected**: BROKEN (max_corr < 0.4)

**Rationale**: Confirms cage-breaking in intermediate high-D chaotic regime

---

### Level 5: N-Body (42D)

**Physics**: N=7 gravitational bodies

**Inputs**: [m₁...m₇, x₁...x₇, y₁...y₇, vx₁...vx₇, vy₁...vy₇, G, t] (42 dimensions)

**Output**: Total energy E(t)

**Characteristics**:
- Strongly chaotic
- Known from Exp 10: At N=6 (36D), max_corr=0.13, R²=-0.17 (failure)
- **Very high dimensionality**

**Expected**: STRONGLY BROKEN (max_corr < 0.2)

**Rationale**: Confirms strong cage-breaking but potential performance degradation

---

## Key Metrics

### Primary: Cage Status
- **max_corr < 0.5**: BROKEN
- **0.5 ≤ max_corr < 0.7**: TRANSITION
- **max_corr ≥ 0.7**: LOCKED

### Performance
- **R² (test)**: Must be >0.8 for reliable cage analysis
- **R² (extrapolation)**: Tests generalization vs. memorization
- **RMSE**: Quantifies prediction error

### Complexity Indicators
- **Dimensionality**: Input space size
- **Lyapunov exponent**: Quantifies chaos (not computed here, but implicit)
- **Analytical solution**: Yes/No

---

## Success Criteria

### Minimum Viable Success (MVS)
1. ✅ Clear monotonic trend: complexity ↑ → max_corr ↓
2. ✅ Levels 1-2 LOCKED (max_corr > 0.7)
3. ✅ Levels 4-5 BROKEN (max_corr < 0.5)
4. ✅ All levels R² > 0.7 (reliable results)

### Strong Success
- MVS + Level 3 shows TRANSITION (0.5 < max_corr < 0.7)
- Transition occurs between Levels 2-4 (3D to 18D)
- Extrapolation R² > 0.7 for all levels

### Breakthrough Success
- Clear phase transition with sharp boundary
- Quantitative model: max_corr = f(dimensionality, chaos_strength)
- Transfer to predicting cage status for new problems

---

## Falsification Criteria

**Experiment FAILS if:**
1. **No monotonic trend**: max_corr doesn't decrease with complexity
   - **Implication**: Cage status is NOT complexity-dependent

2. **All levels LOCKED**: Even high-D systems reconstruct variables
   - **Implication**: Architecture too weak, or human representations more robust

3. **Performance degradation**: High-D systems have R² < 0.7
   - **Implication**: Dimensionality threshold exceeded, need stronger architecture

4. **Inconsistent with previous experiments**: Contradicts Exp 2, 3, 10 findings
   - **Implication**: Methodology issue, need to reconcile

**All failure modes provide valuable information!**

---

## Implementation Details

### Architecture

**Optical Chaos Machine** (from B1, validated):
- Random projection: 4096 optical features
- FFT-based interference
- Intensity detection: |FFT|²
- Nonlinear activation: tanh(brightness × intensity)
- Ridge readout: α=0.1

**Hyperparameters**:
- `n_features`: 4096
- `brightness`: 0.001 (tuned from B1)
- `alpha`: 0.1

### Datasets

**Training**: 3000 samples per level
**Test**: 500 samples per level
**Extrapolation**: 500 samples with extended parameter ranges

### Cage Analysis

For each input variable i:
1. Compute features = model.get_features(X)
2. Calculate corr_i = max(|corrcoef(X[:, i], features.T)|)
3. max_corr = max(corr_i for all i)
4. Status = BROKEN if max_corr < 0.5, else LOCKED

---

## How to Run

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn scipy
```

### Execution
```bash
python experiment_D1_complexity_ladder.py
```

### Expected Runtime
- **Total**: ~15-20 minutes
- Per level: ~3-4 minutes

### Outputs

**Console**: Detailed progress for each level
**Visualizations**: `results/level_*.png` (5 files)
**Summary**: `results/D1_complete_results.json`
**Phase Transition**: `results/D1_phase_transition_curve.png`

---

## Interpretation Guide

### Reading Results

**If max_corr decreases monotonically**:
✅ **SUCCESS** - Cage status IS complexity-dependent

**If transition occurs at Level 3 (6D)**:
✅ **STRONG SUCCESS** - Boundary identified precisely

**If Levels 1-2 LOCKED, Levels 4-5 BROKEN**:
✅ **MVS ACHIEVED** - Clear boundary exists

**If all levels show similar max_corr**:
❌ **FAILURE** - Cage status not complexity-dependent

**If high-D levels have R² < 0.7**:
⚠️ **PARTIAL** - Architecture capacity exceeded

### Key Questions Answered

1. **What is the dimensionality threshold?**
   - Answer: The dimension at which max_corr drops below 0.5

2. **Is chaos necessary for cage-breaking?**
   - Compare Level 2 (integrable) vs. Level 3 (chaotic) at similar dimensions

3. **Can the model handle high-D without performance loss?**
   - Check if Level 5 maintains R² > 0.8

4. **Is the transition sharp or gradual?**
   - Examine slope of max_corr vs. dimensionality curve

---

## Connection to Research Program

### D1's Role

D1 is **Phase 1** of the 4-phase Physics Discovery Engine:

```
D1 (Boundary Mapping)
  ↓ Identifies threshold τ
D2 (Forced Discovery)
  ↓ Uses τ to design problems
D3 (Law Extraction)
  ↓ Extracts equations from D2
D4 (Cross-Domain Transfer)
  ↓ Tests universality
Physics Discovery Engine
```

**D1 provides the foundation** - without knowing where the cage breaks, we cannot systematically force it (D2) or extract emergent laws (D3).

### Expected Impact

**If Successful**:
- Quantitative threshold for cage-breaking
- Design principles for forcing emergent representations
- Validation that complexity alone can break the cage

**Enables**:
- D2: Design problems at threshold + 50% margin
- D3: Focus on cage-broken models from D1 Levels 4-5
- D4: Test if threshold transfers across domains

---

## Scientific Significance

### Immediate Contributions

1. **Empirical boundary** for cage-breaking in dynamical systems
2. **Validation** that dimensionality + chaos → cage-breaking
3. **Quantitative model** for predicting cage status

### Future Applications

1. **Problem design**: Know when to expect emergent representations
2. **Architecture design**: Match capacity to target complexity
3. **Interpretability**: Understand when models use novel features

### Broader Impact

This is the first systematic mapping of the "Darwin's Cage boundary" - the threshold where AI models transition from reconstructing human variables to discovering genuinely novel representations.

**Potential breakthrough**: If we can reliably predict and induce cage-breaking, we can systematically discover physics beyond human formulations.

---

## Validation Checklist

Before trusting results:

### Physics
- [ ] All simulators validated independently
- [ ] Energy/momentum conservation checked (where applicable)
- [ ] No NaN/Inf in generated data
- [ ] Output ranges span 2-3 orders of magnitude

### Code Quality
- [ ] Fixed random seeds (seed=42)
- [ ] No data leakage (scaler fit on train only)
- [ ] All functions documented
- [ ] Edge cases handled (integration failures)

### Consistency
- [ ] Level 2 reproduces Exp 10 2-body results (R² > 0.95, max_corr > 0.9)
- [ ] Level 5 similar to Exp 10 N-body (max_corr < 0.2)
- [ ] No contradictions with previous experiments

---

## References

### Theoretical Background
1. **Poincaré, H.** (1890). "Sur le problème des trois corps et les équations de la dynamique." (3-body problem)
2. **Lorenz, E.N.** (1963). "Deterministic Nonperiodic Flow." (Chaos theory)
3. **Samid, G.** (2024). "Darwin's Cage: The Trap of Human-Defined Variables in AI."

### Previous Experiments
- **Experiment 2** (Relativity): Best cage-breaking (max_corr=0.01)
- **Experiment 10** (N-body): Dimensionality effect (36D → max_corr=0.13)
- **Experiment B1** (Symmetry): 40D failure (threshold identification)

---

## Next Steps

### If D1 Succeeds

**Immediate**: Analyze phase transition curve
- Fit max_corr = f(dim, chaos)
- Identify optimal complexity for D2

**Next Experiment**: D2 (Forcing Emergent Representations)
- Use threshold + 50% margin
- Design "representation traps"

### If D1 Partially Succeeds

**Scenario**: High-D levels fail (R² < 0.7)
- Reduce N-body from N=7 to N=5 (30D)
- Increase training data (3000 → 5000)
- Tune brightness parameter

### If D1 Fails

**Scenario**: No monotonic trend
- Re-examine hypothesis
- Test alternative complexity measures (Lyapunov exponent)
- Consider architectural modifications

---

## Conclusion

Experiment D1 is the **critical first step** in building a systematic Physics Discovery Engine. By empirically mapping the cage-breaking boundary, we establish:

1. **When** cage-breaking occurs (dimensionality threshold)
2. **Why** it occurs (complexity overwhelms reconstruction)
3. **How** to exploit it (design principles for D2)

This experiment transforms cage-breaking from a **rare observation** (3 cases in 11 experiments) to a **systematic capability** that can be predicted and induced.

**Predicted probability of MVS**: 85%

**Predicted probability of strong success**: 65%

**Expected outcome**: Clear phase transition between Levels 2-4, enabling systematic exploitation in D2-D4.

---

**Last Updated**: November 27, 2025
**Authors**: Francisco Angulo (Agnuxo1) & Claude Code
**Status**: Ready for execution
**Expected Runtime**: ~15-20 minutes
**Part of**: Physics Discovery Engine Research Program (Phase 1/4)
