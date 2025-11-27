# Experiment B1: Symmetry Discovery (Rotational Invariance)

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

This experiment tests whether an optical chaos model can discover that rotational kinetic energy is invariant under coordinate transformations, **without being explicitly told about rotation symmetry**. This tests Noether's theorem—the most fundamental principle in physics: every continuous symmetry corresponds to a conservation law.

**Scientific Question**: Can the model learn that E_rot is the SAME regardless of which direction we call the "x-axis" or "y-axis"?

If successful, this would be **definitive evidence of cage-breaking** via discovery of geometric symmetry principles.

---

## Motivation

### Why This Experiment?

After comprehensive analysis of Experiments 1-10, this experiment was designed to:

1. **Test the Deepest Physics Principle**: Noether's theorem (symmetry → conservation) is more fundamental than F=ma, Maxwell's equations, or even relativity
2. **Fill Critical Gaps**: Symmetry discovery has NEVER been tested in the series
3. **Avoid Known Weaknesses**: No division operations, no trigonometry, no transfer learning
4. **Intermediate Dimensionality**: Tests the untested range (40 inputs) between low-dim success (3 inputs) and high-dim failure (36 inputs)
5. **Clear Falsifiability**: Binary test—rotation invariance YES/NO with quantifiable variance

### Expected Probability of Success

Based on architectural analysis of Experiments 1-10:
- **80% probability** of R² > 0.90 (successful learning)
- **70% probability** of cage-breaking (discovering emergent features)
- **90% probability** of scientifically valid results (regardless of outcome)

---

## Physics Background

### Noether's Theorem

**Emmy Noether's Theorem (1918)**: Every differentiable symmetry of the action of a physical system has a corresponding conservation law.

**Rotational Symmetry → Angular Momentum Conservation**

In our 2D system:
- **Symmetry**: Physics laws don't change if we rotate our coordinate system
- **Conservation**: Angular momentum L_z is conserved
- **Consequence**: Rotational kinetic energy E_rot = L²/(2I) is **rotation-invariant**

### The Physics Formula

```
E_rot = L_z² / (2I)

where:
  L_z = Σ m_i × (x_i × vy_i - y_i × vx_i)  [Angular momentum, z-component]
  I = Σ m_i × (x_i² + y_i²)                 [Moment of inertia]
```

**Key Property**: E_rot computed in ANY rotated coordinate frame gives the SAME value.

This is computed in the **center-of-mass frame** to eliminate translation effects.

---

## Experimental Design

### System Specification

**Physics System**: 10 point masses in 2D space
- **N = 10** particles with random masses (0.1-10 kg)
- Random spatial configurations: circular, elliptical, scattered, clustered
- Random velocity patterns: rotating, expanding, random, stationary

**Input Specification** (40 dimensions):
```
X = [x₁, x₂, ..., x₁₀,    # x-coordinates (10 values)
     y₁, y₂, ..., y₁₀,    # y-coordinates (10 values)
     vx₁, vx₂, ..., vx₁₀,  # x-velocities (10 values)
     vy₁, vy₂, ..., vy₁₀]  # y-velocities (10 values)
```

**Critical Detail**: Coordinate frame is RANDOMLY ROTATED for each sample!

**Output Specification** (1 scalar):
```
y = E_rot = L_z² / (2I)  [Rotational kinetic energy, rotation-invariant]
```

### Novel Aspect: Rotation Test

The **KEY TEST** that distinguishes this experiment:

1. Generate base configuration (e.g., 10 particles in specific positions/velocities)
2. Apply 10 different random rotations to the SAME configuration
3. Model predicts E_rot for each rotated version
4. **Success**: All 10 predictions are nearly identical (variance < 5%)
5. **Failure**: Predictions change significantly with rotation angle

This directly tests if the model discovered rotation symmetry.

---

## Dataset

### Training Set
- **Size**: 4,000 samples
- **Rotation angles**: θ ∈ [0, 2π] uniform random
- **Configurations**: Diverse (circular, elliptical, random, clustered)
- **Velocities**: Diverse (rotating, expanding, random, stationary)

### Test Set
- **Size**: 1,000 samples
- **Same distribution** as training

### Special Test Sets

1. **Rotation Invariance Test**: 500 base configs × 10 rotations each
2. **Rotation Extrapolation**: Train θ ∈ [0, π/4], test θ ∈ [π/4, 2π]
3. **Configuration Extrapolation**: Train circular, test elliptical/random
4. **Noise Robustness**: 5% Gaussian noise added

---

## Models

### Optical Chaos Machine

**Architecture**:
1. **Random Projection**: 40 inputs → 4096 optical features (fixed random matrix)
2. **FFT Mixing**: Simulates wave interference in frequency domain
3. **Intensity Detection**: |FFT|² (magnitude squared)
4. **Nonlinear Activation**: tanh(intensity × brightness)
5. **Ridge Readout**: Linear regression on optical features

**Key**: Reservoir layer is FIXED (no backprop). Only readout trains.

**Hyperparameters**:
- `n_features`: 4096 (optical reservoir size)
- `brightness`: 0.001 (tuned via validation)
- `alpha`: 0.1 (Ridge regularization)

### Darwinian Baseline

**Architecture**:
- Polynomial features (degree 3)
- Ridge regression

**Purpose**: Verify problem is learnable, compare performance

---

## Benchmark Test Suite

### Test 1: Standard Accuracy
- **Metric**: R² score on held-out test set
- **Pass**: R² > 0.90

### Test 2: Rotation Invariance ⭐ **THE KEY TEST**
- **Protocol**: 500 configs × 10 rotations each = 5000 predictions
- **Metric**: Fraction of configs with relative std < 5%
- **Pass**: > 85% of configs pass variance criterion
- **Interpretation**: If PASS, model discovered rotation symmetry!

### Test 3: Rotation Magnitude Extrapolation
- **Protocol**: Train θ ∈ [0°, 45°], test θ ∈ [45°, 360°]
- **Metric**: R² on extrapolation set
- **Pass**: R² > 0.80
- **Interpretation**: Tests if invariance generalizes to unseen angles

### Test 4: Configuration Extrapolation
- **Protocol**: Train on one config type, test on others
- **Metric**: R² on extrapolation set
- **Pass**: R² > 0.70
- **Interpretation**: Tests if learned representation generalizes

### Test 5: Noise Robustness
- **Protocol**: Add 5% Gaussian noise to inputs
- **Metric**: R² with noisy inputs
- **Pass**: R² > 0.80
- **Interpretation**: Tests stability of learned representation

### Test 6: Cage Analysis ⭐ **CRITICAL INTERPRETATION**
- **Protocol**: Correlate internal optical features with:
  - **Cartesian coords** (x, y, vx, vy) → If high correlation (>0.9): LOCKED
  - **Emergent features** (r², v², L_z) → If high correlation (>0.7): BROKEN
- **Pass (Cage Broken)**: max(Cartesian) < 0.5 AND max(emergent) > 0.6
- **Interpretation**: Did model reconstruct coordinates or discover geometry?

---

## Success Criteria

### ✅ PASS - Cage-Breaking Confirmed

**Requirements**:
1. Standard R² > 0.90
2. Rotation invariance pass rate > 0.85
3. Max correlation with Cartesian (x,y) < 0.5
4. Max correlation with emergent (r², L_z) > 0.6
5. At least 2 extrapolation tests R² > 0.70

**Interpretation**:
- Model discovered rotational symmetry WITHOUT being told
- Learned emergent geometric features (r², angular momentum)
- Did NOT reconstruct Cartesian coordinates
- **STRONGEST evidence of cage-breaking in entire series**

### ⚠️ PARTIAL - High Performance, Locked Cage

**Requirements**:
1. Standard R² > 0.95
2. Rotation invariance pass rate > 0.90
3. Max correlation with Cartesian (x,y) > 0.9

**Interpretation**:
- Model learned physics accurately
- Discovered rotation invariance
- But did so via coordinate reconstruction
- Still valuable: validates 40D learning capability
- Suggests 40D is transitional threshold

### ❌ FAIL - Negative Result

**Requirements**:
1. Standard R² < 0.70 OR
2. Rotation invariance pass rate < 0.50

**Interpretation**:
- 40 dimensions exceeded architectural threshold
- Energy calculation too complex
- Mitigation: Reduce to N=5 particles (20D)

**Scientific Value**: Even failure provides valuable data about dimensionality limits

---

## Predicted Outcomes

### Scenario A: Success (70% probability)

**Expected Results**:
- Standard R²: 0.92 - 0.98
- Rotation invariance: 89% pass rate
- Cage: BROKEN (max Cartesian = 0.42, max emergent = 0.78)
- Extrapolation R²: 0.82, 0.74

**Impact**: Strongest cage-breaking evidence via symmetry discovery

### Scenario B: High Performance, Locked Cage (20% probability)

**Expected Results**:
- Standard R²: 0.97
- Rotation invariance: 94% pass rate
- Cage: LOCKED (max Cartesian = 0.93)

**Impact**: Refines dimensionality hypothesis, shows 40D is borderline

### Scenario C: Failure (10% probability)

**Expected Results**:
- Standard R²: 0.62
- Rotation invariance: 45% pass rate

**Impact**: Identifies dimensionality threshold, guides future experiments

---

## Comparison with Previous Experiments

| Experiment | Dimensionality | R² | Cage Status | Key Finding |
|------------|----------------|-----|-------------|-------------|
| 1. Newtonian | 2 (low) | 0.9999 | 🔒 Locked | Learns physics, reconstructs variables |
| 2. Relativity | 2 (low) | 1.0000 | 🔓 Broken | Geometric learning, strong extrapolation |
| 3. Phase | 128 (high) | 0.9998 | 🔓 Broken* | Phase extraction, no extrapolation |
| 10. 2-Body | 3 (low) | 0.9794 | 🔒 Locked | Low-dim success |
| 10. N-Body | 36 (high) | -0.17 | 🔓 Broken | High-dim failure |
| **B1. Symmetry** | **40 (intermediate)** | **?** | **?** | **Tests symmetry discovery** |

*Only within training distribution

**Key Differences**:
- B1 is first to test symmetry discovery explicitly
- B1 has intermediate dimensionality (40D) - untested range
- B1 has structured high-D data (10×4) vs. flat 36D in Exp 10
- B1 uses emergent target (total energy) like successful 2-body case

---

## Implementation Details

### File Structure

```
experiment_B1_symmetry/
├── experiment_B1_symmetry.py      # Main experiment (~550 lines)
├── benchmark_experiment_B1.py     # 6 benchmark tests (~450 lines)
├── README.md                      # This file
└── results/                       # Generated at runtime
    ├── experiment_B1_main_results.png
    ├── rotation_invariance_test.png
    ├── cage_analysis.png
    ├── metrics.json
    └── benchmark_results.json
```

### Key Functions

**Physics Simulator**:
- `calculate_rotational_energy()` - Computes E_rot = L²/(2I)
- `apply_rotation()` - Rotates coordinates by angle θ
- `generate_base_configuration()` - Creates particle system
- `generate_sample()` - Generates (X, y) with random rotation
- `generate_dataset()` - Produces full training/test sets

**Models**:
- `OpticalChaosMachine` - FFT-based chaos model
- `DarwinianModel` - Polynomial baseline

**Validation**:
- Energy invariance check: |E_before - E_after| / E_before < 1e-9
- No NaN/Inf in outputs
- Baseline learnability check: R² > 0.7

---

## How to Run

### Prerequisites

```bash
pip install numpy matplotlib scikit-learn scipy
```

### Basic Execution

```bash
# Run main experiment
python experiment_B1_symmetry.py

# Run comprehensive benchmark suite
python benchmark_experiment_B1.py
```

### Expected Runtime

- Main experiment: ~2-3 minutes (5000 samples)
- Benchmark suite: ~5-7 minutes (includes rotation invariance test with 5000 predictions)

### Outputs

1. **Console Output**: Detailed progress and results
2. **Visualizations**: Saved to `results/` directory
3. **Metrics**: JSON files with quantitative results

---

## Interpretation Guide

### How to Read Results

**If R² > 0.90 AND Rotation Invariance Pass Rate > 0.85**:
1. Check cage analysis correlations
2. If max(Cartesian) < 0.5 and max(emergent) > 0.6:
   - ✅ **CAGE-BREAKING CONFIRMED**
   - Model discovered symmetry without being told!
3. If max(Cartesian) > 0.9:
   - ⚠️ **HIGH PERFORMANCE, LOCKED CAGE**
   - Model learned via coordinate reconstruction

**If R² > 0.70 but < 0.90**:
- Moderate success
- Check if performance improves with reduced dimensionality (N=5)

**If R² < 0.70**:
- Failure mode activated
- 40D likely exceeded threshold
- Recommendation: Retry with N=5 particles (20D)

### Significance of Rotation Invariance Test

**Pass rate > 85%** = Model discovered that:
- Physical laws don't depend on coordinate choice
- Energy is the same in all rotated frames
- Rotation is a SYMMETRY of the system

This is **deeper than learning a formula**—it's learning a **structural principle**.

### Significance of Cage Analysis

**BROKEN cage** = Model learned:
- r² (radial distance squared) - geometric feature
- v² (speed squared) - geometric feature
- L_z (angular momentum) - rotation-invariant quantity

**WITHOUT reconstructing**:
- x, y (Cartesian positions)
- vx, vy (Cartesian velocities)

This means the model discovered a **representation of physics different from human coordinates**.

---

## Scientific Impact

### If PASS (Cage-Breaking Confirmed)

**Immediate Impact**:
- First demonstration of symmetry discovery without human guidance
- Strongest cage-breaking evidence in experimental series
- Validates optical chaos model for geometric learning

**Future Directions**:
1. Test other symmetries (translational, scaling, gauge)
2. Test with N=3 (12D) to see if even lower dimensionality breaks cage
3. Extend to 3D systems
4. Apply to real physics problems (molecular dynamics, astrophysics)

### If PARTIAL (High Performance, Locked Cage)

**Immediate Impact**:
- Refines dimensionality hypothesis
- Shows 40D is transitional threshold
- Validates physics learning at intermediate dimensionality

**Future Directions**:
1. Test N=15 (60D) to find exact breaking point
2. Compare with N=5 (20D) to establish gradient
3. Investigate hybrid approaches (structured + chaos)

### If FAIL (Poor Performance)

**Immediate Impact**:
- Identifies architectural limits
- Provides boundary for dimensionality range

**Future Directions**:
1. Immediate retry with N=5 (20D)
2. Simplify to total KE instead of rotational KE
3. Test alternative architectures

**All outcomes provide scientifically valuable information.**

---

## Validation Checklist

Before trusting results, verify:

### Physics Validation
- [ ] Energy invariant under rotation (error < 1e-9)
- [ ] Energy range spans 2-3 orders of magnitude
- [ ] No NaN/Inf in generated data
- [ ] Baseline R² > 0.7 (problem is learnable)

### Code Quality
- [ ] Fixed random seeds (reproducibility)
- [ ] All functions have docstrings
- [ ] Rotation matrix tested independently
- [ ] Ground truth validated

### Architecture
- [ ] Brightness tuned (0.0001, 0.001, 0.01, 0.1 tested)
- [ ] Ridge alpha verified (0.1 default)
- [ ] MinMaxScaler fit on train only (no data leakage)

---

## References

### Theoretical Background

1. **Noether, E.** (1918). "Invariant Variation Problems." *Göttinger Nachrichten*.
2. **Samid, G.** (2024). "Darwin's Cage: The Trap of Human-Defined Variables in AI."
3. **Angulo, F. (Agnuxo1)** (2024). "Physics vs. Darwin: Experimental Validation Series."

### Related Experiments

- **Experiment 2** (Einstein's Train): Best previous cage-breaking evidence
- **Experiment 10** (N-body): Dimensionality effect on cage status
- **Experiment 1** (Newtonian): Example of locked cage with high performance

---

## Conclusion

Experiment B1 tests the **most fundamental principle in physics**: that physical laws are independent of coordinate system choice. If the optical chaos model discovers this WITHOUT being told, it would be **definitive proof** that AI can learn physics in a fundamentally different way than humans—through emergent geometric features rather than explicit coordinate reconstruction.

This experiment was carefully designed to:
- Avoid all known architectural weaknesses (no division, no trig)
- Fill the biggest gap in the experimental series (symmetry discovery)
- Test the critical intermediate dimensionality range (40D)
- Provide clear, falsifiable predictions
- Deliver scientifically valuable results regardless of outcome

**Predicted probability of obtaining clear, interpretable results: 90%**

**Regardless of whether the cage breaks or locks, we advance understanding of how AI learns physics.**

---

**Last Updated**: November 27, 2025
**Authors**: Francisco Angulo (Agnuxo1) & Claude Code
**Status**: Ready for execution
