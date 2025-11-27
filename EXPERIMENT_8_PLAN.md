# Experiment 8: Classical vs Quantum Mechanics
## Testing Complexity Hypothesis: Simple vs Complex Physics

## Objective

Compare cage status between:
- **Simple Physics**: Classical harmonic oscillator (intuitive, analytical solution)
- **Complex Physics**: Quantum particle in a box (counterintuitive, discrete states)

**Hypothesis**: Quantum system (complex) should break the cage, while classical system (simple) should lock it.

---

## Part A: Classical Harmonic Oscillator (Simple)

### Physics
**Equation**: $x(t) = A \cos(\omega t + \phi)$

Where:
- $A$: Amplitude [0.1, 10.0] m
- $\omega$: Angular frequency [0.5, 5.0] rad/s
- $\phi$: Phase [0, $2\pi$] rad
- $t$: Time [0, 10] s

### Simulator Implementation
```python
class ClassicalHarmonicOscillator:
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        A = np.random.uniform(0.1, 10.0, n_samples)
        omega = np.random.uniform(0.5, 5.0, n_samples)
        phi = np.random.uniform(0, 2*np.pi, n_samples)
        t = np.random.uniform(0, 10.0, n_samples)
        
        # Truth: x(t) = A * cos(omega*t + phi)
        x = A * np.cos(omega * t + phi)
        
        X = np.column_stack((A, omega, phi, t))
        return X, x
```

### Expected Results
- **R²**: > 0.99 (high accuracy)
- **Cage Status**: 🔒 **LOCKED** (correlation with A, omega, phi > 0.9)
- **Reason**: Intuitive physics, evolution prepared us for this

---

## Part B: Quantum Particle in a Box (Complex)

### Physics
**Wave Function**: $\psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)$

**Probability Density**: $|\psi_n(x)|^2 = \frac{2}{L} \sin^2\left(\frac{n\pi x}{L}\right)$

Where:
- $n$: Quantum number (1, 2, 3, ..., 10) - **DISCRETE**
- $L$: Box width [1.0, 10.0] m
- $x$: Position [0, L] m

**Key Complexity**: 
- Quantization (discrete n)
- Non-intuitive (probability, not position)
- No classical analog

### Simulator Implementation
```python
class QuantumParticleInBox:
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        n = np.random.randint(1, 11, n_samples)  # Discrete quantum number
        L = np.random.uniform(1.0, 10.0, n_samples)
        x = np.random.uniform(0, L, n_samples)  # Position within box
        
        # Truth: |psi|^2 = (2/L) * sin^2(n*pi*x/L)
        prob_density = (2.0 / L) * np.sin(n * np.pi * x / L)**2
        
        X = np.column_stack((n, L, x))
        return X, prob_density
```

### Expected Results
- **R²**: > 0.95 (high accuracy)
- **Cage Status**: 🔓 **BROKEN** (correlation with n, L < 0.3)
- **Reason**: Counterintuitive physics, evolution didn't prepare us

---

## Methodology

### 1. Data Generation
- **Part A**: 2000 samples, classical oscillator
- **Part B**: 2000 samples, quantum particle
- Same random seed for reproducibility

### 2. Models
- **Baseline**: Polynomial Regression (degree 4)
- **Chaos Model**: Optical Chaos (4096 features, brightness=0.001)

### 3. Evaluation
- **Standard R²**: Random train/test split (80/20)
- **Cage Analysis**: 
  - Part A: Correlate features with A, omega, phi
  - Part B: Correlate features with n, L
- **Extrapolation**: 
  - Part A: Train on t < 5, test on t > 5
  - Part B: Train on n ≤ 5, test on n > 5

### 4. Success Criteria
- **Hypothesis confirmed if**:
  - Part A: Cage LOCKED (correlation > 0.9)
  - Part B: Cage BROKEN (correlation < 0.3)
  - Both achieve high R² (> 0.95)

---

## Implementation Checklist

- [ ] Implement `ClassicalHarmonicOscillator` simulator
- [ ] Implement `QuantumParticleInBox` simulator
- [ ] Create main experiment script with both parts
- [ ] Train baseline and chaos models on both parts
- [ ] Calculate R² scores
- [ ] Perform cage analysis (correlation with human variables)
- [ ] Test extrapolation
- [ ] Create visualizations comparing both parts
- [ ] Write benchmark script
- [ ] Document results in README

---

## Files Structure

```
experiment_8_classical_vs_quantum/
├── experiment_8_classical_vs_quantum.py
├── benchmark_experiment_8.py
└── README.md
```

