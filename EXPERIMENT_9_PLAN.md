# Experiment 9: Linear vs Nonlinear (Chaos)
## Testing Complexity Hypothesis: Predictable vs Chaotic Systems

## Objective

Compare cage status between:
- **Simple Physics**: Linear RLC circuit (predictable, analytical solution)
- **Complex Physics**: Lorenz attractor (chaotic, sensitive to initial conditions)

**Hypothesis**: Chaotic system (complex) should break the cage, while linear system (simple) should lock it.

---

## Part A: Linear RLC Circuit (Simple)

### Physics
**Damped Oscillator**: $Q(t) = Q_0 e^{-\gamma t} \cos(\omega_d t + \phi)$

Where:
- $Q_0$: Initial charge [0.1, 10.0] C
- $\gamma$: Damping coefficient [0.1, 2.0] s⁻¹
- $\omega_d$: Damped frequency [0.5, 5.0] rad/s
- $\phi$: Phase [0, $2\pi$] rad
- $t$: Time [0, 10] s

**Key Simplicity**: 
- Linear differential equation
- Analytical solution exists
- Predictable behavior

### Simulator Implementation
```python
class LinearRLCCircuit:
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        Q0 = np.random.uniform(0.1, 10.0, n_samples)
        gamma = np.random.uniform(0.1, 2.0, n_samples)
        omega_d = np.random.uniform(0.5, 5.0, n_samples)
        phi = np.random.uniform(0, 2*np.pi, n_samples)
        t = np.random.uniform(0, 10.0, n_samples)
        
        # Truth: Q(t) = Q0 * exp(-gamma*t) * cos(omega_d*t + phi)
        Q = Q0 * np.exp(-gamma * t) * np.cos(omega_d * t + phi)
        
        X = np.column_stack((Q0, gamma, omega_d, phi, t))
        return X, Q
```

### Expected Results
- **R²**: > 0.99 (high accuracy)
- **Cage Status**: 🔒 **LOCKED** (correlation with Q0, gamma, omega_d > 0.9)
- **Reason**: Linear, predictable, evolution prepared us

---

## Part B: Lorenz Attractor (Complex)

### Physics
**Lorenz System** (chaotic differential equations):
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

**Key Complexity**:
- Nonlinear, coupled equations
- Chaotic behavior (sensitive to initial conditions)
- No analytical solution
- Strange attractor

### Simulator Implementation
```python
class LorenzAttractor:
    def __init__(self):
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
    
    def _lorenz_ode(self, t, state):
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]
    
    def generate_dataset(self, n_samples=2000):
        from scipy.integrate import solve_ivp
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            x0 = np.random.uniform(-20, 20)
            y0 = np.random.uniform(-20, 20)
            z0 = np.random.uniform(0, 50)
            t_eval = np.random.uniform(0, 20)
            
            # Integrate Lorenz system
            sol = solve_ivp(
                self._lorenz_ode,
                [0, t_eval],
                [x0, y0, z0],
                t_eval=[t_eval],
                dense_output=True
            )
            
            # Output: x(t) coordinate
            x_final = sol.y[0][0]
            
            X.append([x0, y0, z0, t_eval])
            y.append(x_final)
        
        return np.array(X), np.array(y)
```

### Expected Results
- **R²**: > 0.90 (moderate accuracy, chaos is hard)
- **Cage Status**: 🔓 **BROKEN** (correlation with x0, y0, z0 < 0.3)
- **Reason**: Chaotic, non-linear, evolution didn't prepare us

---

## Methodology

### 1. Data Generation
- **Part A**: 2000 samples, RLC circuit
- **Part B**: 2000 samples, Lorenz attractor
- Same random seed for reproducibility

### 2. Models
- **Baseline**: Polynomial Regression (degree 4)
- **Chaos Model**: Optical Chaos (4096 features, brightness=0.001)

### 3. Evaluation
- **Standard R²**: Random train/test split (80/20)
- **Cage Analysis**: 
  - Part A: Correlate features with Q0, gamma, omega_d
  - Part B: Correlate features with x0, y0, z0
- **Sensitivity Test**: 
  - Part A: Small variations in parameters
  - Part B: Small variations in initial conditions (chaos should amplify)

### 4. Success Criteria
- **Hypothesis confirmed if**:
  - Part A: Cage LOCKED (correlation > 0.9)
  - Part B: Cage BROKEN (correlation < 0.3)
  - Both achieve reasonable R²

---

## Implementation Checklist

- [ ] Implement `LinearRLCCircuit` simulator
- [ ] Implement `LorenzAttractor` simulator (using scipy.integrate)
- [ ] Create main experiment script with both parts
- [ ] Train baseline and chaos models on both parts
- [ ] Calculate R² scores
- [ ] Perform cage analysis
- [ ] Test sensitivity to initial conditions
- [ ] Create visualizations (phase space for Lorenz)
- [ ] Write benchmark script
- [ ] Document results in README

---

## Files Structure

```
experiment_9_linear_vs_chaos/
├── experiment_9_linear_vs_chaos.py
├── benchmark_experiment_9.py
└── README.md
```

## Notes

- Lorenz system requires numerical integration (scipy.integrate.solve_ivp)
- Chaos makes prediction harder, but we're testing cage status, not just accuracy
- Sensitivity test is crucial: chaos should show exponential divergence

