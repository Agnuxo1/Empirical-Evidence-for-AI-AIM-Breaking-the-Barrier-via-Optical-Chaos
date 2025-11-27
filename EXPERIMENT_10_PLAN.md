# Experiment 10: Low vs High Dimensionality
## Testing Complexity Hypothesis: Few-Body vs Many-Body Systems

## Objective

Compare cage status between:
- **Simple Physics**: 2-body gravitational system (Kepler orbits, analytical solution)
- **Complex Physics**: N-body gravitational system (N=5-10, no analytical solution, chaotic)

**Hypothesis**: Many-body system (complex, high-dimensional) should break the cage, while 2-body system (simple, low-dimensional) should lock it.

---

## Part A: 2-Body System (Simple)

### Physics
**Keplerian Orbit**: $r(\theta) = \frac{a(1-e^2)}{1+e\cos(\theta)}$

Where:
- $a$: Semi-major axis [1.0, 10.0] AU
- $e$: Eccentricity [0, 0.9]
- $\theta$: True anomaly [0, $2\pi$] rad

**Key Simplicity**:
- Analytical solution exists
- 2 bodies, low dimensionality
- Predictable elliptical orbits

### Simulator Implementation
```python
class TwoBodySystem:
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        a = np.random.uniform(1.0, 10.0, n_samples)  # AU
        e = np.random.uniform(0, 0.9, n_samples)
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        
        # Truth: r(theta) = a(1-e^2) / (1 + e*cos(theta))
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        
        X = np.column_stack((a, e, theta))
        return X, r
```

### Expected Results
- **R²**: > 0.99 (high accuracy)
- **Cage Status**: 🔒 **LOCKED** (correlation with a, e > 0.9)
- **Reason**: Low dimensionality, analytical solution, evolution prepared us

---

## Part B: N-Body System (Complex)

### Physics
**N-Body Gravitational System** (N=5-10 bodies):

Equations of motion:
$$\ddot{\vec{r}}_i = G \sum_{j \neq i} \frac{m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}$$

Where:
- $G = 1$ (normalized)
- $m_i$: Mass of body i [0.1, 1.0] (normalized)
- $\vec{r}_i$: Position vector of body i
- $\vec{v}_i$: Velocity vector of body i

**Input**: Initial conditions for all N bodies
- Positions: $\vec{r}_i \in [-10, 10]$ (3D)
- Velocities: $\vec{v}_i \in [-1, 1]$ (3D)
- Masses: $m_i \in [0.1, 1.0]$
- Time: $t \in [0, 10]$ (normalized time units)

**Output**: Total energy of the system at time t
$$E(t) = \sum_i \frac{1}{2} m_i |\vec{v}_i|^2 - G \sum_{i<j} \frac{m_i m_j}{|\vec{r}_i - \vec{r}_j|}$$

**Key Complexity**:
- No analytical solution (N > 2)
- High dimensionality (3N positions + 3N velocities = 6N dimensions)
- Chaotic behavior for N ≥ 3
- Emergent properties (energy, not individual positions)

### Simulator Implementation
```python
class NBodySystem:
    def __init__(self, N=5):
        self.N = N  # Number of bodies
        self.G = 1.0  # Gravitational constant (normalized)
    
    def _nbody_ode(self, t, state):
        # state: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, ...]
        # Reshape to N bodies, each with 6 DOF
        positions = state[:3*self.N].reshape(self.N, 3)
        velocities = state[3*self.N:].reshape(self.N, 3)
        masses = self.masses
        
        # Calculate accelerations
        accelerations = np.zeros((self.N, 3))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 1e-6:  # Avoid division by zero
                        accelerations[i] += self.G * masses[j] * r_vec / (r_mag**3)
        
        # Return derivatives: [velocities, accelerations]
        return np.concatenate([velocities.flatten(), accelerations.flatten()])
    
    def generate_dataset(self, n_samples=2000):
        from scipy.integrate import solve_ivp
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Random initial conditions
            positions = np.random.uniform(-10, 10, (self.N, 3))
            velocities = np.random.uniform(-1, 1, (self.N, 3))
            masses = np.random.uniform(0.1, 1.0, self.N)
            t_eval = np.random.uniform(0, 10)
            
            self.masses = masses
            
            # Initial state: [positions, velocities]
            initial_state = np.concatenate([positions.flatten(), velocities.flatten()])
            
            # Integrate N-body system
            sol = solve_ivp(
                self._nbody_ode,
                [0, t_eval],
                initial_state,
                t_eval=[t_eval],
                dense_output=True,
                rtol=1e-6
            )
            
            # Extract final state
            final_state = sol.y[:, -1]
            final_positions = final_state[:3*self.N].reshape(self.N, 3)
            final_velocities = final_state[3*self.N:].reshape(self.N, 3)
            
            # Calculate total energy
            kinetic = 0.5 * np.sum(masses * np.sum(final_velocities**2, axis=1))
            potential = 0
            for i in range(self.N):
                for j in range(i+1, self.N):
                    r = np.linalg.norm(final_positions[j] - final_positions[i])
                    if r > 1e-6:
                        potential -= self.G * masses[i] * masses[j] / r
            total_energy = kinetic + potential
            
            # Input: flattened initial conditions + time
            X.append(np.concatenate([positions.flatten(), velocities.flatten(), masses, [t_eval]]))
            y.append(total_energy)
        
        return np.array(X), np.array(y)
```

### Expected Results
- **R²**: > 0.85 (moderate accuracy, chaos makes it hard)
- **Cage Status**: 🔓 **BROKEN** (correlation with individual positions/velocities < 0.3)
- **Reason**: High dimensionality, emergent properties, evolution didn't prepare us

---

## Methodology

### 1. Data Generation
- **Part A**: 2000 samples, 2-body system
- **Part B**: 2000 samples, N-body system (N=5)
- Same random seed for reproducibility

### 2. Models
- **Baseline**: Polynomial Regression (degree 4)
- **Chaos Model**: Optical Chaos (4096 features, brightness=0.001)

### 3. Evaluation
- **Standard R²**: Random train/test split (80/20)
- **Cage Analysis**: 
  - Part A: Correlate features with a, e
  - Part B: Correlate features with individual positions/velocities vs. total energy
- **Scalability Test**: Test with N=3, 5, 7, 10 bodies

### 4. Success Criteria
- **Hypothesis confirmed if**:
  - Part A: Cage LOCKED (correlation > 0.9)
  - Part B: Cage BROKEN (correlation < 0.3)
  - Both achieve reasonable R²

---

## Implementation Checklist

- [ ] Implement `TwoBodySystem` simulator
- [ ] Implement `NBodySystem` simulator (using scipy.integrate)
- [ ] Create main experiment script with both parts
- [ ] Train baseline and chaos models on both parts
- [ ] Calculate R² scores
- [ ] Perform cage analysis
- [ ] Test scalability (N=3, 5, 7, 10)
- [ ] Create visualizations (orbits for 2-body, energy evolution for N-body)
- [ ] Write benchmark script
- [ ] Document results in README

---

## Files Structure

```
experiment_10_low_vs_high_dim/
├── experiment_10_low_vs_high_dim.py
├── benchmark_experiment_10.py
└── README.md
```

## Notes

- N-body system requires numerical integration (scipy.integrate.solve_ivp)
- High dimensionality: N=5 → 6*5 + 5 + 1 = 36 input dimensions
- Energy conservation can be used to validate simulator
- May need to limit N to keep computation tractable

