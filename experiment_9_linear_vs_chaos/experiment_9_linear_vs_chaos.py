"""
Physics vs. Darwin: Experiment 9
Linear vs Nonlinear (Chaos)
Testing Complexity Hypothesis: Predictable vs Chaotic Systems
------------------------------------------------------------

CREDITS AND REFERENCES:
-----------------------
Darwin's Cage Theory:
- Theory Creator: Gideon Samid
- Reference: Samid, G. (2025). Negotiating Darwin's Barrier: Evolution Limits Our View of Reality, AI Breaks Through. Applied Physics Research, 17(2), 102. https://doi.org/10.5539/apr.v17n2p102
- Publication: Applied Physics Research; Vol. 17, No. 2; 2025. ISSN 1916-9639 E-ISSN 1916-9647. Published by Canadian Center of Science and Education
- Available at: https://www.researchgate.net/publication/396377476_Negotiating_Darwin's_Barrier_Evolution_Limits_Our_View_of_Reality_AI_Breaks_Through

Experiments, AI Models, Architectures, and Reports:
- Author: Francisco Angulo de Lafuente
- Responsibilities: Experimental design, AI model creation, architecture development, results analysis, and report writing

Objective:
Compare cage status between:
- Simple Physics: Linear RLC circuit (predictable, analytical solution)
- Complex Physics: Lorenz attractor (chaotic, sensitive to initial conditions)

Hypothesis: Chaotic system should break the cage, linear system should lock it.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.integrate import solve_ivp

# --- 1. PHYSICS SIMULATORS ---

class LinearRLCCircuit:
    """
    Part A: Simple Physics
    Linear RLC circuit: Q(t) = Q0 * exp(-gamma*t) * cos(omega_d*t + phi)
    """
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        Q0 = np.random.uniform(0.1, 10.0, n_samples)  # Initial charge [C]
        gamma = np.random.uniform(0.1, 2.0, n_samples)  # Damping coefficient [s⁻¹]
        omega_d = np.random.uniform(0.5, 5.0, n_samples)  # Damped frequency [rad/s]
        phi = np.random.uniform(0, 2*np.pi, n_samples)  # Phase [rad]
        t = np.random.uniform(0, 10.0, n_samples)  # Time [s]
        
        # Truth: Q(t) = Q0 * exp(-gamma*t) * cos(omega_d*t + phi)
        Q = Q0 * np.exp(-gamma * t) * np.cos(omega_d * t + phi)
        
        X = np.column_stack((Q0, gamma, omega_d, phi, t))
        return X, Q

class LorenzAttractor:
    """
    Part B: Complex Physics
    Lorenz attractor (chaotic differential equations)
    """
    def __init__(self):
        self.sigma = 10.0  # Prandtl number
        self.rho = 28.0    # Rayleigh number
        self.beta = 8.0 / 3.0  # Geometric factor
    
    def _lorenz_ode(self, t, state):
        """Lorenz system ODE: dx/dt, dy/dt, dz/dt"""
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]
    
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Random initial conditions
            x0 = np.random.uniform(-20, 20)
            y0 = np.random.uniform(-20, 20)
            z0 = np.random.uniform(0, 50)
            t_eval = np.random.uniform(0, 20)  # Time to evaluate
            
            # Initial state
            initial_state = [x0, y0, z0]
            
            # Integrate Lorenz system
            try:
                sol = solve_ivp(
                    self._lorenz_ode,
                    [0, t_eval],
                    initial_state,
                    t_eval=[t_eval],
                    dense_output=True,
                    rtol=1e-6,
                    atol=1e-9
                )
                
                # Output: x(t) coordinate
                x_final = sol.y[0][0]
                
                X.append([x0, y0, z0, t_eval])
                y.append(x_final)
            except Exception as e:
                # If integration fails, skip this sample
                continue
        
        return np.array(X), np.array(y)

# --- 2. OPTICAL CHAOS MODEL ---

class OpticalChaosMachine:
    def __init__(self, n_features=4096, brightness=0.001):
        self.n_features = n_features
        self.brightness = brightness
        self.readout = Ridge(alpha=0.1)
        self.optical_matrix = None
        
    def _optical_interference(self, X):
        n_samples, n_input = X.shape
        
        if self.optical_matrix is None:
            np.random.seed(1337)
            self.optical_matrix = np.random.normal(0, 1, (n_input, self.n_features))
        
        light_field = X @ self.optical_matrix
        interference_pattern = np.fft.rfft(light_field, axis=1)
        intensity = np.abs(interference_pattern)**2
        intensity = np.tanh(intensity * self.brightness)
        
        return intensity
    
    def fit(self, X, y):
        X_optical = self._optical_interference(X)
        self.readout.fit(X_optical, y)
    
    def predict(self, X):
        X_optical = self._optical_interference(X)
        return self.readout.predict(X_optical)
    
    def get_internal_state(self, X):
        return self._optical_interference(X)

# --- 3. DARWINIAN BASELINE ---

class DarwinianModel:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=4)
        self.model = Ridge(alpha=0.1)
    
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

# --- 4. CAGE ANALYSIS ---

def analyze_cage(model, X_test, X_test_raw, variable_names):
    """
    Analyze if internal features correlate with human variables.
    Returns max and mean correlation for each variable.
    """
    internal_states = model.get_internal_state(X_test)
    n_features = internal_states.shape[1]
    
    max_correlations = {}
    mean_correlations = {}
    
    for i, var_name in enumerate(variable_names):
        var_values = X_test_raw[:, i]
        correlations = []
        
        # Check correlation with ALL internal features
        for j in range(n_features):
            corr = np.corrcoef(internal_states[:, j], var_values)[0, 1]
            if not np.isnan(corr):
                correlations.append(np.abs(corr))
        
        if correlations:
            max_correlations[var_name] = np.max(correlations)
            mean_correlations[var_name] = np.mean(correlations)
        else:
            max_correlations[var_name] = 0.0
            mean_correlations[var_name] = 0.0
    
    return max_correlations, mean_correlations

# --- 5. MAIN EXPERIMENT ---

def run_experiment():
    print("🌀 STARTING EXPERIMENT 9: LINEAR vs CHAOS")
    print("=" * 70)
    print("Testing Complexity Hypothesis: Predictable vs Chaotic Systems")
    print("=" * 70)
    
    results = {}
    
    # ===== PART A: LINEAR RLC CIRCUIT (SIMPLE) =====
    print("\n[PART A] LINEAR RLC CIRCUIT (Simple Physics)")
    print("-" * 70)
    
    # Generate data
    print("  Generating dataset...")
    linear_sim = LinearRLCCircuit()
    X_linear, y_linear = linear_sim.generate_dataset(n_samples=2000)
    
    # Scale data
    scaler_linear = MinMaxScaler()
    X_linear_scaled = scaler_linear.fit_transform(X_linear)
    
    # Split data
    X_train_l, X_test_l, X_train_l_s, X_test_l_s, y_train_l, y_test_l = train_test_split(
        X_linear, X_linear_scaled, y_linear, test_size=0.2, random_state=42
    )
    
    print(f"  Generated {len(X_linear)} samples")
    print(f"  Input shape: {X_linear.shape}")
    print(f"  Output range: [{np.min(y_linear):.4f}, {np.max(y_linear):.4f}]")
    
    # Train models
    print("  Training models...")
    
    # Darwinian baseline
    darwin_linear = DarwinianModel()
    darwin_linear.fit(X_train_l, y_train_l)
    y_pred_darwin_l = darwin_linear.predict(X_test_l)
    r2_darwin_l = r2_score(y_test_l, y_pred_darwin_l)
    
    # Chaos model - optimize brightness
    best_r2_l = -np.inf
    best_chaos_l = None
    best_brightness_l = 0.001
    
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_l_s, y_train_l)
        y_pred_test = chaos_test.predict(X_test_l_s)
        r2_test = r2_score(y_test_l, y_pred_test)
        if r2_test > best_r2_l:
            best_r2_l = r2_test
            best_chaos_l = chaos_test
            best_brightness_l = brightness
    
    chaos_linear = best_chaos_l
    y_pred_chaos_l = chaos_linear.predict(X_test_l_s)
    r2_chaos_l = best_r2_l
    
    print(f"  Darwinian R²: {r2_darwin_l:.4f}")
    print(f"  Chaos R²: {r2_chaos_l:.4f} (brightness={best_brightness_l})")
    
    # Cage analysis
    print("  Analyzing cage status...")
    cage_linear, mean_cage_linear = analyze_cage(
        chaos_linear, X_test_l_s, X_test_l,
        ['Q0', 'Gamma', 'Omega_d', 'Phase', 'Time']
    )
    
    max_corr_linear = max(cage_linear.values())
    mean_corr_linear = np.mean(list(mean_cage_linear.values()))
    print(f"  Max correlation with human variables: {max_corr_linear:.4f}")
    print(f"  Mean correlation with human variables: {mean_corr_linear:.4f}")
    for var, corr in cage_linear.items():
        print(f"    - {var}: max={corr:.4f}, mean={mean_cage_linear[var]:.4f}")
    
    if max_corr_linear > 0.9:
        cage_status_l = "LOCKED"
    elif max_corr_linear < 0.3:
        cage_status_l = "BROKEN"
    else:
        cage_status_l = "UNCLEAR"
    
    print(f"  Cage Status: {cage_status_l}")
    
    results['linear'] = {
        'r2_darwin': r2_darwin_l,
        'r2_chaos': r2_chaos_l,
        'cage_correlations': cage_linear,
        'cage_mean_correlations': mean_cage_linear,
        'cage_status': cage_status_l,
        'max_correlation': max_corr_linear,
        'mean_correlation': mean_corr_linear
    }
    
    # ===== PART B: LORENZ ATTRACTOR (COMPLEX) =====
    print("\n[PART B] LORENZ ATTRACTOR (Complex Physics - Chaos)")
    print("-" * 70)
    
    # Generate data
    print("  Generating dataset...")
    lorenz_sim = LorenzAttractor()
    X_lorenz, y_lorenz = lorenz_sim.generate_dataset(n_samples=2000)
    
    print(f"  Generated {len(X_lorenz)} samples (some may have failed integration)")
    print(f"  Input shape: {X_lorenz.shape}")
    print(f"  Output range: [{np.min(y_lorenz):.4f}, {np.max(y_lorenz):.4f}]")
    
    # Scale data
    scaler_lorenz = MinMaxScaler()
    X_lorenz_scaled = scaler_lorenz.fit_transform(X_lorenz)
    
    # Split data
    X_train_lo, X_test_lo, X_train_lo_s, X_test_lo_s, y_train_lo, y_test_lo = train_test_split(
        X_lorenz, X_lorenz_scaled, y_lorenz, test_size=0.2, random_state=42
    )
    
    # Train models
    print("  Training models...")
    
    # Darwinian baseline
    darwin_lorenz = DarwinianModel()
    darwin_lorenz.fit(X_train_lo, y_train_lo)
    y_pred_darwin_lo = darwin_lorenz.predict(X_test_lo)
    r2_darwin_lo = r2_score(y_test_lo, y_pred_darwin_lo)
    
    # Chaos model - optimize brightness
    best_r2_lo = -np.inf
    best_chaos_lo = None
    best_brightness_lo = 0.001
    
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_lo_s, y_train_lo)
        y_pred_test = chaos_test.predict(X_test_lo_s)
        r2_test = r2_score(y_test_lo, y_pred_test)
        if r2_test > best_r2_lo:
            best_r2_lo = r2_test
            best_chaos_lo = chaos_test
            best_brightness_lo = brightness
    
    chaos_lorenz = best_chaos_lo
    y_pred_chaos_lo = chaos_lorenz.predict(X_test_lo_s)
    r2_chaos_lo = best_r2_lo
    
    print(f"  Darwinian R²: {r2_darwin_lo:.4f}")
    print(f"  Chaos R²: {r2_chaos_lo:.4f} (brightness={best_brightness_lo})")
    
    # Cage analysis
    print("  Analyzing cage status...")
    cage_lorenz, mean_cage_lorenz = analyze_cage(
        chaos_lorenz, X_test_lo_s, X_test_lo,
        ['x0', 'y0', 'z0', 'Time']
    )
    
    max_corr_lorenz = max(cage_lorenz.values())
    mean_corr_lorenz = np.mean(list(mean_cage_lorenz.values()))
    print(f"  Max correlation with human variables: {max_corr_lorenz:.4f}")
    print(f"  Mean correlation with human variables: {mean_corr_lorenz:.4f}")
    for var, corr in cage_lorenz.items():
        print(f"    - {var}: max={corr:.4f}, mean={mean_cage_lorenz[var]:.4f}")
    
    if max_corr_lorenz > 0.9:
        cage_status_lo = "LOCKED"
    elif max_corr_lorenz < 0.3:
        cage_status_lo = "BROKEN"
    else:
        cage_status_lo = "UNCLEAR"
    
    print(f"  Cage Status: {cage_status_lo}")
    
    results['lorenz'] = {
        'r2_darwin': r2_darwin_lo,
        'r2_chaos': r2_chaos_lo,
        'cage_correlations': cage_lorenz,
        'cage_mean_correlations': mean_cage_lorenz,
        'cage_status': cage_status_lo,
        'max_correlation': max_corr_lorenz,
        'mean_correlation': mean_corr_lorenz
    }
    
    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("COMPARISON: Linear vs Chaotic Systems")
    print("=" * 70)
    print(f"\nLinear RLC (Simple):")
    print(f"  R² Chaos: {r2_chaos_l:.4f}")
    print(f"  Max Correlation: {max_corr_linear:.4f}")
    print(f"  Mean Correlation: {mean_corr_linear:.4f}")
    print(f"  Cage Status: {cage_status_l}")
    print(f"\nLorenz (Complex - Chaos):")
    print(f"  R² Chaos: {r2_chaos_lo:.4f}")
    print(f"  Max Correlation: {max_corr_lorenz:.4f}")
    print(f"  Mean Correlation: {mean_corr_lorenz:.4f}")
    print(f"  Cage Status: {cage_status_lo}")
    
    # Hypothesis test
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)
    if max_corr_linear > 0.9 and max_corr_lorenz < 0.3:
        print("✅ HYPOTHESIS CONFIRMED: Linear system locks cage, chaotic breaks it")
    elif max_corr_linear < 0.3 and max_corr_lorenz > 0.9:
        print("❌ HYPOTHESIS REFUTED: Opposite pattern observed")
    else:
        print("⚠️ HYPOTHESIS UNCLEAR: Mixed or unclear results")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear: Predictions
    axes[0, 0].scatter(y_test_l, y_pred_chaos_l, alpha=0.5, s=20)
    axes[0, 0].plot([y_test_l.min(), y_test_l.max()], [y_test_l.min(), y_test_l.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Charge Q(t)')
    axes[0, 0].set_ylabel('Predicted Charge')
    axes[0, 0].set_title(f'Linear RLC: Chaos Model (R² = {r2_chaos_l:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Linear: Cage correlations
    vars_l = list(cage_linear.keys())
    corrs_l = list(cage_linear.values())
    axes[0, 1].bar(vars_l, corrs_l, color='blue', alpha=0.7)
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Locked threshold')
    axes[0, 1].axhline(y=0.3, color='g', linestyle='--', label='Broken threshold')
    axes[0, 1].set_ylabel('Max Correlation')
    axes[0, 1].set_title(f'Linear RLC: Cage Analysis ({cage_status_l})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Lorenz: Predictions
    axes[1, 0].scatter(y_test_lo, y_pred_chaos_lo, alpha=0.5, s=20)
    axes[1, 0].plot([y_test_lo.min(), y_test_lo.max()], [y_test_lo.min(), y_test_lo.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True x(t) Coordinate')
    axes[1, 0].set_ylabel('Predicted x(t)')
    axes[1, 0].set_title(f'Lorenz: Chaos Model (R² = {r2_chaos_lo:.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Lorenz: Cage correlations
    vars_lo = list(cage_lorenz.keys())
    corrs_lo = list(cage_lorenz.values())
    axes[1, 1].bar(vars_lo, corrs_lo, color='red', alpha=0.7)
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='Locked threshold')
    axes[1, 1].axhline(y=0.3, color='g', linestyle='--', label='Broken threshold')
    axes[1, 1].set_ylabel('Max Correlation')
    axes[1, 1].set_title(f'Lorenz: Cage Analysis ({cage_status_lo})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('experiment_9_linear_vs_chaos.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graph saved as 'experiment_9_linear_vs_chaos.png'")
    plt.show()
    
    return results

if __name__ == "__main__":
    results = run_experiment()

