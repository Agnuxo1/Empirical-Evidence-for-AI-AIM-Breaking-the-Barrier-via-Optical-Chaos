"""
Physics vs. Darwin: Experiment 10
Low vs High Dimensionality
Testing Complexity Hypothesis: Few-Body vs Many-Body Systems
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
- Simple Physics: 2-body gravitational system (Kepler orbits, analytical solution)
- Complex Physics: N-body gravitational system (N=5-10, no analytical solution, chaotic)

Hypothesis: Many-body system should break the cage, 2-body system should lock it.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.integrate import solve_ivp

# --- 1. PHYSICS SIMULATORS ---

class TwoBodySystem:
    """
    Part A: Simple Physics
    2-body gravitational system: r(theta) = a(1-e^2) / (1 + e*cos(theta))
    """
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        a = np.random.uniform(1.0, 10.0, n_samples)  # Semi-major axis [AU]
        e = np.random.uniform(0, 0.9, n_samples)  # Eccentricity
        theta = np.random.uniform(0, 2*np.pi, n_samples)  # True anomaly [rad]
        
        # Truth: r(theta) = a(1-e^2) / (1 + e*cos(theta))
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        
        X = np.column_stack((a, e, theta))
        return X, r

class NBodySystem:
    """
    Part B: Complex Physics
    N-body gravitational system (N=5 bodies)
    """
    def __init__(self, N=5):
        self.N = N  # Number of bodies
        self.G = 1.0  # Gravitational constant (normalized)
    
    def _nbody_ode(self, t, state):
        """
        N-body ODE: d²r/dt² = G * sum(m_j * (r_j - r_i) / |r_j - r_i|³)
        state: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, ...]
        """
        # Reshape to N bodies, each with 6 DOF (3 position + 3 velocity)
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
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Random initial conditions
            positions = np.random.uniform(-10, 10, (self.N, 3))
            velocities = np.random.uniform(-1, 1, (self.N, 3))
            masses = np.random.uniform(0.1, 1.0, self.N)
            t_eval = np.random.uniform(0, 10)  # Time to evaluate
            
            self.masses = masses
            
            # Initial state: [positions, velocities]
            initial_state = np.concatenate([positions.flatten(), velocities.flatten()])
            
            # Integrate N-body system
            try:
                sol = solve_ivp(
                    self._nbody_ode,
                    [0, t_eval],
                    initial_state,
                    t_eval=[t_eval],
                    dense_output=True,
                    rtol=1e-6,
                    atol=1e-9
                )
                
                # Extract final state
                final_state = sol.y[:, -1]
                final_positions = final_state[:3*self.N].reshape(self.N, 3)
                final_velocities = final_state[3*self.N:].reshape(self.N, 3)
                
                # Calculate total energy: E = T + V
                # Kinetic energy
                kinetic = 0.5 * np.sum(masses * np.sum(final_velocities**2, axis=1))
                # Potential energy
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
            # Check if either variable has zero variance (would cause division by zero)
            if np.std(internal_states[:, j]) < 1e-10 or np.std(var_values) < 1e-10:
                continue  # Skip if variance is too small
            corr = np.corrcoef(internal_states[:, j], var_values)[0, 1]
            if not np.isnan(corr) and not np.isinf(corr):
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
    print("🌌 STARTING EXPERIMENT 10: LOW vs HIGH DIMENSIONALITY")
    print("=" * 70)
    print("Testing Complexity Hypothesis: Few-Body vs Many-Body Systems")
    print("=" * 70)
    
    results = {}
    
    # ===== PART A: 2-BODY SYSTEM (SIMPLE) =====
    print("\n[PART A] 2-BODY SYSTEM (Simple Physics)")
    print("-" * 70)
    
    # Generate data
    print("  Generating dataset...")
    two_body_sim = TwoBodySystem()
    X_two, y_two = two_body_sim.generate_dataset(n_samples=2000)
    
    # Scale data
    scaler_two = MinMaxScaler()
    X_two_scaled = scaler_two.fit_transform(X_two)
    
    # Split data
    X_train_2, X_test_2, X_train_2_s, X_test_2_s, y_train_2, y_test_2 = train_test_split(
        X_two, X_two_scaled, y_two, test_size=0.2, random_state=42
    )
    
    print(f"  Generated {len(X_two)} samples")
    print(f"  Input shape: {X_two.shape}")
    print(f"  Output range: [{np.min(y_two):.4f}, {np.max(y_two):.4f}]")
    
    # Train models
    print("  Training models...")
    
    # Darwinian baseline
    darwin_two = DarwinianModel()
    darwin_two.fit(X_train_2, y_train_2)
    y_pred_darwin_2 = darwin_two.predict(X_test_2)
    r2_darwin_2 = r2_score(y_test_2, y_pred_darwin_2)
    
    # Chaos model - optimize brightness
    best_r2_2 = -np.inf
    best_chaos_2 = None
    best_brightness_2 = 0.001
    
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_2_s, y_train_2)
        y_pred_test = chaos_test.predict(X_test_2_s)
        r2_test = r2_score(y_test_2, y_pred_test)
        if r2_test > best_r2_2:
            best_r2_2 = r2_test
            best_chaos_2 = chaos_test
            best_brightness_2 = brightness
    
    chaos_two = best_chaos_2
    y_pred_chaos_2 = chaos_two.predict(X_test_2_s)
    r2_chaos_2 = best_r2_2
    
    print(f"  Darwinian R²: {r2_darwin_2:.4f}")
    print(f"  Chaos R²: {r2_chaos_2:.4f} (brightness={best_brightness_2})")
    
    # Cage analysis
    print("  Analyzing cage status...")
    cage_two, mean_cage_two = analyze_cage(
        chaos_two, X_test_2_s, X_test_2,
        ['Semi-major_a', 'Eccentricity_e', 'True_anomaly_theta']
    )
    
    max_corr_two = max(cage_two.values())
    mean_corr_two = np.mean(list(mean_cage_two.values()))
    print(f"  Max correlation with human variables: {max_corr_two:.4f}")
    print(f"  Mean correlation with human variables: {mean_corr_two:.4f}")
    for var, corr in cage_two.items():
        print(f"    - {var}: max={corr:.4f}, mean={mean_cage_two[var]:.4f}")
    
    if max_corr_two > 0.9:
        cage_status_2 = "LOCKED"
    elif max_corr_two < 0.3:
        cage_status_2 = "BROKEN"
    else:
        cage_status_2 = "UNCLEAR"
    
    print(f"  Cage Status: {cage_status_2}")
    
    results['two_body'] = {
        'r2_darwin': r2_darwin_2,
        'r2_chaos': r2_chaos_2,
        'cage_correlations': cage_two,
        'cage_mean_correlations': mean_cage_two,
        'cage_status': cage_status_2,
        'max_correlation': max_corr_two,
        'mean_correlation': mean_corr_two
    }
    
    # ===== PART B: N-BODY SYSTEM (COMPLEX) =====
    print("\n[PART B] N-BODY SYSTEM (Complex Physics - N=5)")
    print("-" * 70)
    
    # Generate data
    print("  Generating dataset...")
    n_body_sim = NBodySystem(N=5)
    X_nbody, y_nbody = n_body_sim.generate_dataset(n_samples=2000)
    
    print(f"  Generated {len(X_nbody)} samples (some may have failed integration)")
    print(f"  Input shape: {X_nbody.shape} (high dimensionality!)")
    print(f"  Output range: [{np.min(y_nbody):.4f}, {np.max(y_nbody):.4f}]")
    
    # Scale data
    scaler_nbody = MinMaxScaler()
    X_nbody_scaled = scaler_nbody.fit_transform(X_nbody)
    
    # Split data
    X_train_n, X_test_n, X_train_n_s, X_test_n_s, y_train_n, y_test_n = train_test_split(
        X_nbody, X_nbody_scaled, y_nbody, test_size=0.2, random_state=42
    )
    
    # Train models
    print("  Training models...")
    
    # Darwinian baseline
    darwin_nbody = DarwinianModel()
    darwin_nbody.fit(X_train_n, y_train_n)
    y_pred_darwin_n = darwin_nbody.predict(X_test_n)
    r2_darwin_n = r2_score(y_test_n, y_pred_darwin_n)
    
    # Chaos model - optimize brightness
    best_r2_n = -np.inf
    best_chaos_n = None
    best_brightness_n = 0.001
    
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_n_s, y_train_n)
        y_pred_test = chaos_test.predict(X_test_n_s)
        r2_test = r2_score(y_test_n, y_pred_test)
        if r2_test > best_r2_n:
            best_r2_n = r2_test
            best_chaos_n = chaos_test
            best_brightness_n = brightness
    
    chaos_nbody = best_chaos_n
    y_pred_chaos_n = chaos_nbody.predict(X_test_n_s)
    r2_chaos_n = best_r2_n
    
    print(f"  Darwinian R²: {r2_darwin_n:.4f}")
    print(f"  Chaos R²: {r2_chaos_n:.4f} (brightness={best_brightness_n})")
    
    # Cage analysis - for N-body, we analyze correlation with individual positions/velocities
    # Since we have 5 bodies with 3 positions + 3 velocities each = 30 variables, plus 5 masses + 1 time = 36 total
    # IMPORTANT: Analyze ALL variables for unbiased cage analysis
    print("  Analyzing cage status...")
    n_vars = X_test_n.shape[1]
    var_names = []
    
    # Create meaningful names for all variables
    for i in range(n_vars):
        if i < 15:  # Positions (5 bodies × 3 coords)
            body_idx = i // 3
            coord_idx = i % 3
            var_names.append(f'Body{body_idx+1}_pos_{["x","y","z"][coord_idx]}')
        elif i < 30:  # Velocities (5 bodies × 3 coords)
            body_idx = (i - 15) // 3
            coord_idx = (i - 15) % 3
            var_names.append(f'Body{body_idx+1}_vel_{["x","y","z"][coord_idx]}')
        elif i < 35:  # Masses (5 bodies)
            body_idx = i - 30
            var_names.append(f'Mass{body_idx+1}')
        else:  # Time
            var_names.append('Time')
    
    cage_nbody, mean_cage_nbody = analyze_cage(
        chaos_nbody, X_test_n_s, X_test_n,
        var_names
    )
    
    max_corr_nbody = max(cage_nbody.values())
    mean_corr_nbody = np.mean(list(mean_cage_nbody.values()))
    print(f"  Max correlation with human variables: {max_corr_nbody:.4f}")
    print(f"  Mean correlation with human variables: {mean_corr_nbody:.4f}")
    
    # Show top 5 and bottom 5 correlations
    sorted_corrs = sorted(cage_nbody.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5 correlations:")
    for var, corr in sorted_corrs[:5]:
        print(f"    - {var}: max={corr:.4f}, mean={mean_cage_nbody[var]:.4f}")
    if len(cage_nbody) > 5:
        print(f"  Bottom 5 correlations:")
        for var, corr in sorted_corrs[-5:]:
            print(f"    - {var}: max={corr:.4f}, mean={mean_cage_nbody[var]:.4f}")
        print(f"  ... analyzed {len(cage_nbody)} total variables")
    
    if max_corr_nbody > 0.9:
        cage_status_n = "LOCKED"
    elif max_corr_nbody < 0.3:
        cage_status_n = "BROKEN"
    else:
        cage_status_n = "UNCLEAR"
    
    print(f"  Cage Status: {cage_status_n}")
    
    results['n_body'] = {
        'r2_darwin': r2_darwin_n,
        'r2_chaos': r2_chaos_n,
        'cage_correlations': cage_nbody,
        'cage_mean_correlations': mean_cage_nbody,
        'cage_status': cage_status_n,
        'max_correlation': max_corr_nbody,
        'mean_correlation': mean_corr_nbody
    }
    
    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("COMPARISON: Low vs High Dimensionality")
    print("=" * 70)
    print(f"\n2-Body (Simple):")
    print(f"  Input dim: {X_two.shape[1]}")
    print(f"  R² Chaos: {r2_chaos_2:.4f}")
    print(f"  Max Correlation: {max_corr_two:.4f}")
    print(f"  Mean Correlation: {mean_corr_two:.4f}")
    print(f"  Cage Status: {cage_status_2}")
    print(f"\nN-Body (Complex):")
    print(f"  Input dim: {X_nbody.shape[1]} (N=5 bodies)")
    print(f"  R² Chaos: {r2_chaos_n:.4f}")
    print(f"  Max Correlation: {max_corr_nbody:.4f}")
    print(f"  Mean Correlation: {mean_corr_nbody:.4f}")
    print(f"  Cage Status: {cage_status_n}")
    
    # Hypothesis test
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)
    if max_corr_two > 0.9 and max_corr_nbody < 0.3:
        print("✅ HYPOTHESIS CONFIRMED: Low-dim locks cage, high-dim breaks it")
    elif max_corr_two < 0.3 and max_corr_nbody > 0.9:
        print("❌ HYPOTHESIS REFUTED: Opposite pattern observed")
    else:
        print("⚠️ HYPOTHESIS UNCLEAR: Mixed or unclear results")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2-Body: Predictions
    axes[0, 0].scatter(y_test_2, y_pred_chaos_2, alpha=0.5, s=20)
    axes[0, 0].plot([y_test_2.min(), y_test_2.max()], [y_test_2.min(), y_test_2.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Radial Distance r(θ)')
    axes[0, 0].set_ylabel('Predicted Distance')
    axes[0, 0].set_title(f'2-Body: Chaos Model (R² = {r2_chaos_2:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2-Body: Cage correlations
    vars_2 = list(cage_two.keys())
    corrs_2 = list(cage_two.values())
    axes[0, 1].bar(vars_2, corrs_2, color='blue', alpha=0.7)
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Locked threshold')
    axes[0, 1].axhline(y=0.3, color='g', linestyle='--', label='Broken threshold')
    axes[0, 1].set_ylabel('Max Correlation')
    axes[0, 1].set_title(f'2-Body: Cage Analysis ({cage_status_2})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # N-Body: Predictions
    axes[1, 0].scatter(y_test_n, y_pred_chaos_n, alpha=0.5, s=20)
    axes[1, 0].plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Total Energy E(t)')
    axes[1, 0].set_ylabel('Predicted Energy')
    axes[1, 0].set_title(f'N-Body (N=5): Chaos Model (R² = {r2_chaos_n:.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # N-Body: Cage correlations (show distribution histogram)
    corrs_n = list(cage_nbody.values())
    axes[1, 1].hist(corrs_n, bins=20, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0.9, color='r', linestyle='--', label='Locked threshold')
    axes[1, 1].axvline(x=0.3, color='g', linestyle='--', label='Broken threshold')
    axes[1, 1].set_xlabel('Max Correlation')
    axes[1, 1].set_ylabel('Number of Variables')
    axes[1, 1].set_title(f'N-Body: Cage Analysis ({cage_status_n})\nDistribution of {len(corrs_n)} variables')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_10_low_vs_high_dim.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graph saved as 'experiment_10_low_vs_high_dim.png'")
    plt.show()
    
    return results

if __name__ == "__main__":
    results = run_experiment()

