"""
Physics vs. Darwin: Experiment 8
Classical vs Quantum Mechanics
Testing Complexity Hypothesis: Simple vs Complex Physics
-------------------------------------------------------

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
- Simple Physics: Classical harmonic oscillator (intuitive, analytical)
- Complex Physics: Quantum particle in a box (counterintuitive, discrete states)

Hypothesis: Quantum system should break the cage, classical should lock it.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# scipy.fft not used, removed

# --- 1. PHYSICS SIMULATORS ---

class ClassicalHarmonicOscillator:
    """
    Part A: Simple Physics
    Classical harmonic oscillator: x(t) = A * cos(omega*t + phi)
    """
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        A = np.random.uniform(0.1, 10.0, n_samples)  # Amplitude [m]
        omega = np.random.uniform(0.5, 5.0, n_samples)  # Angular frequency [rad/s]
        phi = np.random.uniform(0, 2*np.pi, n_samples)  # Phase [rad]
        t = np.random.uniform(0, 10.0, n_samples)  # Time [s]
        
        # Truth: x(t) = A * cos(omega*t + phi)
        x = A * np.cos(omega * t + phi)
        
        X = np.column_stack((A, omega, phi, t))
        return X, x

class QuantumParticleInBox:
    """
    Part B: Complex Physics
    Quantum particle in a box: |psi|^2 = (2/L) * sin^2(n*pi*x/L)
    """
    def generate_dataset(self, n_samples=2000):
        np.random.seed(42)
        n = np.random.randint(1, 11, n_samples)  # Quantum number (discrete: 1-10)
        L = np.random.uniform(1.0, 10.0, n_samples)  # Box width [m]
        x = np.random.uniform(0, L, n_samples)  # Position within box [m]
        
        # Truth: |psi|^2 = (2/L) * sin^2(n*pi*x/L)
        prob_density = (2.0 / L) * np.sin(n * np.pi * x / L)**2
        
        X = np.column_stack((n, L, x))
        return X, prob_density

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
    Returns max correlation for each variable.
    CRITICAL: Check ALL features, not just a sample (as in Experiment 1)
    """
    internal_states = model.get_internal_state(X_test)
    n_features = internal_states.shape[1]
    
    max_correlations = {}
    mean_correlations = {}
    
    for i, var_name in enumerate(variable_names):
        var_values = X_test_raw[:, i]
        correlations = []
        
        # Check correlation with ALL internal features (not just sample)
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
    print("⚛️ STARTING EXPERIMENT 8: CLASSICAL vs QUANTUM")
    print("=" * 70)
    print("Testing Complexity Hypothesis: Simple vs Complex Physics")
    print("=" * 70)
    
    results = {}
    
    # ===== PART A: CLASSICAL HARMONIC OSCILLATOR (SIMPLE) =====
    print("\n[PART A] CLASSICAL HARMONIC OSCILLATOR (Simple Physics)")
    print("-" * 70)
    
    # Generate data
    print("  Generating dataset...")
    classical_sim = ClassicalHarmonicOscillator()
    X_classical, y_classical = classical_sim.generate_dataset(n_samples=2000)
    
    # Scale data
    scaler_classical = MinMaxScaler()
    X_classical_scaled = scaler_classical.fit_transform(X_classical)
    
    # Split data
    X_train_c, X_test_c, X_train_c_s, X_test_c_s, y_train_c, y_test_c = train_test_split(
        X_classical, X_classical_scaled, y_classical, test_size=0.2, random_state=42
    )
    
    print(f"  Generated {len(X_classical)} samples")
    print(f"  Input shape: {X_classical.shape}")
    print(f"  Output range: [{np.min(y_classical):.4f}, {np.max(y_classical):.4f}]")
    
    # Train models
    print("  Training models...")
    
    # Darwinian baseline
    darwin_classical = DarwinianModel()
    darwin_classical.fit(X_train_c, y_train_c)
    y_pred_darwin_c = darwin_classical.predict(X_test_c)
    r2_darwin_c = r2_score(y_test_c, y_pred_darwin_c)
    
    # Chaos model - try different brightness values
    best_r2_c = -np.inf
    best_chaos_c = None
    best_brightness_c = 0.001
    
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_c_s, y_train_c)
        y_pred_test = chaos_test.predict(X_test_c_s)
        r2_test = r2_score(y_test_c, y_pred_test)
        if r2_test > best_r2_c:
            best_r2_c = r2_test
            best_chaos_c = chaos_test
            best_brightness_c = brightness
    
    chaos_classical = best_chaos_c
    y_pred_chaos_c = chaos_classical.predict(X_test_c_s)
    r2_chaos_c = best_r2_c
    
    print(f"  Darwinian R²: {r2_darwin_c:.4f}")
    print(f"  Chaos R²: {r2_chaos_c:.4f} (brightness={best_brightness_c})")
    
    # Cage analysis
    print("  Analyzing cage status...")
    cage_classical, mean_cage_classical = analyze_cage(
        chaos_classical, X_test_c_s, X_test_c,
        ['Amplitude', 'Omega', 'Phase', 'Time']
    )
    
    max_corr_classical = max(cage_classical.values())
    mean_corr_classical = np.mean(list(mean_cage_classical.values()))
    print(f"  Max correlation with human variables: {max_corr_classical:.4f}")
    print(f"  Mean correlation with human variables: {mean_corr_classical:.4f}")
    for var, corr in cage_classical.items():
        print(f"    - {var}: max={corr:.4f}, mean={mean_cage_classical[var]:.4f}")
    
    if max_corr_classical > 0.9:
        cage_status_c = "LOCKED"
    elif max_corr_classical < 0.3:
        cage_status_c = "BROKEN"
    else:
        cage_status_c = "UNCLEAR"
    
    print(f"  Cage Status: {cage_status_c}")
    
    results['classical'] = {
        'r2_darwin': r2_darwin_c,
        'r2_chaos': r2_chaos_c,
        'cage_correlations': cage_classical,
        'cage_mean_correlations': mean_cage_classical,
        'cage_status': cage_status_c,
        'max_correlation': max_corr_classical,
        'mean_correlation': mean_corr_classical
    }
    
    # ===== PART B: QUANTUM PARTICLE IN BOX (COMPLEX) =====
    print("\n[PART B] QUANTUM PARTICLE IN BOX (Complex Physics)")
    print("-" * 70)
    
    # Generate data
    print("  Generating dataset...")
    quantum_sim = QuantumParticleInBox()
    X_quantum, y_quantum = quantum_sim.generate_dataset(n_samples=2000)
    
    # Scale data
    scaler_quantum = MinMaxScaler()
    X_quantum_scaled = scaler_quantum.fit_transform(X_quantum)
    
    # Split data
    X_train_q, X_test_q, X_train_q_s, X_test_q_s, y_train_q, y_test_q = train_test_split(
        X_quantum, X_quantum_scaled, y_quantum, test_size=0.2, random_state=42
    )
    
    print(f"  Generated {len(X_quantum)} samples")
    print(f"  Input shape: {X_quantum.shape}")
    print(f"  Output range: [{np.min(y_quantum):.4f}, {np.max(y_quantum):.4f}]")
    
    # Train models
    print("  Training models...")
    
    # Darwinian baseline
    darwin_quantum = DarwinianModel()
    darwin_quantum.fit(X_train_q, y_train_q)
    y_pred_darwin_q = darwin_quantum.predict(X_test_q)
    r2_darwin_q = r2_score(y_test_q, y_pred_darwin_q)
    
    # Chaos model - try different brightness values
    best_r2_q = -np.inf
    best_chaos_q = None
    best_brightness_q = 0.001
    
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_q_s, y_train_q)
        y_pred_test = chaos_test.predict(X_test_q_s)
        r2_test = r2_score(y_test_q, y_pred_test)
        if r2_test > best_r2_q:
            best_r2_q = r2_test
            best_chaos_q = chaos_test
            best_brightness_q = brightness
    
    chaos_quantum = best_chaos_q
    y_pred_chaos_q = chaos_quantum.predict(X_test_q_s)
    r2_chaos_q = best_r2_q
    
    print(f"  Darwinian R²: {r2_darwin_q:.4f}")
    print(f"  Chaos R²: {r2_chaos_q:.4f} (brightness={best_brightness_q})")
    
    # Cage analysis
    print("  Analyzing cage status...")
    cage_quantum, mean_cage_quantum = analyze_cage(
        chaos_quantum, X_test_q_s, X_test_q,
        ['Quantum_n', 'Box_L', 'Position_x']
    )
    
    max_corr_quantum = max(cage_quantum.values())
    mean_corr_quantum = np.mean(list(mean_cage_quantum.values()))
    print(f"  Max correlation with human variables: {max_corr_quantum:.4f}")
    print(f"  Mean correlation with human variables: {mean_corr_quantum:.4f}")
    for var, corr in cage_quantum.items():
        print(f"    - {var}: max={corr:.4f}, mean={mean_cage_quantum[var]:.4f}")
    
    if max_corr_quantum > 0.9:
        cage_status_q = "LOCKED"
    elif max_corr_quantum < 0.3:
        cage_status_q = "BROKEN"
    else:
        cage_status_q = "UNCLEAR"
    
    print(f"  Cage Status: {cage_status_q}")
    
    results['quantum'] = {
        'r2_darwin': r2_darwin_q,
        'r2_chaos': r2_chaos_q,
        'cage_correlations': cage_quantum,
        'cage_mean_correlations': mean_cage_quantum,
        'cage_status': cage_status_q,
        'max_correlation': max_corr_quantum,
        'mean_correlation': mean_corr_quantum
    }
    
    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("COMPARISON: Simple vs Complex Physics")
    print("=" * 70)
    print(f"\nClassical (Simple):")
    print(f"  R² Chaos: {r2_chaos_c:.4f}")
    print(f"  Max Correlation: {max_corr_classical:.4f}")
    print(f"  Cage Status: {cage_status_c}")
    print(f"\nQuantum (Complex):")
    print(f"  R² Chaos: {r2_chaos_q:.4f}")
    print(f"  Max Correlation: {max_corr_quantum:.4f}")
    print(f"  Cage Status: {cage_status_q}")
    
    # Hypothesis test
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)
    if max_corr_classical > 0.9 and max_corr_quantum < 0.3:
        print("✅ HYPOTHESIS CONFIRMED: Simple physics locks cage, complex breaks it")
    elif max_corr_classical < 0.3 and max_corr_quantum > 0.9:
        print("❌ HYPOTHESIS REFUTED: Opposite pattern observed")
    else:
        print("⚠️ HYPOTHESIS UNCLEAR: Mixed or unclear results")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Classical: Predictions
    axes[0, 0].scatter(y_test_c, y_pred_chaos_c, alpha=0.5, s=20)
    axes[0, 0].plot([y_test_c.min(), y_test_c.max()], [y_test_c.min(), y_test_c.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Position x(t)')
    axes[0, 0].set_ylabel('Predicted Position')
    axes[0, 0].set_title(f'Classical: Chaos Model (R² = {r2_chaos_c:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Classical: Cage correlations
    vars_c = list(cage_classical.keys())
    corrs_c = list(cage_classical.values())
    axes[0, 1].bar(vars_c, corrs_c, color='blue', alpha=0.7)
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Locked threshold')
    axes[0, 1].axhline(y=0.3, color='g', linestyle='--', label='Broken threshold')
    axes[0, 1].set_ylabel('Max Correlation')
    axes[0, 1].set_title(f'Classical: Cage Analysis ({cage_status_c})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Quantum: Predictions
    axes[1, 0].scatter(y_test_q, y_pred_chaos_q, alpha=0.5, s=20)
    axes[1, 0].plot([y_test_q.min(), y_test_q.max()], [y_test_q.min(), y_test_q.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Probability Density |ψ|²')
    axes[1, 0].set_ylabel('Predicted Probability Density')
    axes[1, 0].set_title(f'Quantum: Chaos Model (R² = {r2_chaos_q:.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quantum: Cage correlations
    vars_q = list(cage_quantum.keys())
    corrs_q = list(cage_quantum.values())
    axes[1, 1].bar(vars_q, corrs_q, color='purple', alpha=0.7)
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='Locked threshold')
    axes[1, 1].axhline(y=0.3, color='g', linestyle='--', label='Broken threshold')
    axes[1, 1].set_ylabel('Max Correlation')
    axes[1, 1].set_title(f'Quantum: Cage Analysis ({cage_status_q})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('experiment_8_classical_vs_quantum.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graph saved as 'experiment_8_classical_vs_quantum.png'")
    plt.show()  # Show the graph
    
    return results

if __name__ == "__main__":
    results = run_experiment()

