"""
Physics vs. Darwin: Experiment 7
Emergent Order (Phase Transitions)
----------------------------------

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
Test whether a chaos-based system can detect phase transitions in physical systems
without explicit knowledge of temperature or free energy concepts.

Hypothesis:
A chaotic optical system might naturally resonate with emergent patterns that
signal phase transitions, potentially detecting critical points without human
thermodynamic concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. ISING MODEL SIMULATOR (2D) ---
class IsingModelSimulator:
    """
    Simulates 2D Ising model for phase transition detection.
    
    The Ising model shows a phase transition at critical temperature:
    T_c ≈ 2.269 (in units of J/k_B)
    
    Below T_c: Ferromagnetic (ordered)
    Above T_c: Paramagnetic (disordered)
    """
    def __init__(self, size=20):
        """
        size: Lattice size (size x size)
        """
        self.size = size
        self.n_spins = size * size
        
    def generate_configuration(self, temperature):
        """
        Generate a spin configuration at given temperature using Metropolis algorithm.
        Returns: spin configuration (flattened) and magnetization
        """
        # Initialize random spins
        spins = np.random.choice([-1, 1], size=(self.size, self.size))
        
        # Metropolis algorithm
        # Use more steps for better thermalization
        # Typically need ~100-1000 sweeps (N steps each) for equilibrium
        n_steps = self.n_spins * 50  # Increased for better convergence
        
        for _ in range(n_steps):
            # Random site
            i, j = np.random.randint(0, self.size, 2)
            
            # Calculate energy change
            neighbors = (
                spins[(i+1) % self.size, j] +
                spins[(i-1) % self.size, j] +
                spins[i, (j+1) % self.size] +
                spins[i, (j-1) % self.size]
            )
            delta_E = 2 * spins[i, j] * neighbors
            
            # Metropolis acceptance
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / temperature):
                spins[i, j] *= -1
        
        # Calculate magnetization (order parameter)
        magnetization = np.mean(spins)
        
        return spins.flatten(), magnetization
    
    def generate_dataset(self, n_samples=2000, temp_range=(0.5, 4.0)):
        """
        Generate dataset of Ising configurations at different temperatures.
        
        Input: Spin configuration (flattened)
        Output: Temperature (hidden) or magnetization
        """
        np.random.seed(42)
        
        # Generate temperatures
        temperatures = np.random.uniform(temp_range[0], temp_range[1], n_samples)
        
        # Generate configurations
        configurations = []
        magnetizations = []
        
        print(f"  Generating {n_samples} configurations...")
        for i, T in enumerate(temperatures):
            if (i + 1) % 500 == 0:
                print(f"    Progress: {i+1}/{n_samples}")
            config, mag = self.generate_configuration(T)
            configurations.append(config)
            magnetizations.append(mag)
        
        configurations = np.array(configurations)
        magnetizations = np.array(magnetizations)
        
        # Input: Spin configuration (flattened)
        X = configurations
        # Output: Magnetization (order parameter)
        y = magnetizations
        
        return X, y, temperatures
    
    def calculate_critical_temperature(self):
        """Return theoretical critical temperature"""
        # For 2D Ising: T_c = 2 / ln(1 + sqrt(2)) ≈ 2.269
        return 2.0 / np.log(1 + np.sqrt(2))

# --- 2. DARWINIAN BASELINE MODEL ---
class DarwinianModel:
    """Baseline using polynomial features"""
    def __init__(self, degree=2):
        # Lower degree due to high dimensionality
        self.poly = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=1.0)  # Higher regularization for high-dim
        
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

# --- 3. PHASE TRANSITION CHAOS MODEL ---
class PhaseTransitionChaosModel:
    """Optical chaos model for detecting phase transitions"""
    def __init__(self, n_features=2048, brightness=0.001):
        # Reduced features due to high input dimensionality
        self.n_features = n_features
        self.brightness = brightness
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=0.1)
        self.reservoir = None
        
    def _chaos_transform(self, X):
        """Chaotic optical transformation"""
        n_samples = X.shape[0]
        
        if self.reservoir is None:
            np.random.seed(999)
            # Input is already high-dimensional (size*size), so we project to features
            self.reservoir = np.random.randn(X.shape[1], self.n_features)
            
        # Optical mixing
        optical_field = X @ self.reservoir
        optical_field *= self.brightness
        
        # FFT interference
        spectrum = np.fft.rfft(optical_field, axis=1)
        intensity = np.abs(spectrum) ** 2
        
        # Normalize
        intensity = np.tanh(intensity)
        
        return intensity
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        features = self._chaos_transform(X_scaled)
        self.readout.fit(features, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        features = self._chaos_transform(X_scaled)
        return self.readout.predict(features)
    
    def get_internal_state(self, X):
        """Get internal features for cage analysis"""
        X_scaled = self.scaler.transform(X)
        return self._chaos_transform(X_scaled)

# --- 4. MAIN EXPERIMENT ---
def run_experiment_7():
    print("❄️ STARTING EXPERIMENT 7: PHASE TRANSITIONS (EMERGENT ORDER)")
    print("=" * 70)
    print("Testing if chaos can detect phase transitions without temperature")
    print("=" * 70)
    
    # 1. Generate Data
    print("\n[Generating Dataset]")
    print("  Using 2D Ising Model (20x20 lattice)")
    sim = IsingModelSimulator(size=20)
    
    # Smaller dataset due to computational cost
    X, y, temperatures = sim.generate_dataset(n_samples=1000, temp_range=(0.5, 4.0))
    
    print(f"\n  Generated {len(X)} samples")
    print(f"  Input shape: {X.shape} (each sample is {X.shape[1]} spins)")
    print(f"  Magnetization range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Temperature range: [{temperatures.min():.2f}, {temperatures.max():.2f}]")
    print(f"  Critical temperature (theoretical): {sim.calculate_critical_temperature():.3f}")
    
    # Validate data: Check if magnetization shows phase transition
    print("\n[Validating Data]")
    # Sort by temperature
    sort_idx = np.argsort(temperatures)
    temps_sorted = temperatures[sort_idx]
    mags_sorted = y[sort_idx]
    
    # Check if we see phase transition (magnetization drops near T_c)
    T_c = sim.calculate_critical_temperature()
    below_Tc = mags_sorted[temps_sorted < T_c]
    above_Tc = mags_sorted[temps_sorted > T_c]
    
    print(f"  Mean magnetization below T_c: {np.abs(below_Tc).mean():.4f}")
    print(f"  Mean magnetization above T_c: {np.abs(above_Tc).mean():.4f}")
    print(f"  Phase transition visible: {'Yes' if np.abs(below_Tc).mean() > np.abs(above_Tc).mean() else 'No'}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        X, y, temperatures, test_size=0.2, random_state=42
    )
    
    # 3. Train Models
    print("\n[Training Models]")
    
    # Darwinian Model
    print("  - Training Darwinian Baseline (Polynomial)...")
    darwin_model = DarwinianModel(degree=2)
    darwin_model.fit(X_train, y_train)
    y_pred_darwin = darwin_model.predict(X_test)
    
    # Phase Transition Chaos Model
    print("  - Training Phase Transition Chaos Model...")
    # brightness=0.0001 is optimal for this problem (validated in benchmark)
    phase_model = PhaseTransitionChaosModel(n_features=2048, brightness=0.0001)
    phase_model.fit(X_train, y_train)
    y_pred_phase = phase_model.predict(X_test)
    
    # 4. Evaluate
    print("\n[Results]")
    r2_darwin = r2_score(y_test, y_pred_darwin)
    r2_phase = r2_score(y_test, y_pred_phase)
    
    print(f"  Darwinian R²: {r2_darwin:.4f}")
    print(f"  Phase Chaos R²: {r2_phase:.4f}")
    
    # 5. Cage Analysis
    print("\n[Cage Analysis: Internal Feature Correlations]")
    print("  Checking if internal features correlate with temperature...")
    
    # Get internal states
    internal_states = phase_model.get_internal_state(X_test)
    
    # Correlations with temperature (hidden variable)
    corrs_temp = []
    
    n_sample = min(500, internal_states.shape[1])
    sample_indices = np.random.choice(internal_states.shape[1], n_sample, replace=False)
    
    for idx in sample_indices:
        corr = np.abs(np.corrcoef(internal_states[:, idx], T_test)[0, 1])
        corrs_temp.append(corr)
    
    max_corr_temp = np.max(corrs_temp)
    mean_corr_temp = np.mean(corrs_temp)
    
    print(f"  Max correlation with Temperature: {max_corr_temp:.4f}")
    print(f"  Mean correlation with Temperature: {mean_corr_temp:.4f}")
    
    # Also check correlation with magnetization (order parameter)
    corrs_mag = []
    for idx in sample_indices:
        corr = np.abs(np.corrcoef(internal_states[:, idx], y_test)[0, 1])
        corrs_mag.append(corr)
    
    max_corr_mag = np.max(corrs_mag)
    
    print(f"  Max correlation with Magnetization: {max_corr_mag:.4f}")
    
    # Cage verdict
    if max_corr_temp > 0.8:
        print("\n🔒 CAGE STATUS: LOCKED")
        print("   The model reconstructed temperature variable")
    elif max_corr_temp < 0.5 and r2_phase > 0.7:
        print("\n🔓 CAGE STATUS: BROKEN")
        print("   The model detected phase transition without reconstructing temperature")
    else:
        print("\n🟡 CAGE STATUS: UNCLEAR")
        print("   Intermediate correlation levels")
    
    # 6. Visualization
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Prediction Scatter (Phase Model)
    plt.subplot(2, 4, 1)
    plt.scatter(y_test, y_pred_phase, alpha=0.3, s=5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
    plt.title(f"Phase Chaos Model\nR² = {r2_phase:.3f}")
    plt.xlabel("True Magnetization")
    plt.ylabel("Predicted Magnetization")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Scatter (Darwinian)
    plt.subplot(2, 4, 2)
    plt.scatter(y_test, y_pred_darwin, alpha=0.3, s=5, c='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
    plt.title(f"Darwinian Baseline\nR² = {r2_darwin:.3f}")
    plt.xlabel("True Magnetization")
    plt.ylabel("Predicted Magnetization")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Phase Transition (True Physics)
    plt.subplot(2, 4, 3)
    sort_idx_test = np.argsort(T_test)
    plt.scatter(T_test[sort_idx_test], y_test[sort_idx_test], alpha=0.3, s=5, c='blue')
    plt.axvline(T_c, color='red', linestyle='--', label=f'T_c = {T_c:.3f}')
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.title("True Phase Transition")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Predicted Phase Transition
    plt.subplot(2, 4, 4)
    plt.scatter(T_test[sort_idx_test], y_pred_phase[sort_idx_test], alpha=0.3, s=5, c='green')
    plt.axvline(T_c, color='red', linestyle='--', label=f'T_c = {T_c:.3f}')
    plt.xlabel("Temperature")
    plt.ylabel("Predicted Magnetization")
    plt.title("Predicted Phase Transition")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Temperature Correlations
    plt.subplot(2, 4, 5)
    plt.hist(corrs_temp, bins=50, alpha=0.7, color='red')
    plt.axvline(max_corr_temp, color='black', linestyle='--', label=f'Max: {max_corr_temp:.3f}')
    plt.xlabel("Correlation with Temperature")
    plt.ylabel("Count")
    plt.title("Internal Correlations\nwith Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Magnetization Correlations
    plt.subplot(2, 4, 6)
    plt.hist(corrs_mag, bins=50, alpha=0.7, color='blue')
    plt.axvline(max_corr_mag, color='black', linestyle='--', label=f'Max: {max_corr_mag:.3f}')
    plt.xlabel("Correlation with Magnetization")
    plt.ylabel("Count")
    plt.title("Internal Correlations\nwith Magnetization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Error vs Temperature
    plt.subplot(2, 4, 7)
    error = np.abs(y_test - y_pred_phase)
    plt.scatter(T_test, error, alpha=0.3, s=5)
    plt.axvline(T_c, color='red', linestyle='--', label=f'T_c')
    plt.xlabel("Temperature")
    plt.ylabel("Absolute Error")
    plt.title("Prediction Error vs Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Model Comparison
    plt.subplot(2, 4, 8)
    models = ['Darwinian', 'Phase\nChaos']
    r2_scores = [r2_darwin, r2_phase]
    colors = ['orange', 'blue']
    bars = plt.bar(models, r2_scores, color=colors, alpha=0.7)
    plt.ylabel("R² Score")
    plt.title("Model Performance")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiment_7_phase_transitions.png', dpi=150)
    print("\n📊 Graph saved as 'experiment_7_phase_transitions.png'")
    plt.show()
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Darwinian R²: {r2_darwin:.4f}")
    print(f"Phase Chaos R²: {r2_phase:.4f}")
    print(f"\nCage Analysis:")
    print(f"  Max Temperature Correlation: {max_corr_temp:.4f}")
    print(f"  Max Magnetization Correlation: {max_corr_mag:.4f}")
    print(f"  Critical Temperature: {T_c:.3f}")

if __name__ == "__main__":
    run_experiment_7()

