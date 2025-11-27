"""
Physics vs. Darwin: Experiment 6
Quantum Interference (The Double Slit)
--------------------------------------

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
Test whether an optical AI system can learn quantum interference patterns
from the double-slit experiment without explicit knowledge of wave functions,
superposition, or probability amplitudes.

Hypothesis:
A chaotic optical system might naturally resonate with interference patterns,
potentially discovering the quantum behavior without human concepts like
"wave function" or "probability amplitude".
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. QUANTUM INTERFERENCE SIMULATOR ---
class DoubleSlitSimulator:
    """
    Simulates the double-slit experiment quantum interference pattern.
    
    The probability distribution on the screen follows:
    P(x) ∝ |ψ₁(x) + ψ₂(x)|²
    
    Where ψ₁ and ψ₂ are wave functions from each slit.
    This creates an interference pattern with fringes.
    """
    def __init__(self, wavelength=1.0, slit_separation=2.0, screen_distance=10.0):
        """
        Parameters:
        - wavelength: De Broglie wavelength (normalized)
        - slit_separation: Distance between slits
        - screen_distance: Distance from slits to screen
        """
        self.wavelength = wavelength
        self.slit_separation = slit_separation
        self.screen_distance = screen_distance
    
    def calculate_interference_pattern(self, x_positions, phase_diff=None):
        """
        Calculate quantum interference pattern on screen.
        
        For double-slit:
        P(x) = A * cos²(π * d * x / (λ * L))
        
        Where:
        - d: slit separation
        - x: position on screen
        - λ: wavelength
        - L: screen distance
        - A: normalization constant
        """
        # Path difference from each slit
        # Simplified: assuming small angles, path difference ≈ d*x/L
        path_diff = self.slit_separation * x_positions / self.screen_distance
        
        # Phase difference
        phase = 2 * np.pi * path_diff / self.wavelength
        
        # Interference pattern (probability)
        # P(x) = (1 + cos(phase)) / 2 for normalized pattern
        probability = 0.5 * (1 + np.cos(phase))
        
        # Ensure non-negative
        probability = np.maximum(probability, 0.0)
        
        # Only normalize if we have multiple points (for full pattern generation)
        # For single point predictions, return the raw probability value
        if len(probability) > 1:
            # Normalize so sum equals number of points (for probability density)
            probability = probability / np.sum(probability) * len(probability)
        
        return probability
    
    def generate_dataset(self, n_samples=3000):
        """
        Generate dataset of double-slit interference patterns.
        
        Input: [wavelength, slit_separation, screen_distance, position_on_screen]
        Output: Detection probability at that position
        """
        np.random.seed(42)
        
        # Vary parameters
        wavelength = np.random.uniform(0.5, 2.0, n_samples)
        slit_separation = np.random.uniform(1.0, 5.0, n_samples)
        screen_distance = np.random.uniform(5.0, 20.0, n_samples)
        
        # Position on screen (normalized to [-10, 10])
        position = np.random.uniform(-10.0, 10.0, n_samples)
        
        # Calculate probability for each sample
        probabilities = []
        for i in range(n_samples):
            sim = DoubleSlitSimulator(
                wavelength=wavelength[i],
                slit_separation=slit_separation[i],
                screen_distance=screen_distance[i]
            )
            prob = sim.calculate_interference_pattern(np.array([position[i]]))
            probabilities.append(prob[0])
        
        probabilities = np.array(probabilities)
        
        # Input: [wavelength, slit_separation, screen_distance, position]
        X = np.column_stack((wavelength, slit_separation, screen_distance, position))
        # Output: probability
        y = probabilities
        
        return X, y
    
    def generate_full_pattern(self, wavelength, slit_separation, screen_distance, n_points=200):
        """Generate full interference pattern for visualization"""
        x_positions = np.linspace(-10, 10, n_points)
        sim = DoubleSlitSimulator(wavelength, slit_separation, screen_distance)
        pattern = sim.calculate_interference_pattern(x_positions)
        return x_positions, pattern

# --- 2. DARWINIAN BASELINE MODEL ---
class DarwinianModel:
    """Baseline using polynomial features"""
    def __init__(self, degree=4):
        self.poly = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=0.1)
        
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

# --- 3. QUANTUM CHAOS MODEL ---
class QuantumChaosModel:
    """Optical chaos model for quantum interference patterns"""
    def __init__(self, n_features=4096, brightness=0.001):
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
            self.reservoir = np.random.randn(X.shape[1], self.n_features)
            
        # Optical mixing
        optical_field = X @ self.reservoir
        optical_field *= self.brightness
        
        # FFT interference (naturally captures wave-like behavior)
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
def run_experiment_6():
    print("🌊 STARTING EXPERIMENT 6: QUANTUM INTERFERENCE (DOUBLE SLIT)")
    print("=" * 70)
    print("Testing if chaos can learn quantum interference patterns")
    print("without wave function concepts")
    print("=" * 70)
    
    # 1. Generate Data
    print("\n[Generating Dataset]")
    sim = DoubleSlitSimulator()
    X, y = sim.generate_dataset(n_samples=3000)
    print(f"   Generated {len(X)} samples")
    print(f"   Probability range: {np.min(y):.4f} - {np.max(y):.4f}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Train Models
    print("\n[Training Models]")
    
    # Darwinian Model
    print("  - Training Darwinian Baseline (Polynomial)...")
    darwin_model = DarwinianModel(degree=4)
    darwin_model.fit(X_train, y_train)
    y_pred_darwin = darwin_model.predict(X_test)
    
    # Quantum Chaos Model
    print("  - Training Quantum Chaos Model (Optical Interference)...")
    quantum_model = QuantumChaosModel(n_features=4096, brightness=0.001)
    quantum_model.fit(X_train, y_train)
    y_pred_quantum = quantum_model.predict(X_test)
    
    # 4. Evaluate
    print("\n[Results]")
    r2_darwin = r2_score(y_test, y_pred_darwin)
    r2_quantum = r2_score(y_test, y_pred_quantum)
    
    print(f"  Darwinian R²: {r2_darwin:.4f}")
    print(f"  Quantum Chaos R²: {r2_quantum:.4f}")
    
    # 5. Cage Analysis
    print("\n[Cage Analysis: Internal Feature Correlations]")
    print("  Checking if internal features correlate with wave concepts...")
    
    # Calculate wave-related quantities
    wavelength, slit_sep, screen_dist, position = X_test.T
    
    # Phase (key quantum concept): φ = 2π * d * x / (λ * L)
    phase = 2 * np.pi * slit_sep * position / (wavelength * screen_dist)
    
    # Path difference (classical wave concept)
    path_diff = slit_sep * position / screen_dist
    
    # Wavenumber: k = 2π / λ
    wavenumber = 2 * np.pi / wavelength
    
    # Get internal states
    internal_states = quantum_model.get_internal_state(X_test)
    
    # Correlations
    corrs_phase = []
    corrs_path_diff = []
    corrs_wavenumber = []
    
    n_sample = min(1000, internal_states.shape[1])
    sample_indices = np.random.choice(internal_states.shape[1], n_sample, replace=False)
    
    for idx in sample_indices:
        corr_p = np.abs(np.corrcoef(internal_states[:, idx], phase)[0, 1])
        corr_pd = np.abs(np.corrcoef(internal_states[:, idx], path_diff)[0, 1])
        corr_k = np.abs(np.corrcoef(internal_states[:, idx], wavenumber)[0, 1])
        corrs_phase.append(corr_p)
        corrs_path_diff.append(corr_pd)
        corrs_wavenumber.append(corr_k)
    
    max_corr_phase = np.max(corrs_phase)
    max_corr_path = np.max(corrs_path_diff)
    max_corr_wavenumber = np.max(corrs_wavenumber)
    mean_corr_phase = np.mean(corrs_phase)
    
    print(f"  Max correlation with Phase: {max_corr_phase:.4f}")
    print(f"  Max correlation with Path Difference: {max_corr_path:.4f}")
    print(f"  Max correlation with Wavenumber: {max_corr_wavenumber:.4f}")
    print(f"  Mean correlation with Phase: {mean_corr_phase:.4f}")
    
    # Cage verdict
    if max_corr_phase > 0.8 or max_corr_path > 0.8:
        print("\n🔒 CAGE STATUS: LOCKED")
        print("   The model reconstructed wave concepts (phase/path difference)")
    elif max_corr_phase < 0.5 and max_corr_path < 0.5 and r2_quantum > 0.9:
        print("\n🔓 CAGE STATUS: BROKEN")
        print("   The model learned interference without reconstructing wave concepts")
    else:
        print("\n🟡 CAGE STATUS: UNCLEAR")
        print("   Intermediate correlation levels")
    
    # 6. Visualization
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Prediction Scatter (Quantum Model)
    plt.subplot(2, 4, 1)
    plt.scatter(y_test, y_pred_quantum, alpha=0.3, s=5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
    plt.title(f"Quantum Chaos Model\nR² = {r2_quantum:.3f}")
    plt.xlabel("True Probability")
    plt.ylabel("Predicted Probability")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Scatter (Darwinian)
    plt.subplot(2, 4, 2)
    plt.scatter(y_test, y_pred_darwin, alpha=0.3, s=5, c='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
    plt.title(f"Darwinian Baseline\nR² = {r2_darwin:.3f}")
    plt.xlabel("True Probability")
    plt.ylabel("Predicted Probability")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Example Interference Pattern (True)
    plt.subplot(2, 4, 3)
    example_idx = 0
    wl, sep, dist, _ = X_test[example_idx]
    x_pattern, pattern = sim.generate_full_pattern(wl, sep, dist)
    plt.plot(x_pattern, pattern, 'b-', lw=2, label='True Pattern')
    plt.title("Example Interference Pattern\n(True Physics)")
    plt.xlabel("Position on Screen")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Predicted Pattern
    plt.subplot(2, 4, 4)
    # Predict full pattern
    x_pred = np.linspace(-10, 10, 200)
    X_pattern = np.column_stack((
        np.full(200, wl),
        np.full(200, sep),
        np.full(200, dist),
        x_pred
    ))
    y_pattern_pred = quantum_model.predict(X_pattern)
    plt.plot(x_pred, y_pattern_pred, 'r--', lw=2, label='Predicted')
    plt.plot(x_pattern, pattern, 'b-', lw=1, alpha=0.5, label='True')
    plt.title("Predicted vs True Pattern")
    plt.xlabel("Position on Screen")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Phase Correlations
    plt.subplot(2, 4, 5)
    plt.hist(corrs_phase, bins=50, alpha=0.7, color='green')
    plt.axvline(max_corr_phase, color='red', linestyle='--', label=f'Max: {max_corr_phase:.3f}')
    plt.xlabel("Correlation with Phase")
    plt.ylabel("Count")
    plt.title("Internal Correlations\nwith Phase")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Path Difference Correlations
    plt.subplot(2, 4, 6)
    plt.hist(corrs_path_diff, bins=50, alpha=0.7, color='purple')
    plt.axvline(max_corr_path, color='red', linestyle='--', label=f'Max: {max_corr_path:.3f}')
    plt.xlabel("Correlation with Path Difference")
    plt.ylabel("Count")
    plt.title("Internal Correlations\nwith Path Difference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Error Distribution
    plt.subplot(2, 4, 7)
    error_quantum = np.abs(y_test - y_pred_quantum)
    error_darwin = np.abs(y_test - y_pred_darwin)
    plt.hist(error_quantum, bins=50, alpha=0.7, label='Quantum', color='blue')
    plt.hist(error_darwin, bins=50, alpha=0.7, label='Darwinian', color='orange')
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Model Comparison
    plt.subplot(2, 4, 8)
    models = ['Darwinian', 'Quantum\nChaos']
    r2_scores = [r2_darwin, r2_quantum]
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
    plt.savefig('experiment_6_quantum_interference.png', dpi=150)
    print("\n📊 Graph saved as 'experiment_6_quantum_interference.png'")
    plt.show()
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Darwinian R²: {r2_darwin:.4f}")
    print(f"Quantum Chaos R²: {r2_quantum:.4f}")
    print(f"\nCage Analysis:")
    print(f"  Max Phase Correlation: {max_corr_phase:.4f}")
    print(f"  Max Path Difference Correlation: {max_corr_path:.4f}")
    print(f"  Max Wavenumber Correlation: {max_corr_wavenumber:.4f}")

if __name__ == "__main__":
    run_experiment_6()

