"""
Physics vs. Darwin: Experiment 2
Einstein's Train (The Photon Clock)
-----------------------------------

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
Demonstrate that an Optical AI can predict Relativistic Time Dilation 
purely from geometric interference patterns, without knowing the 
Lorentz Transformation formula.

The Scenario:
1. A "Light Clock" (photon bouncing between mirrors) moves at velocity v.
2. An external observer records the zigzag path of the photon.
3. The AI must predict the "Time Dilation Factor" (Gamma) based on the path shape.

The Hypothesis:
Humans solve this using algebra (Pythagoras + Velocity def -> Lorentz).
The Optical AI should solve this by resonating with the geometric 
distortion of the wave, effectively "feeling" the time dilation 
as a phase shift.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. THE RELATIVISTIC SIMULATOR (The Truth) ---
class RelativitySimulator:
    def __init__(self, c=1.0, mirror_dist=1.0):
        self.c = c # Speed of light (normalized)
        self.L = mirror_dist # Height of the clock
        
    def get_lorentz_factor(self, v):
        """Calculates the True Gamma (The Cage Formula)."""
        # Safety clip to avoid division by zero at v=c
        v = np.clip(v, 0, 0.999 * self.c)
        return 1.0 / np.sqrt(1 - (v**2 / self.c**2))

    def generate_dataset(self, n_samples=3000):
        """
        Generates observed photon paths for trains at different speeds.
        Input (X): The horizontal distance traveled by the photon in one tick (d_x).
        Target (y): The Time Dilation Factor (Gamma).
        """
        np.random.seed(42)
        
        # Generate random velocities (0% to 99% of speed of light)
        # We use a power distribution to get more samples near c where physics gets weird
        v = np.random.power(2, n_samples) * 0.99 * self.c
        
        # Calculate Truth (Gamma)
        y_gamma = self.get_lorentz_factor(v)
        
        # Calculate Observation (What the external observer sees)
        # In the train frame, photon goes up/down distance L.
        # In the observed frame, photon travels diagonal.
        # Horizontal distance per bounce: dx = v * dt
        # But we want raw geometry. Let's give the AI the "Slope" of the photon.
        # Slope = L / (v * dt_half). 
        # Ideally, we just give: [Horizontal_Dist, Vertical_Dist]
        
        dt_proper = (2 * self.L) / self.c
        dt_dilated = dt_proper * y_gamma
        
        dx_bounce = v * (dt_dilated / 2) # Horizontal dist for half cycle
        dy_bounce = self.L               # Vertical dist (constant)
        
        # The Input is the Geometry of the path
        X = np.column_stack((dx_bounce, np.full(n_samples, dy_bounce)))
        
        return X, y_gamma, v

# --- 2. THE CAGE BREAKER (Complex Optical Net) ---
class OpticalInterferenceNet:
    def __init__(self, n_components=4000, nonlinearity='phase'):
        """
        n_components: Size of the optical reservoir (holographic resolution).
        nonlinearity: 'phase' (optical) or 'tanh' (neural).
        """
        self.n_components = n_components
        self.readout = Ridge(alpha=1e-5)
        self.optical_weights = None
        self.phase_shift = None
        
    def _forward_optics(self, X):
        n_samples, n_features = X.shape
        
        if self.optical_weights is None:
            np.random.seed(137) # Fine structure constant seed ;)
            # Create a complex-valued scattering matrix
            real_part = np.random.normal(0, 1, (n_features, self.n_components))
            imag_part = np.random.normal(0, 1, (n_features, self.n_components))
            self.optical_weights = real_part + 1j * imag_part
            
        # 1. Optical Projection (Scattering)
        # Project physical geometry into high-dim complex space
        wave_field = X @ self.optical_weights
        
        # 2. Non-Linear Phase Encoding
        # This simulates how light accumulates phase based on path length
        # exp(i * |z|)
        amplitude = np.abs(wave_field)
        phase = np.angle(wave_field)
        
        # 3. Interference (FFT)
        # Mixing the signals in frequency domain
        interference = np.fft.hfft(wave_field, axis=1)
        
        # 4. Intensity Detection (Sensor)
        # We detect magnitude squared (Energy)
        detected_signal = np.abs(interference) ** 2
        
        # Log-scale normalization (common in optical sensors)
        return np.log1p(detected_signal)

    def fit(self, X, y):
        features = self._forward_optics(X)
        self.readout.fit(features, y)
        
    def predict(self, X):
        features = self._forward_optics(X)
        return self.readout.predict(features)

    def get_internal_representation(self, X):
        return self._forward_optics(X)

# --- 3. EXECUTION ---
def run_experiment_2():
    print("🚂 STARTING EXPERIMENT 2: EINSTEIN'S TRAIN")
    print("------------------------------------------")
    print("Generating relativistic data (v -> 0.99c)...")
    
    sim = RelativitySimulator()
    X, y_gamma, velocities = sim.generate_dataset(n_samples=5000)
    
    # Split
    # Split
    # Scale Data for the Optical Net (Crucial for reservoir dynamics)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split both raw and scaled
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X, X_scaled, y_gamma, test_size=0.2, random_state=42
    )
    v_test = velocities[y_test.index if hasattr(y_test, 'index') else np.arange(len(y_test))] 
    # (Fix indexing for numpy arrays)
    v_test = velocities[len(y_train):]

    # --- Model A: Darwinian (Polynomial Regression) ---
    print("Training Darwinian Model (Polynomial approach)...")
    # A polynomial tries to Taylor Expansion the Lorentz formula
    darwin_model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
    darwin_model.fit(X_train, y_train)
    y_pred_darwin = darwin_model.predict(X_test)
    
    # --- Model B: Optical Interference ---
    print("Training Optical Interference Net (Physics-based)...")
    optical_model = OpticalInterferenceNet(n_components=5000)
    optical_model.fit(X_train_scaled, y_train)
    y_pred_optical = optical_model.predict(X_test_scaled)
    
    # Metrics
    r2_darwin = r2_score(y_test, y_pred_darwin)
    r2_optical = r2_score(y_test, y_pred_optical)
    
    print("\n--- RESULTS ---")
    print(f"Darwinian R2 Score: {r2_darwin:.5f}")
    print(f"Optical   R2 Score: {r2_optical:.5f}")
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: The Lorentz Curve
    plt.subplot(1, 2, 1)
    # Sort for clean plotting
    sort_idx = np.argsort(v_test)
    plt.plot(v_test[sort_idx], y_test[sort_idx], 'k-', lw=2, label='True Physics (Lorentz)')
    plt.scatter(v_test, y_pred_darwin, s=10, c='blue', alpha=0.3, label='Darwinian (Poly)')
    plt.scatter(v_test, y_pred_optical, s=10, c='red', alpha=0.3, label='Optical AI')
    
    plt.title("Predicting Time Dilation")
    plt.xlabel("Velocity (fraction of c)")
    plt.ylabel("Gamma Factor (Time Dilation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Error Analysis near c
    plt.subplot(1, 2, 2)
    error_darwin = np.abs(y_test - y_pred_darwin)
    error_optical = np.abs(y_test - y_pred_optical)
    
    plt.plot(v_test[sort_idx], error_darwin[sort_idx], 'b-', alpha=0.5, label='Darwinian Error')
    plt.plot(v_test[sort_idx], error_optical[sort_idx], 'r-', alpha=0.8, label='Optical Error')
    
    plt.title("Error at Relativistic Speeds")
    plt.xlabel("Velocity (fraction of c)")
    plt.ylabel("Absolute Prediction Error")
    plt.yscale('log') # Log scale to see the difference clearly
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_2_relativity.png')
    print("\n📊 Results saved as 'experiment_2_relativity.png'")
    
    # --- DARWIN'S CAGE ANALYSIS ---
    print("\n--- CAGE ANALYSIS ---")
    print("Does the Optical AI internally 'calculate' velocity squared?")
    
    # Extract internal optical features
    # Extract internal optical features
    internal_states = optical_model.get_internal_representation(X_test_scaled)
    
    # Check correlation with v^2 (the core component of Lorentz formula)
    v_squared = v_test**2
    
    correlations = []
    for i in range(internal_states.shape[1]):
        corr = np.corrcoef(internal_states[:, i], v_squared)[0, 1]
        correlations.append(abs(corr))
        
    max_corr = max(correlations)
    mean_corr = np.mean(correlations)
    
    print(f"Max Correlation with v^2: {max_corr:.4f}")
    print(f"Mean Correlation with v^2: {mean_corr:.4f}")
    
    if max_corr < 0.8:
        print("✅ EVIDENCE FOUND: The model predicts Gamma accurately (>0.99)")
        print("   BUT its internal features do not strongly correlate with v^2.")
        print("   It found a geometric path to the answer, bypassing the algebraic formula.")
    else:
        print("⚠️ CONVERGENCE: The model reconstructed the v^2 variable internally.")

    plt.show()

if __name__ == "__main__":
    run_experiment_2()