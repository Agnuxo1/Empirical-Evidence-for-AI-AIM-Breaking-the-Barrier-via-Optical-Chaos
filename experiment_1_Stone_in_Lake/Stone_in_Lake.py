"""
Physics vs. Darwin: Experiment 1
The Chaotic Reservoir (The Stone in the Lake)
---------------------------------------------

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
Demonstrate that a "Chaotic Optical Reservoir" (a system with fixed, random, 
disordered connections simulating wave interference) can predict the landing 
spot of a projectile better or equal to a Newtonian equation solver, 
without having any concept of 'gravity', 'velocity', or 'angles'.

The Model:
1. Input: Initial Velocity (v0) and Angle (theta)
2. Optical Layer: Projects input into a high-dimensional Complex domain (simulating a laser through a diffuser).
3. Interference: Applies FFT (Fast Fourier Transform) to simulate wave propagation.
4. Detection: Measures intensity (Magnitude^2).
5. Readout: A simple linear regression trains to map the interference pattern to the result.

There is NO backpropagation in the hidden layer. The "Physics" is fixed chaos.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import scipy.fft

# --- 1. THE TRUTH (Physics Simulator) ---
class PhysicsSimulator:
    def __init__(self, g=9.81):
        self.g = g
        
    def calculate_trajectory(self, v0, angle_deg):
        """Returns the landing distance (R) based on Newtonian Physics."""
        theta = np.radians(angle_deg)
        # Formula: R = (v^2 * sin(2*theta)) / g
        distance = (v0**2 * np.sin(2*theta)) / self.g
        return distance

    def generate_dataset(self, n_samples=2000):
        """Generates random throws."""
        np.random.seed(42)
        # Random velocities between 10 and 100 m/s
        v0 = np.random.uniform(10, 100, n_samples)
        # Random angles between 5 and 85 degrees
        angle = np.random.uniform(5, 85, n_samples)
        
        # Calculate Truth
        y = self.calculate_trajectory(v0, angle)
        
        # Stack inputs (X)
        X = np.column_stack((v0, angle))
        return X, y

# --- 2. THE CAGE BREAKER (Optical Chaos Model) ---
class OpticalChaosMachine:
    def __init__(self, n_features=2000, brightness=1.0):
        """
        n_features: Number of optical paths (simulated neurons/pixels).
        brightness: Scaling factor for the signal.
        """
        self.n_features = n_features
        self.brightness = brightness
        self.readout = Ridge(alpha=0.1) # Linear readout (cheap training)
        self.optical_matrix = None # This will be our fixed "Diffuser"
        
    def _optical_interference(self, X):
        """
        Simulates light passing through a chaotic medium (Random Matrix)
        and interfering (FFT).
        """
        n_samples, n_input = X.shape
        
        # 1. Initialize the chaotic medium (The Diffuser) if not exists
        if self.optical_matrix is None:
            np.random.seed(1337) # Fixed seed for reproducibility of the "Chaos"
            # Random complex weights to simulate phase shifts
            self.optical_matrix = np.random.normal(0, 1, (n_input, self.n_features))
            
        # 2. Projection (Light enters the medium)
        # X shape: [Samples, 2] -> Projected: [Samples, Features]
        light_field = X @ self.optical_matrix
        
        # 3. Wave Propagation (FFT)
        # This simulates the physical mixing of waves in the frequency domain
        interference_pattern = np.fft.rfft(light_field, axis=1)
        
        # 4. Detection (Intensity)
        # Detectors see Magnitude squared (Real numbers)
        intensity = np.abs(interference_pattern)**2
        
        # Normalize (simulating sensor saturation)
        intensity = np.tanh(intensity * self.brightness)
        
        return intensity

    def fit(self, X, y):
        # Transform inputs into "Optical Speckle Patterns"
        X_optical = self._optical_interference(X)
        # Train ONLY the readout (The brain interpreting the pattern)
        self.readout.fit(X_optical, y)
        
    def predict(self, X):
        X_optical = self._optical_interference(X)
        return self.readout.predict(X_optical)

    def get_internal_state(self, X):
        """Helper to visualize what the machine 'sees'"""
        return self._optical_interference(X)

# --- 3. THE DARWINIAN BASELINE (Standard Interpretation) ---
class DarwinianModel:
    def __init__(self):
        # Approximates with a polynomial (tries to learn the formula)
        self.poly = PolynomialFeatures(degree=2)
        self.model = LinearRegression()
        
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

# --- 4. EXECUTION & ANALYSIS ---
def run_experiment():
    print("🧪 STARTING EXPERIMENT 1: THE CHAOTIC RESERVOIR")
    print("-----------------------------------------------")
    
    # 1. Generate Data
    print("Generating 2,000 projectile trajectories...")
    sim = PhysicsSimulator()
    X, y = sim.generate_dataset(n_samples=2000)
    
    # Split Data

    # Scale Data for the Optical Chaos Machine (Crucial to avoid saturation)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Data (We split both raw and scaled data to keep them aligned)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X, X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 2. Train Darwinian Model (The Control)
    print("Training Darwinian Baseline (Polynomial Regression)...")
    darwin = DarwinianModel()
    darwin.fit(X_train, y_train)
    y_pred_darwin = darwin.predict(X_test)
    r2_darwin = r2_score(y_test, y_pred_darwin)
    
    # 3. Train Optical Chaos Model (The Experiment)
    print("Training Optical Chaos Machine (FFT Interference)...")
    # Note: High number of features = High definition hologram
    # We reduce brightness to 0.001 because FFT amplifies the signal significantly (Energy conservation).
    # We want to avoid saturating the tanh function (Sensor saturation).
    chaos_model = OpticalChaosMachine(n_features=4096, brightness=0.001) 
    
    # Check saturation levels on a small batch
    sample_intensity = chaos_model._optical_interference(X_train_scaled[:10])
    print(f"DEBUG: Mean Optical Intensity (after tanh): {np.mean(sample_intensity):.4f}")
    
    chaos_model.fit(X_train_scaled, y_train)
    y_pred_chaos = chaos_model.predict(X_test_scaled)
    r2_chaos = r2_score(y_test, y_pred_chaos)
    
    # 4. Print Results
    print("\n--- RESULTS ---")
    print(f"Newtonian Physics (Truth) Variance: {np.var(y_test):.2f}")
    print(f"Darwinian Baseline R2 Score:        {r2_darwin:.4f}")
    print(f"Optical Chaos Model R2 Score:       {r2_chaos:.4f}")
    
    if r2_chaos > 0.95:
        print("\n✅ SUCCESS: The Chaotic System predicted physics with High Precision!")
    
    # 5. Visualization
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Prediction Correlation
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_darwin, alpha=0.3, label='Darwinian (Poly)', color='blue')
    plt.scatter(y_test, y_pred_chaos, alpha=0.3, label='Optical Chaos', color='red')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'k--', lw=2, label='Perfect Prediction')
    plt.title("Prediction Accuracy: Chaos vs Logic")
    plt.xlabel("True Landing Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: The "Alien" View (Internal State)
    # Visualize the first 50 features of the optical state for sorted distances
    plt.subplot(1, 2, 2)
    
    # Sort data by distance to see if patterns emerge
    sort_idx = np.argsort(y_test)
    sorted_X_scaled = X_test_scaled[sort_idx]
    sorted_y = y_test[sort_idx]
    
    # Get internal chaotic states
    states = chaos_model.get_internal_state(sorted_X_scaled)
    
    # Plot heatmap of the first 100 chaotic features
    plt.imshow(states[:, :100].T, aspect='auto', cmap='inferno', interpolation='nearest')
    plt.colorbar(label="Optical Intensity")
    plt.title("Inside the Cage-Breaker\n(Internal Interference Pattern)")
    plt.xlabel("Test Samples (Sorted by Distance)")
    plt.ylabel("Optical Feature Channel (0-100)")
    
    plt.tight_layout()
    plt.savefig('experiment_1_results.png')
    print("\n📊 Graph saved as 'experiment_1_results.png'")
    plt.show()

    # 6. Darwin's Cage Metric (Correlation Check)
    print("\n--- DARWIN'S CAGE CHECK ---")
    print("Checking if internal features correlate with human concepts (Velocity, Angle)...")
    
    # Get features for test set
    # Get features for test set
    features = chaos_model.get_internal_state(X_test_scaled)
    
    # Calculate correlation of EVERY feature with Velocity and Angle
    corrs_v = [np.corrcoef(features[:, i], X_test[:, 0])[0,1] for i in range(features.shape[1])]
    corrs_a = [np.corrcoef(features[:, i], X_test[:, 1])[0,1] for i in range(features.shape[1])]
    
    max_corr_v = np.max(np.abs(corrs_v))
    max_corr_a = np.max(np.abs(corrs_a))
    
    print(f"Max Correlation with Velocity: {max_corr_v:.4f}")
    print(f"Max Correlation with Angle:    {max_corr_a:.4f}")
    
    if max_corr_v < 0.5 and max_corr_a < 0.5:
        print("🔓 CAGE STATUS: BROKEN. No single internal feature represents 'Speed' or 'Angle'.")
        print("   The knowledge is distributed holographically across the interference pattern.")
    else:
        print("🔒 CAGE STATUS: LOCKED. Some features still mimic human variables.")

if __name__ == "__main__":
    run_experiment()