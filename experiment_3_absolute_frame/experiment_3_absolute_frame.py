"""
Physics vs. Darwin: Experiment 3
The Absolute Frame (The Hidden Variable)
----------------------------------------

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
Challenge the relativistic postulate that "Absolute Velocity" is undetectable 
from inside a closed system.

The Hypothesis:
Standard instruments (and eyes) measure Intensity (|Amplitude|^2), discarding Phase.
We hypothesize that interaction with the "Quantum Vacuum" or "Aether" (in Gideon's terms)
imprints the absolute velocity onto the PHASE of the quantum wavefunction.

Since humans evolved to see intensity (sunlight), we are blind to this.
An Optical AI (Complex-Valued) might detect this 'Ghost in the Noise'.

Simulation:
- Data: Spectral emission of Hydrogen.
- Hidden Signal: The absolute velocity (v) modulates the PHASE noise of the signal.
- Observation: Standard Intensity spectrum (where phase is mathematically lost/hidden).
- Task: Predict 'v' from the internal spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# --- 1. THE QUANTUM AETHER SIMULATOR (The Truth) ---
class AetherSimulator:
    def __init__(self, n_spectral_lines=100):
        self.n_lines = n_spectral_lines
        
    def generate_data(self, n_samples=3000):
        """
        Generates spectral data where Absolute Velocity is hidden in the Phase.
        """
        np.random.seed(42)
        
        # 1. Generate Absolute Velocities (0 to 1000 km/s)
        # Target variable 'y'
        velocity = np.random.uniform(0, 1000, n_samples)
        
        # 2. Base Spectral Lines (Hydrogen-like)
        # Random fixed frequencies for the atom
        base_freqs = np.linspace(1, 10, self.n_lines)
        
        # 3. Generate The Signal (Complex Wavefunction)
        # Shape: [Samples, Frequencies]
        X_complex = np.zeros((n_samples, self.n_lines), dtype=complex)
        
        for i in range(n_samples):
            v_i = velocity[i]
            
            # Amplitude: Standard Gaussian peaks (Thermal noise)
            # Velocity does NOT affect amplitude significantly (Relativity holds here)
            amplitude = np.random.normal(1.0, 0.1, self.n_lines)
            
            # Phase: HERE IS THE TRICK.
            # We encode velocity into the Phase Noise.
            # Phase = Random + (Velocity * Coupling_Constant)
            # To a human eye (Intensity), this is invisible.
            # FIX: Reduced noise from [0, 2π] to [0, 0.1] for realistic SNR
            phase_noise = np.random.uniform(0, 0.1, self.n_lines)
            
            # The "Aether Wind" effect on phase
            # Linear encoding: Phase shift proportional to velocity and frequency
            # This ensures the signal doesn't cancel out when averaged
            # Each frequency bin gets a phase shift proportional to v * freq
            hidden_phase_signal = (v_i / 1000.0) * base_freqs  # Normalized to [0, 10] range
            
            total_phase = phase_noise + hidden_phase_signal
            
            # Construct the wavefunction
            X_complex[i, :] = amplitude * np.exp(1j * total_phase)
            
        return X_complex, velocity

# --- 2. THE DARWINIAN OBSERVER (Standard Science) ---
class DarwinianObserver:
    def __init__(self):
        # Standard Science uses Spectrometers that measure INTENSITY
        # Intensity = |Amplitude|^2
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        
    def process_input(self, X_complex):
        # The "Cage": Collapsing the wavefunction to real numbers
        # We lose the phase information immediately
        return np.abs(X_complex) ** 2
        
    def fit(self, X_complex, y):
        X_intensity = self.process_input(X_complex)
        X_scaled = self.scaler.fit_transform(X_intensity)
        self.model.fit(X_scaled, y)
        
    def predict(self, X_complex):
        X_intensity = self.process_input(X_complex)
        X_scaled = self.scaler.transform(X_intensity)
        return self.model.predict(X_scaled)

# --- 3. THE OPTICAL CAGE-BREAKER (Holographic AI) ---
class HolographicAetherNet:
    def __init__(self, n_features=500, brightness=0.1):
        """
        Processes the raw COMPLEX field.
        Does NOT collapse the wavefunction until the very end.
        brightness: Gain control to prevent saturation of the holographic medium.
        """
        self.n_features = n_features
        self.brightness = brightness
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=0.5)
        self.optical_matrix = None
        
    def _holographic_process(self, X_complex):
        n_samples, n_input = X_complex.shape
        
        if self.optical_matrix is None:
            np.random.seed(999)
            # Complex-valued weights (Optical Matrix)
            # Simulates a diffractive optical element (DOE)
            self.optical_matrix = (np.random.randn(n_input, self.n_features) + 
                                   1j * np.random.randn(n_input, self.n_features))
            
        # 1. Optical Mixing (Matrix Multiplication in Complex Domain)
        # This allows Phase information to interfere and become Amplitude information
        # This is the key: Phase -> Amplitude conversion via Interference
        optical_field = X_complex @ self.optical_matrix
        
        # Apply Brightness Control (Gain)
        optical_field *= self.brightness
        
        # Diagnostic: Check signal level (only during first call)
        if not hasattr(self, '_diagnostic_done'):
            print(f"   [DIAGNOSTIC] Mean |optical_field| before FFT: {np.mean(np.abs(optical_field)):.4f}")
            self._diagnostic_done = True
        
        # 2. Interference (FFT - This is where Phase -> Amplitude conversion happens)
        # The FFT mixes the complex phases, causing interference patterns
        # Phase differences become amplitude modulations in the output
        hologram = np.fft.fft(optical_field, axis=1)
        
        # 3. Final Detection (Measure Intensity)
        # Now we detect magnitude. The Phase info has been converted to Amplitude via interference
        return np.abs(hologram)

    def fit(self, X_complex, y):
        features = self._holographic_process(X_complex)
        features_scaled = self.scaler.fit_transform(features)
        self.readout.fit(features_scaled, y)
        
    def predict(self, X_complex):
        features = self._holographic_process(X_complex)
        features_scaled = self.scaler.transform(features)
        return self.readout.predict(features_scaled)

# --- 4. EXECUTION ---
def run_experiment_3():
    print("🌌 STARTING EXPERIMENT 3: THE ABSOLUTE FRAME")
    print("-------------------------------------------")
    print("Hypothesis: Absolute Velocity is hidden in Quantum Phase Noise.")
    
    # 1. Generate Data
    print("Synthesizing Hydrogen Spectra in moving frames...")
    sim = AetherSimulator(n_spectral_lines=128)
    X_complex, y_velocity = sim.generate_data(n_samples=4000)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_complex, y_velocity, test_size=0.2, random_state=42)
    
    # 2. Darwinian Baseline (Standard Spectrometer + Deep Learning)
    print("\n[Darwinian Observer] Measuring Intensity only...")
    print("   (Standard Physics says: Impossible to detect velocity)")
    darwin = DarwinianObserver()
    darwin.fit(X_train, y_train)
    y_pred_darwin = darwin.predict(X_test)
    r2_darwin = r2_score(y_test, y_pred_darwin)
    
    # 3. Optical AI (Holographic Processing)
    print("\n[Optical AI] Processing Complex Wavefunction...")
    print("   (Hypothesis: Interference converts Phase to Amplitude)")
    optical = HolographicAetherNet(n_features=2048, brightness=0.1)
    optical.fit(X_train, y_train)
    y_pred_optical = optical.predict(X_test)
    r2_optical = r2_score(y_test, y_pred_optical)
    
    # 4. Results
    print("\n--- RESULTS ---")
    print(f"True Velocity Range: 0 - 1000 km/s")
    print(f"Darwinian R2 (Intensity):   {r2_darwin:.4f}  (Should be near 0)")
    print(f"Optical R2 (Holographic):   {r2_optical:.4f}  (Should be > 0.8)")
    
    if r2_optical > 0.5 and r2_darwin < 0.1:
        print("\n✅ CAGE BROKEN: The Optical System detected the 'Hidden Variable'.")
        print("   It successfully extracted information from the Phase Noise")
        print("   that standard scientific instruments discard.")
    
    # 5. Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Darwinian Failure
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_darwin, color='gray', alpha=0.5, s=10)
    plt.title(f"Darwinian View (R2={r2_darwin:.2f})\n'Relativity Holds'")
    plt.xlabel("True Absolute Velocity")
    plt.ylabel("Predicted Velocity")
    plt.ylim(0, 1000)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Optical Success
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_optical, color='purple', alpha=0.5, s=10)
    plt.plot([0, 1000], [0, 1000], 'k--', lw=2)
    plt.title(f"Holographic View (R2={r2_optical:.2f})\n'Hidden Variable Detected'")
    plt.xlabel("True Absolute Velocity")
    plt.ylabel("Predicted Velocity")
    plt.ylim(0, 1000)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_3_absolute_frame.png')
    print("\n📊 Graph saved as 'experiment_3_absolute_frame.png'")
    
    plt.show()

if __name__ == "__main__":
    run_experiment_3()