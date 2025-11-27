"""
Physics vs. Darwin: Experiment 4 (RIGOROUS VERSION 2)
The Transfer Test - Advanced Edition
------------------------------------

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

SCIENTIFIC DESIGN PRINCIPLES:
1. Both domains predict the SAME physical quantity (Characteristic Time, same units)
2. Both domains share a DEEP physical concept (exponential decay with damping)
3. Mathematical structure: τ = f(mass/damping) in both cases
4. Domains differ in physical context (mechanical vs electrical)
5. Includes negative control (unrelated physics)
6. Includes measurement noise for realism
7. Impartial evaluation criteria

Domain A: Mechanical Damped Oscillator
  - Predicts: Decay Time τ = 2m/γ (time for amplitude to drop to 1/e)
  - Input: [mass (kg), damping_coefficient (kg/s)]
  - Physics: F = -kx - γv (spring + linear drag)

Domain B: RC Circuit (Electrical Decay)
  - Predicts: Decay Time τ = RC (time for voltage to drop to 1/e)
  - Input: [resistance (Ω), capacitance (F)]
  - Physics: V(t) = V₀e^(-t/RC) (exponential decay)

Shared Deep Concept: Both follow exponential decay τ = (inertia/resistance)
  - Mechanical: τ = 2m/γ (mass/damping)
  - Electrical: τ = RC (resistance × capacitance, where C acts as "electrical inertia")
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 1. PHYSICS SIMULATORS ---
class DampedMechanicalOscillator:
    """
    Domain A: Mechanical Damped Oscillator
    Predicts: Decay Time τ = 2m/γ
    Input: [mass (kg), damping_coefficient (kg/s)]
    Output: Decay time (seconds)
    """
    def generate_dataset(self, n_samples=3000):
        np.random.seed(42)
        
        # Random parameters
        mass = np.random.uniform(0.1, 5.0, n_samples)        # Mass (kg)
        damping = np.random.uniform(0.1, 2.0, n_samples)     # Damping coefficient (kg/s)
        
        # Truth: Decay time τ = 2m/γ
        tau = 2 * mass / damping
        
        # Add 3% measurement noise (realistic experimental error)
        noise = np.random.normal(0, 0.03 * np.mean(tau), n_samples)
        tau = tau + noise
        # Ensure all values are positive (physical constraint)
        tau = np.maximum(tau, 0.01)
        
        X = np.column_stack((mass, damping))
        return X, tau

class RCCircuit:
    """
    Domain B: RC Circuit (Electrical Decay)
    Predicts: Decay Time τ = RC
    Input: [resistance (Ω), capacitance (F)]
    Output: Decay time (seconds) - SAME as Domain A
    
    NOTE: Same mathematical structure as Damped Oscillator:
    - Mechanical: τ = 2m/γ (inertia/resistance)
    - Electrical: τ = RC (where C is "electrical inertia" and R is "electrical resistance")
    """
    def generate_dataset(self, n_samples=1000):
        np.random.seed(123)
        
        # Random parameters - adjusted to match mechanical decay time scale
        # Mechanical: τ = 2m/γ, with m in [0.1, 5] and γ in [0.1, 2]
        # This gives τ ≈ 0.1 to 100 seconds
        # For RC: τ = RC, we want similar range
        # Let's use: R in [100, 10000] Ω, C in [0.00001, 0.01] F
        R = np.random.uniform(100, 10000, n_samples)         # Resistance (Ω)
        C = np.random.uniform(1e-5, 0.01, n_samples)        # Capacitance (F)
        
        # Truth: Decay time τ = RC
        tau = R * C
        
        # Add 3% measurement noise
        noise = np.random.normal(0, 0.03 * np.mean(tau), n_samples)
        tau = tau + noise
        # Ensure all values are positive (physical constraint)
        tau = np.maximum(tau, 0.01)
        
        X = np.column_stack((R, C))
        return X, tau

class UnrelatedDomain:
    """
    Negative Control: Completely unrelated physics
    Predicts: Total resistance in parallel circuit (using only 2 resistors to match input dimension)
    Structure: R_total = 1/(1/R1 + 1/R2) = R1*R2/(R1+R2)
    This has NO mathematical similarity to exponential decay.
    """
    def generate_dataset(self, n_samples=1000):
        np.random.seed(456)
        
        R1 = np.random.uniform(10, 1000, n_samples)
        R2 = np.random.uniform(10, 1000, n_samples)
        
        # Parallel resistance: R_total = 1/(1/R1 + 1/R2) = R1*R2/(R1+R2)
        R_total = (R1 * R2) / (R1 + R2)
        
        X = np.column_stack((R1, R2))
        return X, R_total

# --- 2. MODELS ---
class SimpleReservoir:
    """
    Baseline: Simple reservoir without feature engineering.
    This should learn domain-specific patterns and fail at transfer.
    """
    def __init__(self, n_features=2000):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=1.0)
        self.reservoir = None
        
    def _transform(self, X):
        if self.reservoir is None:
            np.random.seed(1)
            self.reservoir = np.random.randn(X.shape[1], self.n_features) * 0.1
        
        features = np.tanh(X @ self.reservoir)
        return features
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        features = self._transform(X_scaled)
        self.readout.fit(features, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        features = self._transform(X_scaled)
        return self.readout.predict(features)

class UniversalChaosModel:
    """
    Universal Model: Chaos with FFT mixing.
    Hypothesis: The chaotic interference can discover universal patterns
    like ratios and products that transfer across domains.
    
    CRITICAL: No explicit feature engineering - relies on chaos to discover patterns.
    """
    def __init__(self, n_features=4096, brightness=0.001):
        self.n_features = n_features
        self.brightness = brightness
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=0.1)
        self.reservoir = None
        
    def _chaos_transform(self, X):
        """
        Pure chaos transformation - no explicit feature engineering.
        The FFT mixing allows discovery of multiplicative relationships.
        """
        n_samples = X.shape[0]
        
        if self.reservoir is None:
            np.random.seed(999)
            self.reservoir = np.random.randn(X.shape[1], self.n_features)
            
        # Optical mixing
        optical_field = X @ self.reservoir
        optical_field *= self.brightness
        
        # FFT interference (discovers multiplicative patterns)
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

# --- 3. EXECUTION ---
def run_improved_experiment_4():
    print("🌀 EXPERIMENT 4 (RIGOROUS V2): THE TRANSFER TEST")
    print("=" * 60)
    print("Scientific Design:")
    print("  - Domain A: Damped Mechanical Oscillator (τ = 2m/γ)")
    print("  - Domain B: RC Circuit (τ = RC)")
    print("  - Both predict DECAY TIME (same units, same scale)")
    print("  - Shared concept: exponential decay with damping")
    print("  - Negative control: Parallel resistors (unrelated)")
    print("=" * 60)
    
    # 1. Generate Data
    print("\n[Domain A] Damped Mechanical Oscillator...")
    mech_sim = DampedMechanicalOscillator()
    X_mech, y_mech = mech_sim.generate_dataset(n_samples=3000)
    print(f"   Decay times: {np.min(y_mech):.3f} - {np.max(y_mech):.3f} seconds")
    
    print("[Domain B] RC Circuit...")
    rc_sim = RCCircuit()
    X_rc, y_rc = rc_sim.generate_dataset(n_samples=1000)
    print(f"   Decay times: {np.min(y_rc):.3f} - {np.max(y_rc):.3f} seconds")
    
    print("[Negative Control] Parallel Resistors...")
    res_sim = UnrelatedDomain()
    X_res, y_res = res_sim.generate_dataset(n_samples=1000)
    
    # Verify scales are similar
    scale_ratio = np.mean(y_rc) / np.mean(y_mech)
    print(f"\n[Scale Check] RC/Mechanical mean ratio: {scale_ratio:.3f}")
    if scale_ratio < 0.1 or scale_ratio > 10:
        print("   ⚠️ WARNING: Scale mismatch may affect transfer!")
    
    # 2. Train on Domain A ONLY
    print("\n[Training] Learning from Mechanical Oscillators ONLY...")
    
    model_simple = SimpleReservoir(n_features=2000)
    model_simple.fit(X_mech, y_mech)
    
    model_universal = UniversalChaosModel(n_features=4096, brightness=0.001)
    model_universal.fit(X_mech, y_mech)
    
    # 3. Within-Domain Test
    X_mech_train, X_mech_test, y_mech_train, y_mech_test = train_test_split(
        X_mech, y_mech, test_size=0.2, random_state=42
    )
    
    y_pred_simple_mech = model_simple.predict(X_mech_test)
    y_pred_universal_mech = model_universal.predict(X_mech_test)
    
    r2_simple_mech = r2_score(y_mech_test, y_pred_simple_mech)
    r2_universal_mech = r2_score(y_mech_test, y_pred_universal_mech)
    
    print(f"\n[Control: Within-Domain on Mechanical Oscillators]")
    print(f"  Simple R²:    {r2_simple_mech:.4f}")
    print(f"  Universal R²: {r2_universal_mech:.4f}")
    
    # 4. TRANSFER TEST: Mechanical → RC Circuit
    print(f"\n[TRANSFER TEST: Mechanical → RC Circuit]")
    
    y_pred_simple_rc = model_simple.predict(X_rc)
    y_pred_universal_rc = model_universal.predict(X_rc)
    
    r2_simple_rc = r2_score(y_rc, y_pred_simple_rc)
    r2_universal_rc = r2_score(y_rc, y_pred_universal_rc)
    
    print(f"  Simple Transfer R²:    {r2_simple_rc:.4f}")
    print(f"  Universal Transfer R²: {r2_universal_rc:.4f}")
    
    transfer_advantage = r2_universal_rc - r2_simple_rc
    print(f"  Transfer Advantage:    {transfer_advantage:.4f}")
    
    # 5. NEGATIVE CONTROL: Should FAIL on unrelated domain
    print(f"\n[NEGATIVE CONTROL: Mechanical → Parallel Resistors (should fail)]")
    
    y_pred_simple_res = model_simple.predict(X_res)
    y_pred_universal_res = model_universal.predict(X_res)
    
    r2_simple_res = r2_score(y_res, y_pred_simple_res)
    r2_universal_res = r2_score(y_res, y_pred_universal_res)
    
    print(f"  Simple R² (should be < 0): {r2_simple_res:.4f}")
    print(f"  Universal R² (should be < 0): {r2_universal_res:.4f}")
    
    # 6. VERDICT (Impartial)
    print(f"\n--- VERDICT ---")
    
    if r2_universal_rc > 0.7 and r2_simple_rc < 0.3:
        print(f"✅ CAGE BROKEN: Universal model transfers ({r2_universal_rc:.3f}), Simple fails ({r2_simple_rc:.3f})")
    elif r2_universal_rc > 0.6 and r2_simple_rc < 0.5 and transfer_advantage > 0.15:
        print(f"🟡 PARTIAL SUCCESS: Universal has significant transfer advantage ({transfer_advantage:.3f})")
    elif transfer_advantage > 0.05:
        print(f"🟡 MARGINAL ADVANTAGE: Universal performs slightly better ({transfer_advantage:.3f})")
    else:
        print(f"❌ NO CLEAR ADVANTAGE: Both models perform similarly in transfer")
    
    # Check negative control
    if r2_simple_res > 0.1 or r2_universal_res > 0.1:
        print(f"⚠️ WARNING: Models show unexpected performance on negative control!")
    else:
        print(f"✅ Negative control passed: Both models correctly fail on unrelated domain")
    
    # 7. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Simple Model
    axes[0,0].scatter(y_mech_test, y_pred_simple_mech, alpha=0.3, s=5)
    axes[0,0].plot([min(y_mech_test), max(y_mech_test)], [min(y_mech_test), max(y_mech_test)], 'k--')
    axes[0,0].set_title(f"Simple: Within-Domain\nR²={r2_simple_mech:.3f}")
    axes[0,0].set_xlabel("True Decay Time (s)")
    axes[0,0].set_ylabel("Predicted (s)")
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].scatter(y_rc, y_pred_simple_rc, alpha=0.3, s=5, c='orange')
    axes[0,1].plot([min(y_rc), max(y_rc)], [min(y_rc), max(y_rc)], 'k--')
    axes[0,1].set_title(f"Simple: Transfer\nR²={r2_simple_rc:.3f}")
    axes[0,1].set_xlabel("True Decay Time (s)")
    axes[0,1].set_ylabel("Predicted (s)")
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].scatter(y_res, y_pred_simple_res, alpha=0.3, s=5, c='red')
    axes[0,2].plot([min(y_res), max(y_res)], [min(y_res), max(y_res)], 'k--')
    axes[0,2].set_title(f"Simple: Negative Control\nR²={r2_simple_res:.3f}")
    axes[0,2].set_xlabel("True Resistance (Ω)")
    axes[0,2].set_ylabel("Predicted (Ω)")
    axes[0,2].grid(True, alpha=0.3)
    
    # Row 2: Universal Model
    axes[1,0].scatter(y_mech_test, y_pred_universal_mech, alpha=0.3, s=5)
    axes[1,0].plot([min(y_mech_test), max(y_mech_test)], [min(y_mech_test), max(y_mech_test)], 'k--')
    axes[1,0].set_title(f"Universal: Within-Domain\nR²={r2_universal_mech:.3f}")
    axes[1,0].set_xlabel("True Decay Time (s)")
    axes[1,0].set_ylabel("Predicted (s)")
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter(y_rc, y_pred_universal_rc, alpha=0.3, s=5, c='green')
    axes[1,1].plot([min(y_rc), max(y_rc)], [min(y_rc), max(y_rc)], 'k--')
    axes[1,1].set_title(f"Universal: Transfer\nR²={r2_universal_rc:.3f}")
    axes[1,1].set_xlabel("True Decay Time (s)")
    axes[1,1].set_ylabel("Predicted (s)")
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].scatter(y_res, y_pred_universal_res, alpha=0.3, s=5, c='gray')
    axes[1,2].plot([min(y_res), max(y_res)], [min(y_res), max(y_res)], 'k--')
    axes[1,2].set_title(f"Universal: Negative Control\nR²={r2_universal_res:.3f}")
    axes[1,2].set_xlabel("True Resistance (Ω)")
    axes[1,2].set_ylabel("Predicted (Ω)")
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_4_rigorous.png', dpi=150)
    print("\n📊 Results saved as 'experiment_4_rigorous.png'")
    plt.show()

if __name__ == "__main__":
    run_improved_experiment_4()
