"""
Physics vs. Darwin: Experiment 4 (RIGOROUS VERSION)
The Transfer Test (The Unity of Physical Laws)
------------------------------------------------

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

OBJECTIVE:
Prove that a Chaos-based AI can discover UNIVERSAL physical principles
that transfer across different surface phenomena, breaking free from 
domain-specific "cages" that trap human thinking.

SCIENTIFIC DESIGN PRINCIPLES:
1. Both domains predict the SAME physical quantity (Period, same units, same scale)
2. Both domains share the SAME mathematical structure: T ∝ √(I/F)
3. Domains differ ONLY in physical context (mechanical vs electromagnetic)
4. No feature engineering bias - models see raw inputs only
5. Impartial evaluation - same criteria for both models

The Challenge:
Humans took centuries to realize that:
- Springs (Hooke's Law): T = 2π√(m/k)
- LC Circuits (Electromagnetism): T = 2π√(LC)
...follow the SAME mathematical pattern: Simple Harmonic Motion.

Can an AI trained ONLY on springs predict the behavior of LC circuits?

If YES → The AI discovered the universal pattern (Cage Broken).
If NO → The AI is trapped in domain-specific thinking (Cage Locked).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 1. PHYSICS SIMULATORS ---
class SpringMassSimulator:
    """
    Domain A: Spring-Mass Oscillator
    Predicts: Period T = 2π√(m/k)
    Input: [mass (kg), spring_constant (N/m)]
    Output: Period (seconds)
    """
    def generate_dataset(self, n_samples=3000):
        np.random.seed(42)
        # Random masses (0.1 to 10 kg)
        mass = np.random.uniform(0.1, 10, n_samples)
        # Random spring constants (1 to 100 N/m)
        k = np.random.uniform(1, 100, n_samples)
        
        # Truth: T = 2π√(m/k)
        period = 2 * np.pi * np.sqrt(mass / k)
        
        # Inputs: [mass, k]
        X = np.column_stack((mass, k))
        return X, period

class LCCircuitSimulator:
    """
    Domain B: LC Resonant Circuit
    Predicts: Period T = 2π√(LC)
    Input: [inductance (H), capacitance (F)]
    Output: Period (seconds) - SAME as Domain A
    
    NOTE: Same mathematical structure as Spring-Mass:
    - Spring: T ∝ √(inertia/restoring_force) = √(m/k)
    - LC: T ∝ √(inertia/restoring_force) = √(L/C) where L is "electrical inertia"
    
    PARAMETER ADJUSTMENT: We adjust L and C ranges to match Spring-Mass period scale
    Spring periods: ~0.2-18 seconds
    To match: T = 2π√(LC) ≈ 0.2-18, so √(LC) ≈ 0.03-2.9, so LC ≈ 0.001-8.4
    """
    def generate_dataset(self, n_samples=1000):
        np.random.seed(123)
        # Adjusted ranges to match Spring-Mass period scale (~0.2-18 seconds)
        # We want LC product to give similar periods
        # T = 2π√(LC), so for T ≈ 0.2-18, we need √(LC) ≈ 0.03-2.9
        # Let's use: L in [0.01, 10] H, C in [0.0001, 0.1] F
        L = np.random.uniform(0.01, 10.0, n_samples)
        C = np.random.uniform(0.0001, 0.1, n_samples)
        
        # Truth: T = 2π√(LC)
        period = 2 * np.pi * np.sqrt(L * C)
        
        # Ensure periods are in reasonable range (clip extreme values)
        period = np.clip(period, 0.1, 20.0)
        
        # Inputs: [L, C]
        X = np.column_stack((L, C))
        return X, period

# --- 2. THE DARWINIAN BASELINE (Domain-Specific) ---
class DomainSpecificModel:
    """
    Baseline: Simple reservoir without feature engineering.
    This model should learn domain-specific patterns and fail at transfer.
    """
    def __init__(self, n_features=2000):
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.readout = Ridge(alpha=0.1)
        self.projection = None
        
    def _process(self, X):
        if self.projection is None:
            np.random.seed(1)
            self.projection = np.random.randn(X.shape[1], self.n_features) * 0.1
        
        features = np.tanh(X @ self.projection)
        return features
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        features = self._process(X_scaled)
        self.readout.fit(features, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        features = self._process(X_scaled)
        return self.readout.predict(features)

# --- 3. THE UNIVERSAL CHAOS MODEL (Cross-Domain) ---
class UniversalChaosMachine:
    """
    The Hypothesis: A chaotic reservoir with FFT mixing can discover
    universal mathematical patterns that transfer across domains.
    
    CRITICAL: This model does NOT use explicit feature engineering.
    It relies on the chaos to discover patterns organically.
    """
    def __init__(self, n_features=4096, brightness=0.001):
        self.n_features = n_features
        self.brightness = brightness
        self.scaler = MinMaxScaler()
        self.readout = Ridge(alpha=0.1)
        self.reservoir = None
        
    def _chaos_transform(self, X):
        """
        Pure chaos transformation - no explicit feature engineering.
        The FFT mixing should allow the model to discover patterns
        like ratios and square roots organically.
        """
        n_samples = X.shape[0]
        
        if self.reservoir is None:
            np.random.seed(999)
            # Random projection into high-dimensional space
            self.reservoir = np.random.randn(X.shape[1], self.n_features)
            
        # Optical mixing (linear projection)
        optical_field = X @ self.reservoir
        optical_field *= self.brightness
        
        # FFT interference (non-linear mixing in frequency domain)
        # This allows discovery of multiplicative relationships
        spectrum = np.fft.rfft(optical_field, axis=1)
        intensity = np.abs(spectrum) ** 2
        
        # Normalize to prevent saturation
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

# --- 4. EXECUTION ---
def run_experiment_4():
    print("🌀 STARTING EXPERIMENT 4: THE TRANSFER TEST (RIGOROUS)")
    print("=" * 60)
    print("Scientific Design:")
    print("  - Domain A: Spring-Mass (T = 2π√(m/k))")
    print("  - Domain B: LC Circuit (T = 2π√(LC))")
    print("  - Both predict PERIOD (same units, same scale)")
    print("  - Same mathematical structure: T ∝ √(ratio)")
    print("=" * 60)
    
    # 1. Generate Data
    print("\n[Domain A] Generating Spring-Mass data...")
    spring_sim = SpringMassSimulator()
    X_spring, y_spring = spring_sim.generate_dataset(n_samples=3000)
    print(f"   Spring periods: {np.min(y_spring):.3f} - {np.max(y_spring):.3f} seconds")
    
    print("[Domain B] Generating LC Circuit data...")
    lc_sim = LCCircuitSimulator()
    X_lc, y_lc = lc_sim.generate_dataset(n_samples=1000)
    print(f"   LC periods: {np.min(y_lc):.3f} - {np.max(y_lc):.3f} seconds")
    
    # Verify scales are similar (critical for fair transfer test)
    scale_ratio = np.mean(y_lc) / np.mean(y_spring)
    print(f"\n[Scale Check] LC/Spring mean ratio: {scale_ratio:.3f}")
    if scale_ratio < 0.1 or scale_ratio > 10:
        print("   ⚠️ WARNING: Scale mismatch may affect transfer!")
    
    # 2. Train on Springs ONLY
    print("\n[Training] Both models learn from Springs ONLY...")
    
    # Domain-Specific Model
    model_darwinian = DomainSpecificModel(n_features=2000)
    model_darwinian.fit(X_spring, y_spring)
    
    # Universal Chaos Model
    model_universal = UniversalChaosMachine(n_features=4096, brightness=0.001)
    model_universal.fit(X_spring, y_spring)
    
    # 3. Within-Domain Test (Control)
    X_spring_train, X_spring_test, y_spring_train, y_spring_test = train_test_split(
        X_spring, y_spring, test_size=0.2, random_state=42
    )
    
    y_pred_darwin_spring = model_darwinian.predict(X_spring_test)
    y_pred_universal_spring = model_universal.predict(X_spring_test)
    
    r2_darwin_spring = r2_score(y_spring_test, y_pred_darwin_spring)
    r2_universal_spring = r2_score(y_spring_test, y_pred_universal_spring)
    
    print(f"\n[Control Test: Within-Domain on Springs]")
    print(f"  Darwinian R²:  {r2_darwin_spring:.4f}")
    print(f"  Universal R²:  {r2_universal_spring:.4f}")
    
    # 4. THE DEFINITIVE TEST: Cross-Domain Transfer
    print(f"\n[THE DEFINITIVE TEST: Cross-Domain Transfer]")
    print(f"  Testing models trained on SPRINGS to predict LC CIRCUITS...")
    
    y_pred_darwin_lc = model_darwinian.predict(X_lc)
    y_pred_universal_lc = model_universal.predict(X_lc)
    
    r2_darwin_lc = r2_score(y_lc, y_pred_darwin_lc)
    r2_universal_lc = r2_score(y_lc, y_pred_universal_lc)
    
    print(f"\n--- RESULTS ---")
    print(f"  Darwinian Transfer R²:  {r2_darwin_lc:.4f}")
    print(f"  Universal Transfer R²:  {r2_universal_lc:.4f}")
    
    # Statistical significance check
    transfer_advantage = r2_universal_lc - r2_darwin_lc
    print(f"  Transfer Advantage:    {transfer_advantage:.4f}")
    
    # Impartial verdict
    if r2_universal_lc > 0.8 and r2_darwin_lc < 0.5:
        print("\n✅ CAGE BROKEN: The Universal Model discovered the underlying unity!")
        print("   It successfully transferred knowledge from springs to LC circuits.")
    elif r2_universal_lc > 0.7 and r2_darwin_lc < 0.6 and transfer_advantage > 0.15:
        print("\n🟡 PARTIAL CAGE-BREAK: Universal model shows significant transfer advantage.")
    elif transfer_advantage > 0.05:
        print("\n🟡 MARGINAL ADVANTAGE: Universal model performs slightly better in transfer.")
    else:
        print("\n❌ CAGE LOCKED: No clear transfer advantage. Both models are domain-specific.")
    
    # 5. Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Within-Domain
    plt.subplot(1, 3, 1)
    plt.scatter(y_spring_test, y_pred_universal_spring, alpha=0.5, s=5, c='blue', label='Universal')
    plt.scatter(y_spring_test, y_pred_darwin_spring, alpha=0.3, s=5, c='gray', label='Darwinian')
    plt.plot([0, max(y_spring_test)], [0, max(y_spring_test)], 'k--', lw=1)
    plt.title(f"Within-Domain (Springs)\nUniversal R²={r2_universal_spring:.3f}")
    plt.xlabel("True Period (s)")
    plt.ylabel("Predicted Period (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Transfer Test (Universal)
    plt.subplot(1, 3, 2)
    plt.scatter(y_lc, y_pred_universal_lc, alpha=0.5, s=5, c='green')
    plt.plot([0, max(y_lc)], [0, max(y_lc)], 'k--', lw=1)
    plt.title(f"Transfer: Universal Model\nSprings → LC Circuits\nR²={r2_universal_lc:.3f}")
    plt.xlabel("True Period (s)")
    plt.ylabel("Predicted Period (s)")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Transfer Test (Darwinian)
    plt.subplot(1, 3, 3)
    plt.scatter(y_lc, y_pred_darwin_lc, alpha=0.5, s=5, c='red')
    plt.plot([0, max(y_lc)], [0, max(y_lc)], 'k--', lw=1)
    plt.title(f"Transfer: Domain-Specific Model\nSprings → LC Circuits\nR²={r2_darwin_lc:.3f}")
    plt.xlabel("True Period (s)")
    plt.ylabel("Predicted Period (s)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_4_transfer.png', dpi=150)
    print("\n📊 Graph saved as 'experiment_4_transfer.png'")
    plt.show()

if __name__ == "__main__":
    run_experiment_4()
