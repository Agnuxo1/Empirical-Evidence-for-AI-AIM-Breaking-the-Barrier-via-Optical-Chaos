"""
BENCHMARK CERTIFICATION: Experiment 3 (The Absolute Frame)
----------------------------------------------------------
Objective:
Verify if the Optical AI is genuinely detecting "Absolute Velocity" from 
the Quantum Phase Noise, or if it's finding some other correlation.

Tests:
1. STANDARD ACCURACY: R2 Score. (Target: > 0.8)
2. PHASE SCRAMBLING (The "Null Hypothesis"): 
   - We take the SAME data, but randomize the phase.
   - If the AI relies on Phase, accuracy should drop to ~0.
   - If the AI relies on Amplitude (cheating), accuracy will remain high.
3. EXTRAPOLATION: Train on v < 700, Test on v > 700.

Author: System Auditor
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import from main script
from experiment_3_absolute_frame import HolographicAetherNet, AetherSimulator, DarwinianObserver

def run_benchmark_3():
    print("🌌 STARTING IMPARTIAL BENCHMARK: EXPERIMENT 3")
    print("==============================================")
    
    # --- SETUP ---
    sim = AetherSimulator(n_spectral_lines=128)
    X_complex, y = sim.generate_data(n_samples=4000)
    
    X_train, X_test, y_train, y_test = train_test_split(X_complex, y, test_size=0.2, random_state=42)
    
    # --- TEST 1: STANDARD ACCURACY ---
    print("\n[TEST 1] STANDARD DETECTION (Holographic)")
    model = HolographicAetherNet(n_features=2048, brightness=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_std = r2_score(y_test, y_pred)
    
    print(f"   R2 Score: {r2_std:.5f}")
    if r2_std > 0.8:
        print("   ✅ PASS: Hidden variable detected.")
    else:
        print("   ❌ FAIL: Signal lost in noise.")

    # --- TEST 2: PHASE SCRAMBLING (The Litmus Test) ---
    print("\n[TEST 2] PHASE SCRAMBLING (Destroying the Hidden Signal)")
    # We keep Amplitudes exactly the same, but randomize Phases
    # This destroys the "Aether Wind" signal but keeps the "Thermal Noise"
    
    X_test_scrambled = np.abs(X_test) * np.exp(1j * np.random.uniform(0, 2*np.pi, X_test.shape))
    
    y_pred_scrambled = model.predict(X_test_scrambled)
    r2_scrambled = r2_score(y_test, y_pred_scrambled)
    
    print(f"   Scrambled R2: {r2_scrambled:.5f}")
    
    if r2_scrambled < 0.1:
        print("   ✅ PASS: Model relies 100% on Phase (as hypothesized).")
    else:
        print("   ❌ FAIL: Model is cheating (using Amplitude correlations).")

    # --- TEST 3: EXTRAPOLATION ---
    print("\n[TEST 3] EXTRAPOLATION (Train < 700, Test > 700)")
    mask_train = y < 700
    mask_test = y >= 700
    
    X_train_ex = X_complex[mask_train]
    y_train_ex = y[mask_train]
    X_test_ex = X_complex[mask_test]
    y_test_ex = y[mask_test]
    
    model_ex = HolographicAetherNet(n_features=2048, brightness=0.01)
    model_ex.fit(X_train_ex, y_train_ex)
    y_pred_ex = model_ex.predict(X_test_ex)
    r2_ex = r2_score(y_test_ex, y_pred_ex)
    
    print(f"   Extrapolation R2: {r2_ex:.5f}")
    
    # --- PLOT ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, c='purple', alpha=0.5, s=5, label='Original')
    plt.scatter(y_test, y_pred_scrambled, c='gray', alpha=0.5, s=5, label='Scrambled Phase')
    plt.legend()
    plt.title("Phase Dependence Check")
    plt.xlabel("True Velocity")
    plt.ylabel("Predicted")
    
    plt.tight_layout()
    plt.savefig('benchmark_3_results.png')
    print("\n📊 Benchmark Graph saved as 'benchmark_3_results.png'")

if __name__ == "__main__":
    run_benchmark_3()
