"""
BENCHMARK CERTIFICATION: Experiment 1 (Stone in Lake)
-----------------------------------------------------
Objective: 
Conduct an impartial, rigorous audit of the Optical Chaos Model to verify 
if it genuinely learns the physics of projectile motion or merely memorizes data.

Tests:
1. STANDARD ACCURACY: R2 Score on random split. (Target: > 0.95)
2. EXTRAPOLATION (Generalization): Train on V < 70m/s, Test on V > 70m/s.
   - If it learned the physics (v^2 relationship), it should predict high-speed trajectories.
   - If it overfitted, it will fail.
3. NOISE SENSITIVITY: Add 5% noise to inputs.
   - Measures the stability of the "interference pattern".
4. CAGE ANALYSIS: Correlation of internal states with V and Theta.
   - Verifies if the model is "Cage-Free" (distributed info) or reconstructed variables.

Author: System Auditor
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import the model and simulator from the main script
# Assuming Stone_in_Lake.py is in the same directory
from Stone_in_Lake import OpticalChaosMachine, PhysicsSimulator

def run_benchmark():
    print("⚖️  STARTING IMPARTIAL BENCHMARK: EXPERIMENT 1")
    print("==============================================")
    
    # --- SETUP ---
    sim = PhysicsSimulator()
    # Generate a large dataset for statistical significance
    X, y = sim.generate_dataset(n_samples=5000)
    
    # --- TEST 1: STANDARD ACCURACY ---
    print("\n[TEST 1] STANDARD ACCURACY (Random Split)")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = OpticalChaosMachine(n_features=4096, brightness=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_std = r2_score(y_test, y_pred)
    
    print(f"   R2 Score: {r2_std:.5f}")
    if r2_std > 0.95:
        print("   ✅ PASS: High fidelity prediction.")
    else:
        print("   ❌ FAIL: Model cannot reproduce physics.")

    # --- TEST 2: EXTRAPOLATION (The Truth Test) ---
    print("\n[TEST 2] EXTRAPOLATION (Train V < 70, Test V > 70)")
    # Re-generate to ensure clean split by velocity
    # X is [v0, angle]
    mask_train = X[:, 0] < 70
    mask_test = X[:, 0] >= 70
    
    X_train_ex = X[mask_train]
    y_train_ex = y[mask_train]
    X_test_ex = X[mask_test]
    y_test_ex = y[mask_test]
    
    print(f"   Training Samples: {len(X_train_ex)}")
    print(f"   Testing Samples:  {len(X_test_ex)}")
    
    # Scale (Fit on TRAIN only)
    scaler_ex = MinMaxScaler()
    X_train_ex_s = scaler_ex.fit_transform(X_train_ex)
    X_test_ex_s = scaler_ex.transform(X_test_ex)
    
    model_ex = OpticalChaosMachine(n_features=4096, brightness=0.001)
    model_ex.fit(X_train_ex_s, y_train_ex)
    y_pred_ex = model_ex.predict(X_test_ex_s)
    r2_ex = r2_score(y_test_ex, y_pred_ex)
    
    print(f"   Extrapolation R2: {r2_ex:.5f}")
    if r2_ex > 0.80:
        print("   ✅ PASS: Model generalizes to unseen energies.")
    elif r2_ex > 0.50:
        print("   ⚠️ WARN: Partial generalization.")
    else:
        print("   ❌ FAIL: Model is overfitting/memorizing.")

    # --- TEST 3: NOISE SENSITIVITY ---
    print("\n[TEST 3] NOISE SENSITIVITY (5% Input Noise)")
    # Add noise to raw X
    noise = np.random.normal(0, 0.05 * np.mean(X, axis=0), X.shape)
    X_noisy = X + noise
    
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
    
    scaler_n = MinMaxScaler()
    X_train_n_s = scaler_n.fit_transform(X_train_n)
    X_test_n_s = scaler_n.transform(X_test_n)
    
    model_n = OpticalChaosMachine(n_features=4096, brightness=0.001)
    model_n.fit(X_train_n_s, y_train_n)
    y_pred_n = model_n.predict(X_test_n_s)
    r2_noise = r2_score(y_test_n, y_pred_n)
    
    print(f"   Noisy R2: {r2_noise:.5f}")
    if r2_noise > 0.80:
        print("   💎 ROBUST: System handles noise well.")
    else:
        print("   ❄️ FRAGILE: System relies on precise interference.")

    # --- TEST 4: CAGE ANALYSIS (Internal Correlations) ---
    print("\n[TEST 4] CAGE ANALYSIS (Hidden Variable Search)")
    # Use the standard model from Test 1
    internal_states = model.get_internal_state(X_test) # X_test is already scaled
    
    # Check correlation with V and Angle
    # We want to see if any SINGLE neuron encodes "Velocity" or "Angle"
    # Or if the info is distributed.
    
    max_corr_v = 0
    max_corr_a = 0
    
    # Check a random subset of features to save time (e.g., 500)
    # Or check all if fast enough. 4096 is fast enough.
    
    # Reconstruct raw V and A for correlation check (X_test is scaled)
    # We can just check correlation against the scaled inputs, it's the same monotonic relationship
    v_ref = X_test[:, 0]
    a_ref = X_test[:, 1]
    
    corrs_v = np.abs([np.corrcoef(internal_states[:, i], v_ref)[0,1] for i in range(internal_states.shape[1])])
    corrs_a = np.abs([np.corrcoef(internal_states[:, i], a_ref)[0,1] for i in range(internal_states.shape[1])])
    
    max_corr_v = np.max(corrs_v)
    max_corr_a = np.max(corrs_a)
    mean_corr_v = np.mean(corrs_v)
    
    print(f"   Max Correlation with Velocity: {max_corr_v:.4f}")
    print(f"   Max Correlation with Angle:    {max_corr_a:.4f}")
    print(f"   Mean Correlation with V:       {mean_corr_v:.4f}")
    
    if max_corr_v < 0.5 and max_corr_a < 0.5:
        print("   🔓 CAGE BROKEN: No single feature represents human variables.")
        print("      Information is holographically distributed.")
    else:
        print("   🔒 CAGE LOCKED: Model reconstructed human variables.")

    # --- SUMMARY PLOT ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_ex, y_pred_ex, alpha=0.5, c='blue', s=1)
    plt.plot([0, max(y)], [0, max(y)], 'k--')
    plt.title(f"Extrapolation (R2: {r2_ex:.2f})")
    plt.xlabel("True Distance")
    plt.ylabel("Predicted")
    
    plt.subplot(1, 2, 2)
    plt.hist(corrs_v, bins=50, color='green', alpha=0.7)
    plt.title("Internal Correlations with Velocity")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Count of Optical Features")
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\n📊 Benchmark Graph saved as 'benchmark_results.png'")

if __name__ == "__main__":
    run_benchmark()
