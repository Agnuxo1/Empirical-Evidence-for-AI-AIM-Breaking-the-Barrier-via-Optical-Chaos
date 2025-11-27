"""
CRITICAL STRESS TEST: Einstein's Train
--------------------------------------
Objective: Determine if the Optical AI is "cheating" (overfitting/memorizing) 
or actually learning the physical law.

Tests:
1. EXTRAPOLATION: Train on v < 0.75c, Test on v > 0.75c.
   - If it learns the "concept" of relativity, it should predict the curve continuation.
   - If it's just a lookup table, it will fail miserably on the unseen high-speed regime.

2. NOISE ROBUSTNESS: Add 5% random noise to the input observations.
   - Real physics is noisy. A fragile "placeholder" model will collapse.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Reuse the Simulator and Net from the main experiment
from experiment_2_einstein_train import RelativitySimulator, OpticalInterferenceNet

def run_stress_test():
    print("🔥 STARTING STRESS TEST: CRITICAL REVIEW")
    print("----------------------------------------")
    
    sim = RelativitySimulator()
    # Generate more samples for a dense test
    X, y_gamma, velocities = sim.generate_dataset(n_samples=5000)
    
    # --- TEST 1: EXTRAPOLATION (The "Learning" Check) ---
    print("\n[TEST 1] EXTRAPOLATION: Train on v < 0.75c, Test on v > 0.75c")
    
    # Mask for splitting by PHYSICS (not random)
    mask_train = velocities < (0.75 * sim.c)
    mask_test = velocities >= (0.75 * sim.c)
    
    X_train = X[mask_train]
    y_train = y_gamma[mask_train]
    v_train = velocities[mask_train]
    
    X_test = X[mask_test]
    y_test = y_gamma[mask_test]
    v_test = velocities[mask_test]
    
    print(f"   Training Samples: {len(X_train)} (Low Speed)")
    print(f"   Testing Samples:  {len(X_test)} (High Speed - Unseen Regime)")
    
    # Scale (Fit scaler ONLY on training data to avoid leakage)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Transform test with train parameters
    
    # Train Optical Model
    model = OpticalInterferenceNet(n_components=5000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2_extrap = r2_score(y_test, y_pred)
    print(f"   👉 Extrapolation R2 Score: {r2_extrap:.5f}")
    
    # Plot Extrapolation
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(v_train, y_train, s=1, c='gray', alpha=0.5, label='Training Data (v < 0.75c)')
    plt.plot(v_test, y_test, 'k-', lw=2, label='True Physics (Hidden)')
    plt.scatter(v_test, y_pred, s=5, c='red', label='AI Prediction')
    plt.title(f"Extrapolation Test (R2: {r2_extrap:.2f})")
    plt.xlabel("Velocity (c)")
    plt.ylabel("Gamma")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- TEST 2: NOISE ROBUSTNESS ---
    print("\n[TEST 2] NOISE ROBUSTNESS: Adding 5% Measurement Noise")
    
    # Add noise to X (The observed path length)
    noise_level = 0.05
    noise = np.random.normal(0, noise_level * np.mean(X), X.shape)
    X_noisy = X + noise
    
    # Standard random split
    from sklearn.model_selection import train_test_split
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_noisy, y_gamma, test_size=0.2, random_state=42)
    
    # Scale
    scaler_n = MinMaxScaler()
    X_train_n_s = scaler_n.fit_transform(X_train_n)
    X_test_n_s = scaler_n.transform(X_test_n)
    
    # Train
    model_noisy = OpticalInterferenceNet(n_components=5000)
    model_noisy.fit(X_train_n_s, y_train_n)
    y_pred_n = model_noisy.predict(X_test_n_s)
    
    r2_noise = r2_score(y_test_n, y_pred_n)
    print(f"   👉 Noisy Data R2 Score: {r2_noise:.5f}")
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_n, y_pred_n, alpha=0.3, c='purple')
    plt.plot([min(y_test_n), max(y_test_n)], [min(y_test_n), max(y_test_n)], 'k--')
    plt.title(f"Noise Robustness (R2: {r2_noise:.2f})")
    plt.xlabel("True Gamma")
    plt.ylabel("Predicted Gamma")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stress_test_results.png')
    print("\n📊 Stress Test Graph saved as 'stress_test_results.png'")
    
    # --- VERDICT ---
    print("\n--- CRITICAL VERDICT ---")
    if r2_extrap < 0.5:
        print("❌ FAILED EXTRAPOLATION. The model is a 'Curve Fitter'.")
        print("   It memorized the training data but does not understand the law of relativity.")
        print("   It cannot predict what happens at speeds it hasn't seen.")
    else:
        print("✅ PASSED EXTRAPOLATION. The model has captured the underlying geometry.")
        print("   It successfully predicted the relativistic curve for unseen high speeds.")

if __name__ == "__main__":
    run_stress_test()
