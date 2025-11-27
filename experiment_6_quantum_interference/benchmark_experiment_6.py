"""
BENCHMARK CERTIFICATION: Experiment 6 (Quantum Interference)
-----------------------------------------------------------
Objective: 
Conduct rigorous audit to verify if the model genuinely learns quantum
interference patterns or merely memorizes data points.

Tests:
1. EXTRAPOLATION: Train on certain parameter ranges, test on others
2. PATTERN RECOGNITION: Test if model captures interference fringes
3. CAGE ANALYSIS: Correlation with wave concepts (phase, path difference)
4. NOISE ROBUSTNESS: Test stability with measurement noise
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from experiment_6_quantum_interference import (
    DoubleSlitSimulator,
    QuantumChaosModel,
    DarwinianModel
)

def run_benchmark():
    print("⚖️ STARTING BENCHMARK: EXPERIMENT 6")
    print("=" * 60)
    
    sim = DoubleSlitSimulator()
    
    # Generate large dataset
    print("\n[Generating Dataset]")
    X, y = sim.generate_dataset(n_samples=5000)
    
    # --- TEST 1: EXTRAPOLATION (Wavelength Range) ---
    print("\n[TEST 1] EXTRAPOLATION (Train Short λ, Test Long λ)")
    
    wavelength = X[:, 0]
    threshold = np.median(wavelength)
    
    mask_train = wavelength < threshold
    mask_test = wavelength >= threshold
    
    X_train_ex = X[mask_train]
    y_train_ex = y[mask_train]
    X_test_ex = X[mask_test]
    y_test_ex = y[mask_test]
    
    print(f"   Training samples: {len(X_train_ex)} (λ < {threshold:.2f})")
    print(f"   Testing samples: {len(X_test_ex)} (λ >= {threshold:.2f})")
    
    model_ex = QuantumChaosModel(n_features=4096, brightness=0.001)
    model_ex.fit(X_train_ex, y_train_ex)
    y_pred_ex = model_ex.predict(X_test_ex)
    
    r2_ex = r2_score(y_test_ex, y_pred_ex)
    print(f"   Extrapolation R²: {r2_ex:.4f}")
    
    if r2_ex > 0.80:
        print("   ✅ PASS: Model generalizes to unseen wavelengths")
    elif r2_ex > 0.50:
        print("   ⚠️ WARN: Partial generalization")
    else:
        print("   ❌ FAIL: Model overfits to training wavelength range")
    
    # --- TEST 2: PATTERN RECOGNITION ---
    print("\n[TEST 2] INTERFERENCE PATTERN RECOGNITION")
    
    # Generate a full interference pattern
    wl, sep, dist = 1.0, 2.0, 10.0
    x_pattern, true_pattern = sim.generate_full_pattern(wl, sep, dist, n_points=200)
    
    # Predict pattern
    X_pattern = np.column_stack((
        np.full(200, wl),
        np.full(200, sep),
        np.full(200, dist),
        x_pattern
    ))
    pred_pattern = model_ex.predict(X_pattern)
    
    # Check if pattern shows interference (oscillations)
    # Calculate number of peaks (should be > 1 for interference)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(pred_pattern, height=np.mean(pred_pattern))
    n_peaks = len(peaks)
    
    # Correlation with true pattern
    pattern_corr = np.corrcoef(true_pattern, pred_pattern)[0, 1]
    
    print(f"   Number of interference peaks detected: {n_peaks}")
    print(f"   Correlation with true pattern: {pattern_corr:.4f}")
    
    if n_peaks > 1 and pattern_corr > 0.7:
        print("   ✅ PASS: Model captures interference fringes")
    elif pattern_corr > 0.5:
        print("   ⚠️ WARN: Partial pattern recognition")
    else:
        print("   ❌ FAIL: Model does not capture interference pattern")
    
    # --- TEST 3: NOISE ROBUSTNESS ---
    print("\n[TEST 3] NOISE ROBUSTNESS (5% Measurement Noise)")
    
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Add noise to outputs (simulating measurement uncertainty)
    noise = np.random.normal(0, 0.05 * np.mean(y_train_n), y_train_n.shape)
    y_train_noisy = y_train_n + noise
    y_train_noisy = np.maximum(y_train_noisy, 0)  # Ensure non-negative
    
    model_n = QuantumChaosModel(n_features=4096, brightness=0.001)
    model_n.fit(X_train_n, y_train_noisy)
    y_pred_n = model_n.predict(X_test_n)
    
    r2_noise = r2_score(y_test_n, y_pred_n)
    print(f"   Noisy R²: {r2_noise:.4f}")
    
    if r2_noise > 0.80:
        print("   ✅ PASS: Model is robust to noise")
    elif r2_noise > 0.50:
        print("   ⚠️ WARN: Partial robustness")
    else:
        print("   ❌ FAIL: Model is fragile to noise")
    
    # --- TEST 4: CAGE ANALYSIS ---
    print("\n[TEST 4] CAGE ANALYSIS (Wave Concept Correlations)")
    
    internal_states = model_ex.get_internal_state(X_test_ex)
    
    # Calculate wave concepts
    wl_t, sep_t, dist_t, pos_t = X_test_ex.T
    phase = 2 * np.pi * sep_t * pos_t / (wl_t * dist_t)
    path_diff = sep_t * pos_t / dist_t
    wavenumber = 2 * np.pi / wl_t
    
    # Sample features
    n_sample = min(1000, internal_states.shape[1])
    sample_indices = np.random.choice(internal_states.shape[1], n_sample, replace=False)
    
    corrs_phase = []
    corrs_path = []
    corrs_wavenumber = []
    
    for idx in sample_indices:
        corr_p = np.abs(np.corrcoef(internal_states[:, idx], phase)[0, 1])
        corr_pd = np.abs(np.corrcoef(internal_states[:, idx], path_diff)[0, 1])
        corr_k = np.abs(np.corrcoef(internal_states[:, idx], wavenumber)[0, 1])
        corrs_phase.append(corr_p)
        corrs_path.append(corr_pd)
        corrs_wavenumber.append(corr_k)
    
    max_corr_phase = np.max(corrs_phase)
    max_corr_path = np.max(corrs_path)
    max_corr_wavenumber = np.max(corrs_wavenumber)
    mean_corr_phase = np.mean(corrs_phase)
    
    print(f"   Max correlation with Phase: {max_corr_phase:.4f}")
    print(f"   Max correlation with Path Difference: {max_corr_path:.4f}")
    print(f"   Max correlation with Wavenumber: {max_corr_wavenumber:.4f}")
    print(f"   Mean correlation with Phase: {mean_corr_phase:.4f}")
    
    if max_corr_phase < 0.5 and max_corr_path < 0.5:
        print("   🔓 CAGE BROKEN: No single feature represents wave concepts")
    elif max_corr_phase > 0.8 or max_corr_path > 0.8:
        print("   🔒 CAGE LOCKED: Model reconstructed wave concepts")
    else:
        print("   🟡 CAGE UNCLEAR: Intermediate correlation levels")
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Extrapolation
    plt.subplot(2, 4, 1)
    plt.scatter(y_test_ex, y_pred_ex, alpha=0.3, s=5)
    plt.plot([y_test_ex.min(), y_test_ex.max()], [y_test_ex.min(), y_test_ex.max()], 'k--')
    plt.title(f"Extrapolation (R²={r2_ex:.2f})")
    plt.xlabel("True Probability")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Pattern Recognition
    plt.subplot(2, 4, 2)
    plt.plot(x_pattern, true_pattern, 'b-', lw=2, label='True')
    plt.plot(x_pattern, pred_pattern, 'r--', lw=2, label='Predicted')
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.title(f"Pattern Recognition\n(Corr={pattern_corr:.2f}, Peaks={n_peaks})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Noise Robustness
    plt.subplot(2, 4, 3)
    plt.scatter(y_test_n, y_pred_n, alpha=0.3, s=5, c='orange')
    plt.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'k--')
    plt.title(f"Noise Robustness (R²={r2_noise:.2f})")
    plt.xlabel("True Probability")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    
    # Plot 4-6: Correlations
    plt.subplot(2, 4, 4)
    plt.hist(corrs_phase, bins=50, alpha=0.7, color='green')
    plt.axvline(max_corr_phase, color='red', linestyle='--', label=f'Max: {max_corr_phase:.3f}')
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title("Correlations with Phase")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 5)
    plt.hist(corrs_path, bins=50, alpha=0.7, color='purple')
    plt.axvline(max_corr_path, color='red', linestyle='--', label=f'Max: {max_corr_path:.3f}')
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title("Correlations with Path Diff")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 6)
    plt.hist(corrs_wavenumber, bins=50, alpha=0.7, color='blue')
    plt.axvline(max_corr_wavenumber, color='red', linestyle='--', label=f'Max: {max_corr_wavenumber:.3f}')
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title("Correlations with Wavenumber")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Summary
    plt.subplot(2, 4, 7)
    tests = ['Extrapolation', 'Pattern', 'Noise']
    scores = [r2_ex, pattern_corr, r2_noise]
    colors = ['blue', 'green', 'orange']
    bars = plt.bar(tests, scores, color=colors, alpha=0.7)
    plt.ylabel("Score")
    plt.title("Benchmark Summary")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Plot 8: Cage Status
    plt.subplot(2, 4, 8)
    concepts = ['Phase', 'Path Diff', 'Wavenumber']
    max_corrs = [max_corr_phase, max_corr_path, max_corr_wavenumber]
    colors_cage = ['green', 'purple', 'blue']
    bars = plt.bar(concepts, max_corrs, color=colors_cage, alpha=0.7)
    plt.ylabel("Max Correlation")
    plt.title("Cage Analysis")
    plt.axhline(0.8, color='red', linestyle='--', label='Locked threshold')
    plt.axhline(0.5, color='orange', linestyle='--', label='Broken threshold')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('benchmark_6_results.png', dpi=150)
    print("\n📊 Benchmark graph saved as 'benchmark_6_results.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()

