"""
BENCHMARK CERTIFICATION: Experiment 7 (Phase Transitions)
---------------------------------------------------------
Objective: 
Conduct rigorous audit to verify if the model can genuinely detect phase
transitions or if failures are due to experimental design issues.

Tests:
1. BASELINE COMPARISON: Verify problem is learnable
2. DIMENSIONALITY TEST: Check if high input dimension is the issue
3. BRIGHTNESS TUNING: Verify hyperparameters
4. CAGE ANALYSIS: Correlation with temperature
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from experiment_7_phase_transitions import (
    IsingModelSimulator,
    PhaseTransitionChaosModel
)

def run_benchmark():
    print("⚖️ STARTING BENCHMARK: EXPERIMENT 7")
    print("=" * 60)
    
    sim = IsingModelSimulator(size=20)
    
    # Generate dataset
    print("\n[Generating Dataset]")
    X, y, temperatures = sim.generate_dataset(n_samples=1000, temp_range=(0.5, 4.0))
    
    # --- TEST 1: BASELINE COMPARISON ---
    print("\n[TEST 1] BASELINE COMPARISON (Is Problem Learnable?)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Simple linear model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    linear_model = Ridge(alpha=1.0)
    linear_model.fit(X_train_scaled, y_train)
    y_pred_linear = linear_model.predict(X_test_scaled)
    
    r2_linear = r2_score(y_test, y_pred_linear)
    print(f"   Linear Model R²: {r2_linear:.4f}")
    
    if r2_linear > 0.9:
        print("   ✅ PASS: Problem is learnable")
    else:
        print("   ❌ FAIL: Problem may be too difficult")
    
    # --- TEST 2: CHAOS MODEL PERFORMANCE ---
    print("\n[TEST 2] CHAOS MODEL PERFORMANCE")
    
    phase_model = PhaseTransitionChaosModel(n_features=2048, brightness=0.001)
    phase_model.fit(X_train, y_train)
    y_pred_phase = phase_model.predict(X_test)
    
    r2_phase = r2_score(y_test, y_pred_phase)
    print(f"   Phase Chaos R²: {r2_phase:.4f}")
    
    if r2_phase > 0.7:
        print("   ✅ PASS: Chaos model learns")
    elif r2_phase > 0.0:
        print("   ⚠️ WARN: Partial learning")
    else:
        print("   ❌ FAIL: Chaos model fails completely")
    
    # --- TEST 3: DIMENSIONALITY ANALYSIS ---
    print("\n[TEST 3] DIMENSIONALITY ANALYSIS")
    
    print(f"   Input dimension: {X.shape[1]} (20x20 = 400 spins)")
    print(f"   Output dimension: 1 (magnetization)")
    print(f"   Chaos features: 2048")
    print(f"   Ratio (features/inputs): {2048 / X.shape[1]:.2f}")
    
    # Check if reducing input dimension helps
    # Use PCA or just subsample spins
    n_subsample = 100
    X_small = X[:, :n_subsample]
    X_train_small, X_test_small = X_small[:800], X_small[800:]
    
    model_small = PhaseTransitionChaosModel(n_features=2048, brightness=0.001)
    model_small.fit(X_train_small, y_train)
    y_pred_small = model_small.predict(X_test_small)
    r2_small = r2_score(y_test, y_pred_small)
    
    print(f"   With {n_subsample} inputs: R² = {r2_small:.4f}")
    
    # --- TEST 4: BRIGHTNESS TUNING ---
    print("\n[TEST 4] BRIGHTNESS HYPERPARAMETER")
    
    brightnesses = [0.0001, 0.001, 0.01, 0.1]
    best_brightness = 0.001
    best_r2 = -np.inf
    
    for b in brightnesses:
        model = PhaseTransitionChaosModel(n_features=2048, brightness=b)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_brightness = b
        print(f"   brightness={b:.4f}: R²={r2:.4f}")
    
    print(f"   Best brightness: {best_brightness:.4f} (R²={best_r2:.4f})")
    
    # --- TEST 5: CAGE ANALYSIS ---
    print("\n[TEST 5] CAGE ANALYSIS")
    
    internal_states = phase_model.get_internal_state(X_test)
    
    # Get temperatures for test set (use same split as before)
    _, T_test = train_test_split(temperatures, test_size=0.2, random_state=42)
    
    # Sample features
    n_sample = min(500, internal_states.shape[1])
    sample_indices = np.random.choice(internal_states.shape[1], n_sample, replace=False)
    
    corrs_temp = []
    corrs_mag = []
    
    for idx in sample_indices:
        corr_t = np.abs(np.corrcoef(internal_states[:, idx], T_test)[0, 1])
        corr_m = np.abs(np.corrcoef(internal_states[:, idx], y_test)[0, 1])
        corrs_temp.append(corr_t)
        corrs_mag.append(corr_m)
    
    max_corr_temp = np.max(corrs_temp)
    max_corr_mag = np.max(corrs_mag)
    mean_corr_temp = np.mean(corrs_temp)
    
    print(f"   Max correlation with Temperature: {max_corr_temp:.4f}")
    print(f"   Max correlation with Magnetization: {max_corr_mag:.4f}")
    print(f"   Mean correlation with Temperature: {mean_corr_temp:.4f}")
    
    if max_corr_temp < 0.5 and r2_phase > 0.7:
        print("   🔓 CAGE BROKEN: No temperature reconstruction")
    elif max_corr_temp > 0.8:
        print("   🔒 CAGE LOCKED: Temperature reconstructed")
    else:
        print("   🟡 CAGE UNCLEAR: Intermediate correlations")
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Linear vs Chaos
    plt.subplot(2, 4, 1)
    plt.scatter(y_test, y_pred_linear, alpha=0.3, s=5, label='Linear', c='green')
    plt.scatter(y_test, y_pred_phase, alpha=0.3, s=5, label='Chaos', c='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel("True Magnetization")
    plt.ylabel("Predicted")
    plt.title(f"Linear (R²={r2_linear:.2f}) vs Chaos (R²={r2_phase:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Temperature Correlations
    plt.subplot(2, 4, 2)
    plt.hist(corrs_temp, bins=50, alpha=0.7, color='red')
    plt.axvline(max_corr_temp, color='black', linestyle='--', label=f'Max: {max_corr_temp:.3f}')
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title("Correlations with Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Magnetization Correlations
    plt.subplot(2, 4, 3)
    plt.hist(corrs_mag, bins=50, alpha=0.7, color='blue')
    plt.axvline(max_corr_mag, color='black', linestyle='--', label=f'Max: {max_corr_mag:.3f}')
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title("Correlations with Magnetization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Brightness Comparison
    plt.subplot(2, 4, 4)
    r2_brightness = []
    for b in brightnesses:
        model = PhaseTransitionChaosModel(n_features=2048, brightness=b)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_brightness.append(r2_score(y_test, y_pred))
    plt.plot(brightnesses, r2_brightness, 'o-')
    plt.xlabel("Brightness")
    plt.ylabel("R² Score")
    plt.title("Brightness Tuning")
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Dimensionality Test
    plt.subplot(2, 4, 5)
    dims = [100, 200, 300, 400]
    r2_dims = []
    for d in dims:
        X_d = X[:, :d]
        X_train_d, X_test_d = X_d[:800], X_d[800:]
        model_d = PhaseTransitionChaosModel(n_features=2048, brightness=0.001)
        model_d.fit(X_train_d, y_train)
        y_pred_d = model_d.predict(X_test_d)
        r2_dims.append(r2_score(y_test, y_pred_d))
    plt.plot(dims, r2_dims, 'o-')
    plt.xlabel("Input Dimension")
    plt.ylabel("R² Score")
    plt.title("Dimensionality Impact")
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Summary
    plt.subplot(2, 4, 6)
    models = ['Linear', 'Chaos']
    r2_scores = [r2_linear, r2_phase]
    colors = ['green', 'red']
    bars = plt.bar(models, r2_scores, color=colors, alpha=0.7)
    plt.ylabel("R² Score")
    plt.title("Model Comparison")
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('benchmark_7_results.png', dpi=150)
    print("\n📊 Benchmark graph saved as 'benchmark_7_results.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()

