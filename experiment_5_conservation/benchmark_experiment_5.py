"""
BENCHMARK CERTIFICATION: Experiment 5 (Conservation Laws)
--------------------------------------------------------
Objective: 
Conduct rigorous audit to verify if the model genuinely learns conservation laws
or merely memorizes collision patterns.

Tests:
1. EXTRAPOLATION: Train on small masses, test on large masses
2. TRANSFER: Train on elastic, test on inelastic (and vice versa)
3. CONSERVATION VERIFICATION: Check if predictions respect conservation laws
4. CAGE ANALYSIS: Correlation of internal states with energy/momentum
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiment_5_conservation import (
    CollisionSimulator, 
    ConservationChaosModel,
    DarwinianModel
)

def run_benchmark():
    print("⚖️ STARTING BENCHMARK: EXPERIMENT 5")
    print("=" * 60)
    
    sim = CollisionSimulator()
    
    # Generate datasets
    print("\n[Generating Datasets]")
    X_elastic, y_elastic = sim.generate_elastic_dataset(n_samples=5000)
    X_inelastic, y_inelastic = sim.generate_inelastic_dataset(n_samples=3000)
    
    # --- TEST 1: EXTRAPOLATION (Mass Extrapolation) ---
    print("\n[TEST 1] EXTRAPOLATION (Train Small Masses, Test Large Masses)")
    
    # Split by mass
    m1_el, m2_el = X_elastic[:, 0], X_elastic[:, 1]
    total_mass = m1_el + m2_el
    threshold = np.median(total_mass)
    
    mask_train = total_mass < threshold
    mask_test = total_mass >= threshold
    
    X_train_ex = X_elastic[mask_train]
    y_train_ex = y_elastic[mask_train]
    X_test_ex = X_elastic[mask_test]
    y_test_ex = y_elastic[mask_test]
    
    print(f"   Training samples: {len(X_train_ex)} (mass < {threshold:.2f} kg)")
    print(f"   Testing samples: {len(X_test_ex)} (mass >= {threshold:.2f} kg)")
    
    model_ex = ConservationChaosModel(n_features=4096, brightness=0.001)
    model_ex.fit(X_train_ex, y_train_ex)
    y_pred_ex = model_ex.predict(X_test_ex)
    
    r2_ex = r2_score(y_test_ex, y_pred_ex)
    print(f"   Extrapolation R²: {r2_ex:.4f}")
    
    if r2_ex > 0.80:
        print("   ✅ PASS: Model generalizes to unseen mass ranges")
    elif r2_ex > 0.50:
        print("   ⚠️ WARN: Partial generalization")
    else:
        print("   ❌ FAIL: Model overfits to training mass range")
    
    # --- TEST 2: TRANSFER (Elastic → Inelastic) ---
    print("\n[TEST 2] TRANSFER (Train Elastic, Test Inelastic)")
    
    X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(
        X_elastic, y_elastic, test_size=0.2, random_state=42
    )
    
    model_trans = ConservationChaosModel(n_features=4096, brightness=0.001)
    model_trans.fit(X_train_trans, y_train_trans)
    y_pred_trans = model_trans.predict(X_inelastic)
    
    r2_trans = r2_score(y_inelastic, y_pred_trans)
    print(f"   Transfer R²: {r2_trans:.4f}")
    
    if r2_trans > 0.70:
        print("   ✅ PASS: Model transfers to inelastic collisions")
    elif r2_trans > 0.40:
        print("   ⚠️ WARN: Partial transfer")
    else:
        print("   ❌ FAIL: Model is domain-specific (elastic only)")
    
    # --- TEST 3: CONSERVATION VERIFICATION ---
    print("\n[TEST 3] CONSERVATION LAW VERIFICATION")
    
    # Use model from test 1
    y_pred_cons = model_ex.predict(X_test_ex)
    
    m1_t, m2_t, v1_t, v2_t, e_t = X_test_ex.T
    v1_pred, v2_pred = y_pred_cons.T
    
    # Calculate conservation errors
    p_initial = m1_t * v1_t + m2_t * v2_t
    p_final = m1_t * v1_pred + m2_t * v2_pred
    momentum_error = np.abs(p_final - p_initial)
    
    E_initial = 0.5 * m1_t * v1_t**2 + 0.5 * m2_t * v2_t**2
    E_final = 0.5 * m1_t * v1_pred**2 + 0.5 * m2_t * v2_pred**2
    energy_error = np.abs(E_final - E_initial)
    
    mean_mom_err = np.mean(momentum_error)
    mean_en_err = np.mean(energy_error)
    max_mom_err = np.max(momentum_error)
    max_en_err = np.max(energy_error)
    
    print(f"   Mean momentum error: {mean_mom_err:.4f}")
    print(f"   Max momentum error: {max_mom_err:.4f}")
    print(f"   Mean energy error: {mean_en_err:.4f}")
    print(f"   Max energy error: {max_en_err:.4f}")
    
    if mean_mom_err < 0.1 and mean_en_err < 1.0:
        print("   ✅ PASS: Model respects conservation laws")
    elif mean_mom_err < 1.0:
        print("   ⚠️ WARN: Momentum conserved, but energy errors significant")
    else:
        print("   ❌ FAIL: Model violates conservation laws")
    
    # --- TEST 4: CAGE ANALYSIS ---
    print("\n[TEST 4] CAGE ANALYSIS (Internal Correlations)")
    
    internal_states = model_ex.get_internal_state(X_test_ex)
    
    # Calculate energy and momentum
    energy_total = 0.5 * m1_t * v1_t**2 + 0.5 * m2_t * v2_t**2
    momentum_total = m1_t * v1_t + m2_t * v2_t
    
    # Sample features for efficiency
    n_sample = min(1000, internal_states.shape[1])
    sample_indices = np.random.choice(internal_states.shape[1], n_sample, replace=False)
    
    corrs_energy = []
    corrs_momentum = []
    
    for idx in sample_indices:
        corr_e = np.abs(np.corrcoef(internal_states[:, idx], energy_total)[0, 1])
        corr_p = np.abs(np.corrcoef(internal_states[:, idx], momentum_total)[0, 1])
        corrs_energy.append(corr_e)
        corrs_momentum.append(corr_p)
    
    max_corr_energy = np.max(corrs_energy)
    max_corr_momentum = np.max(corrs_momentum)
    mean_corr_energy = np.mean(corrs_energy)
    mean_corr_momentum = np.mean(corrs_momentum)
    
    print(f"   Max correlation with Energy: {max_corr_energy:.4f}")
    print(f"   Max correlation with Momentum: {max_corr_momentum:.4f}")
    print(f"   Mean correlation with Energy: {mean_corr_energy:.4f}")
    print(f"   Mean correlation with Momentum: {mean_corr_momentum:.4f}")
    
    if max_corr_energy < 0.5 and max_corr_momentum < 0.5:
        print("   🔓 CAGE BROKEN: No single feature represents energy/momentum")
    elif max_corr_energy > 0.8 or max_corr_momentum > 0.8:
        print("   🔒 CAGE LOCKED: Model reconstructed human variables")
    else:
        print("   🟡 CAGE UNCLEAR: Intermediate correlation levels")
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Extrapolation
    plt.subplot(2, 3, 1)
    plt.scatter(y_test_ex[:, 0], y_pred_ex[:, 0], alpha=0.3, s=5)
    plt.plot([y_test_ex.min(), y_test_ex.max()], [y_test_ex.min(), y_test_ex.max()], 'k--')
    plt.title(f"Extrapolation (R²={r2_ex:.2f})")
    plt.xlabel("True Velocity")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Transfer
    plt.subplot(2, 3, 2)
    plt.scatter(y_inelastic[:, 0], y_pred_trans[:, 0], alpha=0.3, s=5, c='orange')
    plt.plot([y_inelastic.min(), y_inelastic.max()], [y_inelastic.min(), y_inelastic.max()], 'k--')
    plt.title(f"Transfer (R²={r2_trans:.2f})")
    plt.xlabel("True Velocity")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Conservation Errors
    plt.subplot(2, 3, 3)
    plt.hist(momentum_error, bins=50, alpha=0.7, label='Momentum', color='blue')
    plt.hist(energy_error, bins=50, alpha=0.7, label='Energy', color='red')
    plt.xlabel("Conservation Error")
    plt.ylabel("Frequency")
    plt.title("Conservation Law Violations")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Energy Correlations
    plt.subplot(2, 3, 4)
    plt.hist(corrs_energy, bins=50, alpha=0.7, color='green')
    plt.axvline(max_corr_energy, color='red', linestyle='--', label=f'Max: {max_corr_energy:.3f}')
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title("Correlations with Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Momentum Correlations
    plt.subplot(2, 3, 5)
    plt.hist(corrs_momentum, bins=50, alpha=0.7, color='purple')
    plt.axvline(max_corr_momentum, color='red', linestyle='--', label=f'Max: {max_corr_momentum:.3f}')
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title("Correlations with Momentum")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Summary
    plt.subplot(2, 3, 6)
    tests = ['Extrapolation', 'Transfer', 'Conservation']
    scores = [r2_ex, r2_trans, 1.0 - min(mean_mom_err, 1.0)]  # Normalize conservation
    colors = ['blue', 'orange', 'green']
    bars = plt.bar(tests, scores, color=colors, alpha=0.7)
    plt.ylabel("Score (normalized)")
    plt.title("Benchmark Summary")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('benchmark_5_results.png', dpi=150)
    print("\n📊 Benchmark graph saved as 'benchmark_5_results.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()

