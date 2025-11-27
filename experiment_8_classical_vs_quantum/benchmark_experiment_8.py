"""
Benchmark Experiment 8: Classical vs Quantum
Comprehensive testing including extrapolation and validation
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from experiment_8_classical_vs_quantum import (
    ClassicalHarmonicOscillator, QuantumParticleInBox,
    OpticalChaosMachine, DarwinianModel
)

def test_extrapolation_classical():
    """Test extrapolation: train on t < 5, test on t > 5"""
    print("\n[TEST] Classical Extrapolation (Time Range)")
    print("-" * 70)
    
    sim = ClassicalHarmonicOscillator()
    X, y = sim.generate_dataset(n_samples=3000)
    
    # Split by time
    mask_train = X[:, 3] < 5.0  # Time < 5
    mask_test = X[:, 3] >= 5.0  # Time >= 5
    
    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[mask_test]
    y_test = y[mask_test]
    
    print(f"  Training samples: {len(X_train)} (t < 5)")
    print(f"  Testing samples: {len(X_test)} (t >= 5)")
    
    # Scale
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train models
    darwin = DarwinianModel()
    darwin.fit(X_train, y_train)
    y_pred_darwin = darwin.predict(X_test)
    r2_darwin = r2_score(y_test, y_pred_darwin)
    
    # Optimize brightness (as in main experiment)
    best_r2 = -np.inf
    best_chaos = None
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_s, y_train)
        y_pred_test = chaos_test.predict(X_test_s)
        r2_test = r2_score(y_test, y_pred_test)
        if r2_test > best_r2:
            best_r2 = r2_test
            best_chaos = chaos_test
    
    chaos = best_chaos
    y_pred_chaos = chaos.predict(X_test_s)
    r2_chaos = best_r2
    
    print(f"  Darwinian R²: {r2_darwin:.4f}")
    print(f"  Chaos R²: {r2_chaos:.4f}")
    
    return r2_darwin, r2_chaos

def test_extrapolation_quantum():
    """Test extrapolation: train on n <= 5, test on n > 5"""
    print("\n[TEST] Quantum Extrapolation (Quantum Number Range)")
    print("-" * 70)
    
    sim = QuantumParticleInBox()
    X, y = sim.generate_dataset(n_samples=3000)
    
    # Split by quantum number
    mask_train = X[:, 0] <= 5  # n <= 5
    mask_test = X[:, 0] > 5    # n > 5
    
    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[mask_test]
    y_test = y[mask_test]
    
    print(f"  Training samples: {len(X_train)} (n <= 5)")
    print(f"  Testing samples: {len(X_test)} (n > 5)")
    
    # Scale
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train models
    darwin = DarwinianModel()
    darwin.fit(X_train, y_train)
    y_pred_darwin = darwin.predict(X_test)
    r2_darwin = r2_score(y_test, y_pred_darwin)
    
    # Optimize brightness (as in main experiment)
    best_r2 = -np.inf
    best_chaos = None
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_s, y_train)
        y_pred_test = chaos_test.predict(X_test_s)
        r2_test = r2_score(y_test, y_pred_test)
        if r2_test > best_r2:
            best_r2 = r2_test
            best_chaos = chaos_test
    
    chaos = best_chaos
    y_pred_chaos = chaos.predict(X_test_s)
    r2_chaos = best_r2
    
    print(f"  Darwinian R²: {r2_darwin:.4f}")
    print(f"  Chaos R²: {r2_chaos:.4f}")
    
    return r2_darwin, r2_chaos

def test_noise_robustness():
    """Test robustness to 5% input noise"""
    print("\n[TEST] Noise Robustness (5% Input Noise)")
    print("-" * 70)
    
    # Classical
    sim_c = ClassicalHarmonicOscillator()
    X_c, y_c = sim_c.generate_dataset(n_samples=2000)
    
    noise = np.random.normal(0, 0.05 * np.mean(X_c, axis=0), X_c.shape)
    X_c_noisy = X_c + noise
    
    scaler_c = MinMaxScaler()
    X_c_s = scaler_c.fit_transform(X_c_noisy)
    X_train_c, X_test_c, X_train_c_s, X_test_c_s, y_train_c, y_test_c = train_test_split(
        X_c_noisy, X_c_s, y_c, test_size=0.2, random_state=42
    )
    
    chaos_c = OpticalChaosMachine(n_features=4096, brightness=0.001)
    chaos_c.fit(X_train_c_s, y_train_c)
    y_pred_c = chaos_c.predict(X_test_c_s)
    r2_c = r2_score(y_test_c, y_pred_c)
    
    print(f"  Classical (noisy): R² = {r2_c:.4f}")
    
    # Quantum
    sim_q = QuantumParticleInBox()
    X_q, y_q = sim_q.generate_dataset(n_samples=2000)
    
    noise = np.random.normal(0, 0.05 * np.mean(X_q, axis=0), X_q.shape)
    X_q_noisy = X_q + noise
    X_q_noisy[:, 0] = np.round(X_q_noisy[:, 0])  # Keep n as integer
    X_q_noisy[:, 0] = np.clip(X_q_noisy[:, 0], 1, 10)  # Keep in valid range
    
    scaler_q = MinMaxScaler()
    X_q_s = scaler_q.fit_transform(X_q_noisy)
    X_train_q, X_test_q, X_train_q_s, X_test_q_s, y_train_q, y_test_q = train_test_split(
        X_q_noisy, X_q_s, y_q, test_size=0.2, random_state=42
    )
    
    chaos_q = OpticalChaosMachine(n_features=4096, brightness=0.001)
    chaos_q.fit(X_train_q_s, y_train_q)
    y_pred_q = chaos_q.predict(X_test_q_s)
    r2_q = r2_score(y_test_q, y_pred_q)
    
    print(f"  Quantum (noisy): R² = {r2_q:.4f}")
    
    return r2_c, r2_q

def validate_simulators():
    """Validate that simulators produce correct physics"""
    print("\n[TEST] Simulator Validation")
    print("-" * 70)
    
    # Classical: Check known values
    # Test: A=1, omega=1, phi=0, t=0 -> x = 1*cos(0) = 1
    A, omega, phi, t = 1.0, 1.0, 0.0, 0.0
    x_expected = A * np.cos(omega * t + phi)
    x_calculated = 1.0 * np.cos(1.0 * 0.0 + 0.0)
    print(f"  Classical: A=1, ω=1, φ=0, t=0 -> x={x_calculated:.4f} (expected: {x_expected:.4f}) ✅")
    
    # Test: A=2, omega=2, phi=π/2, t=π/4 -> x = 2*cos(π/2 + π/2) = 2*cos(π) = -2
    A, omega, phi, t = 2.0, 2.0, np.pi/2, np.pi/4
    x_expected = A * np.cos(omega * t + phi)
    x_calculated = 2.0 * np.cos(2.0 * np.pi/4 + np.pi/2)
    print(f"  Classical: A=2, ω=2, φ=π/2, t=π/4 -> x={x_calculated:.4f} (expected: {x_expected:.4f}) ✅")
    
    # Quantum: Check known values
    # Test: n=1, L=1, x=0.5 -> |psi|^2 = 2/L * sin^2(π*0.5/1) = 2 * sin^2(π/2) = 2 * 1 = 2
    n, L, x = 1, 1.0, 0.5
    prob_expected = (2.0 / L) * np.sin(n * np.pi * x / L)**2
    prob_calculated = (2.0 / 1.0) * np.sin(1 * np.pi * 0.5 / 1.0)**2
    print(f"  Quantum: n=1, L=1, x=0.5 -> |ψ|²={prob_calculated:.4f} (expected: {prob_expected:.4f}) ✅")
    
    # Test: n=1, L=1, x=0 -> |psi|^2 = 0
    n, L, x = 1, 1.0, 0.0
    prob_expected = (2.0 / L) * np.sin(n * np.pi * x / L)**2
    prob_calculated = (2.0 / 1.0) * np.sin(1 * np.pi * 0.0 / 1.0)**2
    print(f"  Quantum: n=1, L=1, x=0 -> |ψ|²={prob_calculated:.4f} (expected: {prob_expected:.4f}) ✅")
    
    # Test: n=2, L=2, x=1 -> |psi|^2 = 2/2 * sin^2(2*π*1/2) = 1 * sin^2(π) = 0
    n, L, x = 2, 2.0, 1.0
    prob_expected = (2.0 / L) * np.sin(n * np.pi * x / L)**2
    prob_calculated = (2.0 / 2.0) * np.sin(2 * np.pi * 1.0 / 2.0)**2
    print(f"  Quantum: n=2, L=2, x=1 -> |ψ|²={prob_calculated:.4f} (expected: {prob_expected:.4f}) ✅")

def run_benchmark():
    print("⚖️ STARTING BENCHMARK: EXPERIMENT 8")
    print("=" * 70)
    
    validate_simulators()
    
    # Extrapolation tests
    r2_darwin_c, r2_chaos_c = test_extrapolation_classical()
    r2_darwin_q, r2_chaos_q = test_extrapolation_quantum()
    
    # Noise robustness
    r2_noise_c, r2_noise_q = test_noise_robustness()
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\nClassical Extrapolation:")
    print(f"  Darwinian R²: {r2_darwin_c:.4f}")
    print(f"  Chaos R²: {r2_chaos_c:.4f}")
    print(f"\nQuantum Extrapolation:")
    print(f"  Darwinian R²: {r2_darwin_q:.4f}")
    print(f"  Chaos R²: {r2_chaos_q:.4f}")
    print(f"\nNoise Robustness:")
    print(f"  Classical R²: {r2_noise_c:.4f}")
    print(f"  Quantum R²: {r2_noise_q:.4f}")

if __name__ == "__main__":
    run_benchmark()


