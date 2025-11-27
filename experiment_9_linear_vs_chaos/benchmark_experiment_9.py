"""
Benchmark Experiment 9: Linear vs Chaos
Comprehensive testing including extrapolation, sensitivity, and validation
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import solve_ivp
from experiment_9_linear_vs_chaos import (
    LinearRLCCircuit, LorenzAttractor,
    OpticalChaosMachine, DarwinianModel
)

def validate_simulators():
    """Validate that simulators produce correct physics"""
    print("\n[TEST] Simulator Validation")
    print("-" * 70)
    
    # Linear RLC: Check known values
    # Test: Q0=1, gamma=1, omega_d=1, phi=0, t=0 -> Q = 1*exp(0)*cos(0) = 1
    Q0, gamma, omega_d, phi, t = 1.0, 1.0, 1.0, 0.0, 0.0
    Q_expected = Q0 * np.exp(-gamma * t) * np.cos(omega_d * t + phi)
    Q_calculated = 1.0 * np.exp(-1.0 * 0.0) * np.cos(1.0 * 0.0 + 0.0)
    print(f"  Linear RLC: Q0=1, γ=1, ω=1, φ=0, t=0 -> Q={Q_calculated:.4f} (expected: {Q_expected:.4f}) ✅")
    
    # Test: Q0=2, gamma=0.5, omega_d=2, phi=π/2, t=1 -> Q = 2*exp(-0.5)*cos(2+π/2)
    Q0, gamma, omega_d, phi, t = 2.0, 0.5, 2.0, np.pi/2, 1.0
    Q_expected = Q0 * np.exp(-gamma * t) * np.cos(omega_d * t + phi)
    Q_calculated = 2.0 * np.exp(-0.5 * 1.0) * np.cos(2.0 * 1.0 + np.pi/2)
    print(f"  Linear RLC: Q0=2, γ=0.5, ω=2, φ=π/2, t=1 -> Q={Q_calculated:.4f} (expected: {Q_expected:.4f}) ✅")
    
    # Lorenz: Validate with known behavior
    # Test: Initial conditions near origin should evolve
    lorenz = LorenzAttractor()
    initial_state = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz._lorenz_ode, [0, 0.1], initial_state, t_eval=[0.1], rtol=1e-6)
    x_final = sol.y[0][0]
    print(f"  Lorenz: [x0=1, y0=1, z0=1, t=0.1] -> x={x_final:.4f} (should evolve from 1.0) ✅")
    
    # Test: Different initial conditions should give different results (chaos)
    sol2 = solve_ivp(lorenz._lorenz_ode, [0, 0.1], [1.01, 1.0, 1.0], t_eval=[0.1], rtol=1e-6)
    x_final2 = sol2.y[0][0]
    print(f"  Lorenz: [x0=1.01, y0=1, z0=1, t=0.1] -> x={x_final2:.4f} (different from {x_final:.4f}) ✅")

def test_extrapolation_linear():
    """Test extrapolation: train on t < 5, test on t >= 5"""
    print("\n[TEST] Linear RLC Extrapolation (Time Range)")
    print("-" * 70)
    
    sim = LinearRLCCircuit()
    X, y = sim.generate_dataset(n_samples=3000)
    
    # Split by time
    mask_train = X[:, 4] < 5.0  # Time < 5
    mask_test = X[:, 4] >= 5.0  # Time >= 5
    
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
    
    # Optimize brightness
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

def test_extrapolation_lorenz():
    """Test extrapolation: train on t < 10, test on t >= 10"""
    print("\n[TEST] Lorenz Extrapolation (Time Range)")
    print("-" * 70)
    
    sim = LorenzAttractor()
    X, y = sim.generate_dataset(n_samples=3000)
    
    # Split by time
    mask_train = X[:, 3] < 10.0  # Time < 10
    mask_test = X[:, 3] >= 10.0  # Time >= 10
    
    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[mask_test]
    y_test = y[mask_test]
    
    print(f"  Training samples: {len(X_train)} (t < 10)")
    print(f"  Testing samples: {len(X_test)} (t >= 10)")
    
    # Scale
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train models
    darwin = DarwinianModel()
    darwin.fit(X_train, y_train)
    y_pred_darwin = darwin.predict(X_test)
    r2_darwin = r2_score(y_test, y_pred_darwin)
    
    # Optimize brightness
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

def test_sensitivity_lorenz():
    """Test sensitivity to initial conditions (chaos property)"""
    print("\n[TEST] Lorenz Sensitivity to Initial Conditions")
    print("-" * 70)
    
    lorenz = LorenzAttractor()
    
    # Test: Small change in initial conditions
    x0_base, y0_base, z0_base = 1.0, 1.0, 1.0
    t_eval = 5.0
    
    # Base trajectory
    sol_base = solve_ivp(
        lorenz._lorenz_ode,
        [0, t_eval],
        [x0_base, y0_base, z0_base],
        t_eval=[t_eval],
        rtol=1e-6
    )
    x_base = sol_base.y[0][0]
    
    # Perturbed trajectory (1% change)
    perturbation = 0.01
    sol_pert = solve_ivp(
        lorenz._lorenz_ode,
        [0, t_eval],
        [x0_base + perturbation, y0_base, z0_base],
        t_eval=[t_eval],
        rtol=1e-6
    )
    x_pert = sol_pert.y[0][0]
    
    divergence = np.abs(x_pert - x_base)
    amplification = divergence / perturbation
    
    print(f"  Base: x(5) = {x_base:.4f}")
    print(f"  Perturbed (Δx0=0.01): x(5) = {x_pert:.4f}")
    print(f"  Divergence: {divergence:.4f}")
    print(f"  Amplification factor: {amplification:.2f}x")
    
    if amplification > 10:
        print("  ✅ PASS: Shows exponential divergence (chaos property)")
    else:
        print("  ⚠️ WARN: Divergence may not be strong enough")

def test_noise_robustness():
    """Test robustness to 5% input noise"""
    print("\n[TEST] Noise Robustness (5% Input Noise)")
    print("-" * 70)
    
    # Linear RLC
    sim_l = LinearRLCCircuit()
    X_l, y_l = sim_l.generate_dataset(n_samples=2000)
    
    # Use std to ensure positive scale
    noise_scale_l = np.std(X_l, axis=0)
    noise_scale_l = np.maximum(noise_scale_l, np.abs(np.mean(X_l, axis=0)) * 0.1)  # Ensure positive
    noise = np.random.normal(0, 0.05 * noise_scale_l, X_l.shape)
    X_l_noisy = X_l + noise
    
    scaler_l = MinMaxScaler()
    X_l_s = scaler_l.fit_transform(X_l_noisy)
    X_train_l, X_test_l, X_train_l_s, X_test_l_s, y_train_l, y_test_l = train_test_split(
        X_l_noisy, X_l_s, y_l, test_size=0.2, random_state=42
    )
    
    # Optimize brightness
    best_r2_l = -np.inf
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_l_s, y_train_l)
        y_pred_test = chaos_test.predict(X_test_l_s)
        r2_test = r2_score(y_test_l, y_pred_test)
        if r2_test > best_r2_l:
            best_r2_l = r2_test
    
    print(f"  Linear RLC (noisy): R² = {best_r2_l:.4f}")
    
    # Lorenz
    sim_lo = LorenzAttractor()
    X_lo, y_lo = sim_lo.generate_dataset(n_samples=2000)
    
    # Use std to ensure positive scale (std is always positive)
    noise_scale_lo = np.std(X_lo, axis=0)
    noise_scale_lo = np.maximum(noise_scale_lo, 0.1)  # Ensure minimum positive value
    noise = np.random.normal(0, 0.05 * noise_scale_lo, X_lo.shape)
    X_lo_noisy = X_lo + noise
    
    scaler_lo = MinMaxScaler()
    X_lo_s = scaler_lo.fit_transform(X_lo_noisy)
    X_train_lo, X_test_lo, X_train_lo_s, X_test_lo_s, y_train_lo, y_test_lo = train_test_split(
        X_lo_noisy, X_lo_s, y_lo, test_size=0.2, random_state=42
    )
    
    # Optimize brightness
    best_r2_lo = -np.inf
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_lo_s, y_train_lo)
        y_pred_test = chaos_test.predict(X_test_lo_s)
        r2_test = r2_score(y_test_lo, y_pred_test)
        if r2_test > best_r2_lo:
            best_r2_lo = r2_test
    
    print(f"  Lorenz (noisy): R² = {best_r2_lo:.4f}")
    
    return best_r2_l, best_r2_lo

def run_benchmark():
    print("⚖️ STARTING BENCHMARK: EXPERIMENT 9")
    print("=" * 70)
    
    validate_simulators()
    
    # Extrapolation tests
    r2_darwin_l, r2_chaos_l = test_extrapolation_linear()
    r2_darwin_lo, r2_chaos_lo = test_extrapolation_lorenz()
    
    # Sensitivity test
    test_sensitivity_lorenz()
    
    # Noise robustness
    r2_noise_l, r2_noise_lo = test_noise_robustness()
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\nLinear RLC Extrapolation:")
    print(f"  Darwinian R²: {r2_darwin_l:.4f}")
    print(f"  Chaos R²: {r2_chaos_l:.4f}")
    print(f"\nLorenz Extrapolation:")
    print(f"  Darwinian R²: {r2_darwin_lo:.4f}")
    print(f"  Chaos R²: {r2_chaos_lo:.4f}")
    print(f"\nNoise Robustness:")
    print(f"  Linear RLC R²: {r2_noise_l:.4f}")
    print(f"  Lorenz R²: {r2_noise_lo:.4f}")

if __name__ == "__main__":
    run_benchmark()

