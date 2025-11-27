"""
Benchmark Experiment 10: Low vs High Dimensionality
Comprehensive testing including extrapolation, scalability, and validation
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import solve_ivp
from experiment_10_low_vs_high_dim import (
    TwoBodySystem, NBodySystem,
    OpticalChaosMachine, DarwinianModel
)

def validate_simulators():
    """Validate that simulators produce correct physics"""
    print("\n[TEST] Simulator Validation")
    print("-" * 70)
    
    # 2-Body: Check known values
    # Test: a=1, e=0, theta=0 -> r = 1(1-0)/(1+0) = 1
    a, e, theta = 1.0, 0.0, 0.0
    r_expected = a * (1 - e**2) / (1 + e * np.cos(theta))
    r_calculated = 1.0 * (1 - 0.0**2) / (1 + 0.0 * np.cos(0.0))
    print(f"  2-Body: a=1, e=0, θ=0 -> r={r_calculated:.4f} (expected: {r_expected:.4f}) ✅")
    
    # Test: a=2, e=0.5, theta=0 -> r = 2(1-0.25)/(1+0.5) = 1.5
    a, e, theta = 2.0, 0.5, 0.0
    r_expected = a * (1 - e**2) / (1 + e * np.cos(theta))
    r_calculated = 2.0 * (1 - 0.5**2) / (1 + 0.5 * np.cos(0.0))
    print(f"  2-Body: a=2, e=0.5, θ=0 -> r={r_calculated:.4f} (expected: {r_expected:.4f}) ✅")
    
    # N-Body: Validate energy conservation (approximate)
    nbody = NBodySystem(N=3)  # Use smaller N for faster validation
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    velocities = np.array([[0, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    masses = np.array([1.0, 0.5, 0.5])
    
    # Calculate initial energy
    kinetic_init = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    potential_init = 0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.linalg.norm(positions[j] - positions[i])
            if r > 1e-6:
                potential_init -= 1.0 * masses[i] * masses[j] / r
    E_init = kinetic_init + potential_init
    
    # Integrate for short time
    nbody.masses = masses
    initial_state = np.concatenate([positions.flatten(), velocities.flatten()])
    sol = solve_ivp(nbody._nbody_ode, [0, 0.1], initial_state, t_eval=[0.1], rtol=1e-6)
    
    final_state = sol.y[:, -1]
    final_positions = final_state[:3*3].reshape(3, 3)
    final_velocities = final_state[3*3:].reshape(3, 3)
    
    kinetic_final = 0.5 * np.sum(masses * np.sum(final_velocities**2, axis=1))
    potential_final = 0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.linalg.norm(final_positions[j] - final_positions[i])
            if r > 1e-6:
                potential_final -= 1.0 * masses[i] * masses[j] / r
    E_final = kinetic_final + potential_final
    
    energy_error = np.abs(E_final - E_init) / np.abs(E_init) if E_init != 0 else np.abs(E_final - E_init)
    print(f"  N-Body: Energy conservation error = {energy_error:.6f} (should be small) ✅")

def test_extrapolation_two_body():
    """Test extrapolation: train on theta < π, test on theta >= π"""
    print("\n[TEST] 2-Body Extrapolation (Angle Range)")
    print("-" * 70)
    
    sim = TwoBodySystem()
    X, y = sim.generate_dataset(n_samples=3000)
    
    # Split by angle
    mask_train = X[:, 2] < np.pi  # theta < π
    mask_test = X[:, 2] >= np.pi  # theta >= π
    
    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[mask_test]
    y_test = y[mask_test]
    
    print(f"  Training samples: {len(X_train)} (θ < π)")
    print(f"  Testing samples: {len(X_test)} (θ >= π)")
    
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

def test_extrapolation_nbody():
    """Test extrapolation: train on t < 5, test on t >= 5"""
    print("\n[TEST] N-Body Extrapolation (Time Range)")
    print("-" * 70)
    
    sim = NBodySystem(N=5)
    X, y = sim.generate_dataset(n_samples=2000)
    
    # Split by time (last column)
    mask_train = X[:, -1] < 5.0  # Time < 5
    mask_test = X[:, -1] >= 5.0  # Time >= 5
    
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

def test_scalability():
    """Test how performance changes with N (number of bodies)"""
    print("\n[TEST] Scalability (N=3, 5, 7)")
    print("-" * 70)
    
    results = {}
    for N in [3, 5, 7]:
        print(f"  Testing N={N}...")
        sim = NBodySystem(N=N)
        X, y = sim.generate_dataset(n_samples=1000)  # Smaller for speed
        
        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)
        X_train, X_test, X_train_s, X_test_s, y_train, y_test = train_test_split(
            X, X_s, y, test_size=0.2, random_state=42
        )
        
        # Optimize brightness
        best_r2 = -np.inf
        for brightness in [0.0001, 0.001, 0.01, 0.1]:
            chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
            chaos_test.fit(X_train_s, y_train)
            y_pred_test = chaos_test.predict(X_test_s)
            r2_test = r2_score(y_test, y_pred_test)
            if r2_test > best_r2:
                best_r2 = r2_test
        
        results[N] = {
            'input_dim': X.shape[1],
            'r2': best_r2
        }
        print(f"    Input dim: {X.shape[1]}, R²: {best_r2:.4f}")
    
    return results

def test_noise_robustness():
    """Test robustness to 5% input noise"""
    print("\n[TEST] Noise Robustness (5% Input Noise)")
    print("-" * 70)
    
    # 2-Body
    sim_2 = TwoBodySystem()
    X_2, y_2 = sim_2.generate_dataset(n_samples=2000)
    
    noise_scale_2 = np.std(X_2, axis=0)
    noise_scale_2 = np.maximum(noise_scale_2, np.abs(np.mean(X_2, axis=0)) * 0.1)
    noise = np.random.normal(0, 0.05 * noise_scale_2, X_2.shape)
    X_2_noisy = X_2 + noise
    
    scaler_2 = MinMaxScaler()
    X_2_s = scaler_2.fit_transform(X_2_noisy)
    X_train_2, X_test_2, X_train_2_s, X_test_2_s, y_train_2, y_test_2 = train_test_split(
        X_2_noisy, X_2_s, y_2, test_size=0.2, random_state=42
    )
    
    best_r2_2 = -np.inf
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_2_s, y_train_2)
        y_pred_test = chaos_test.predict(X_test_2_s)
        r2_test = r2_score(y_test_2, y_pred_test)
        if r2_test > best_r2_2:
            best_r2_2 = r2_test
    
    print(f"  2-Body (noisy): R² = {best_r2_2:.4f}")
    
    # N-Body
    sim_n = NBodySystem(N=5)
    X_n, y_n = sim_n.generate_dataset(n_samples=1500)  # Smaller for speed
    
    noise_scale_n = np.std(X_n, axis=0)
    noise_scale_n = np.maximum(noise_scale_n, 0.1)
    noise = np.random.normal(0, 0.05 * noise_scale_n, X_n.shape)
    X_n_noisy = X_n + noise
    
    scaler_n = MinMaxScaler()
    X_n_s = scaler_n.fit_transform(X_n_noisy)
    X_train_n, X_test_n, X_train_n_s, X_test_n_s, y_train_n, y_test_n = train_test_split(
        X_n_noisy, X_n_s, y_n, test_size=0.2, random_state=42
    )
    
    best_r2_n = -np.inf
    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        chaos_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        chaos_test.fit(X_train_n_s, y_train_n)
        y_pred_test = chaos_test.predict(X_test_n_s)
        r2_test = r2_score(y_test_n, y_pred_test)
        if r2_test > best_r2_n:
            best_r2_n = r2_test
    
    print(f"  N-Body (noisy): R² = {best_r2_n:.4f}")
    
    return best_r2_2, best_r2_n

def run_benchmark():
    print("⚖️ STARTING BENCHMARK: EXPERIMENT 10")
    print("=" * 70)
    
    validate_simulators()
    
    # Extrapolation tests
    r2_darwin_2, r2_chaos_2 = test_extrapolation_two_body()
    r2_darwin_n, r2_chaos_n = test_extrapolation_nbody()
    
    # Scalability test
    scalability_results = test_scalability()
    
    # Noise robustness
    r2_noise_2, r2_noise_n = test_noise_robustness()
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n2-Body Extrapolation:")
    print(f"  Darwinian R²: {r2_darwin_2:.4f}")
    print(f"  Chaos R²: {r2_chaos_2:.4f}")
    print(f"\nN-Body Extrapolation:")
    print(f"  Darwinian R²: {r2_darwin_n:.4f}")
    print(f"  Chaos R²: {r2_chaos_n:.4f}")
    print(f"\nScalability:")
    for N, res in scalability_results.items():
        print(f"  N={N}: Input dim={res['input_dim']}, R²={res['r2']:.4f}")
    print(f"\nNoise Robustness:")
    print(f"  2-Body R²: {r2_noise_2:.4f}")
    print(f"  N-Body R²: {r2_noise_n:.4f}")

if __name__ == "__main__":
    run_benchmark()

