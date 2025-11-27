"""
EXPERIMENT D2: FORCING EMERGENT REPRESENTATIONS VIA GEOMETRIC ENCODING
========================================================================

Objective: Force cage-breaking by using GEOMETRIC input encodings
where human algebraic variables are provably suboptimal.

KEY INSIGHT from D1:
--------------------
Cage-breaking requires GEOMETRIC encoding (fields, patterns, trajectories),
NOT just high dimensionality or chaos.

This experiment tests 3 "representation traps" with geometric encodings:

1. HIDDEN SYMMETRY (Spherical Wave Field)
   - Input: 2D wave interference pattern on grid (geometric)
   - Hidden: Radial symmetry f(r) where r = sqrt(x^2 + y^2)
   - Trap: Grid has no explicit r coordinate
   - Expected: BROKEN cage (model discovers r internally)

2. TRAJECTORY ENERGY MANIFOLD
   - Input: Phase space trajectory image (theta vs omega over time)
   - Hidden: Energy contour E(theta, omega) = const
   - Trap: Image has no explicit energy coordinate
   - Expected: BROKEN cage (model discovers energy manifold)

3. TOPOLOGICAL INVARIANT (Velocity Field)
   - Input: Velocity field [vx, vy] on 16x16 grid (geometric)
   - Hidden: Winding number W in {-2, -1, 0, 1, 2}
   - Trap: W requires global line integral
   - Expected: BROKEN cage (model discovers topological structure)

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

Date: November 27, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import json
import os

# Set random seeds
np.random.seed(42)

# ============================================================================
# OPTICAL CHAOS MACHINE (from D1, validated)
# ============================================================================

class OpticalChaosMachine:
    """FFT-based optical chaos reservoir."""
    def __init__(self, n_features=4096, brightness=0.001, random_state=42):
        self.n_features = n_features
        self.brightness = brightness
        self.random_state = random_state
        self.optical_matrix = None
        self.readout = Ridge(alpha=0.1)
        self.scaler = MinMaxScaler()

    def _initialize_optical_matrix(self, n_inputs):
        rng = np.random.RandomState(self.random_state)
        self.optical_matrix = rng.randn(n_inputs, self.n_features) / np.sqrt(n_inputs)

    def _optical_interference(self, X):
        if self.optical_matrix is None:
            self._initialize_optical_matrix(X.shape[1])

        light_field = X @ self.optical_matrix
        interference_pattern = np.fft.rfft(light_field, axis=1)
        intensity = np.abs(interference_pattern)**2
        intensity = np.tanh(intensity * self.brightness)
        return intensity

    def get_features(self, X):
        return self._optical_interference(X)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        features = self._optical_interference(X_scaled)
        self.readout.fit(features, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        features = self._optical_interference(X_scaled)
        return self.readout.predict(features)


class OpticalChaosClassifier:
    """Same architecture but for classification."""
    def __init__(self, n_features=4096, brightness=0.001, random_state=42):
        self.n_features = n_features
        self.brightness = brightness
        self.random_state = random_state
        self.optical_matrix = None
        self.readout = LogisticRegression(max_iter=1000, random_state=random_state)
        self.scaler = MinMaxScaler()

    def _initialize_optical_matrix(self, n_inputs):
        rng = np.random.RandomState(self.random_state)
        self.optical_matrix = rng.randn(n_inputs, self.n_features) / np.sqrt(n_inputs)

    def _optical_interference(self, X):
        if self.optical_matrix is None:
            self._initialize_optical_matrix(X.shape[1])

        light_field = X @ self.optical_matrix
        interference_pattern = np.fft.rfft(light_field, axis=1)
        intensity = np.abs(interference_pattern)**2
        intensity = np.tanh(intensity * self.brightness)
        return intensity

    def get_features(self, X):
        return self._optical_interference(X)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        features = self._optical_interference(X_scaled)
        self.readout.fit(features, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        features = self._optical_interference(X_scaled)
        return self.readout.predict(features)


# ============================================================================
# CAGE ANALYSIS TOOL
# ============================================================================

def analyze_cage(model, X_test, hidden_variables_dict):
    """
    Unified cage analysis.

    Args:
        model: Trained model with get_features()
        X_test: Test inputs (unscaled)
        hidden_variables_dict: Dict of {name: values} for hidden variables

    Returns:
        dict with correlations and cage status
    """
    X_scaled = model.scaler.transform(X_test)
    features = model.get_features(X_scaled)

    correlations = {}
    max_corr_per_variable = {}

    for name, values in hidden_variables_dict.items():
        values = np.array(values).flatten()
        if len(values) != len(features):
            continue

        # Check for zero variance
        if np.std(values) < 1e-10:
            max_corr_per_variable[name] = 0.0
            correlations[name] = [0.0]
            continue

        corr_matrix = np.corrcoef(values, features.T)[0, 1:]
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        max_corr = np.max(np.abs(corr_matrix))

        correlations[name] = corr_matrix
        max_corr_per_variable[name] = max_corr

    overall_max = max(max_corr_per_variable.values()) if max_corr_per_variable else 0.0

    if overall_max < 0.5:
        status = 'BROKEN'
    elif overall_max < 0.7:
        status = 'TRANSITION'
    else:
        status = 'LOCKED'

    return {
        'max_correlation': overall_max,
        'correlations_by_variable': max_corr_per_variable,
        'status': status
    }


# ============================================================================
# PROBLEM 1: HIDDEN SYMMETRY (Spherical Wave Field)
# ============================================================================

class SphericalWaveSimulator:
    """
    Geometric encoding: 2D wave field on grid
    Hidden physics: Radially symmetric potential V(r)

    This is a DIRECT TEST of geometric vs algebraic encoding:
    - Algebraic [x, y, z] would remain LOCKED (as D1 showed)
    - Geometric [field pattern] should BREAK the cage
    """
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.name = "Spherical Wave Field"

    def generate_sample(self):
        """
        Generate wave interference pattern on 2D grid.
        Physics: Spherically symmetric source creates radial waves.
        """
        # Random source strength and wavelength
        k = np.random.uniform(0.5, 3.0)  # Wave number
        A = np.random.uniform(0.5, 5.0)  # Amplitude
        phase = np.random.uniform(0, 2*np.pi)

        # Create 2D grid
        x = np.linspace(-5, 5, self.grid_size)
        y = np.linspace(-5, 5, self.grid_size)
        X_grid, Y_grid = np.meshgrid(x, y)

        # Radial distance from center
        r_grid = np.sqrt(X_grid**2 + Y_grid**2) + 0.1  # Avoid r=0

        # Spherical wave: psi(r) = A * sin(k*r + phase) / r
        wave_field = A * np.sin(k * r_grid + phase) / r_grid

        # Target: Total energy (integral over field)
        # E = integral |psi|^2 dA ~ sum of field squared
        energy = np.sum(wave_field**2)

        # Input: Flatten field to 1D vector (GEOMETRIC encoding)
        X_input = wave_field.flatten()  # Shape: (256,) for 16x16 grid

        # Hidden variable for cage analysis: r at center point
        r_center = np.sqrt(X_grid[8,8]**2 + Y_grid[8,8]**2)

        return X_input, energy, {'k': k, 'A': A, 'r_center': r_center}

    def generate_dataset(self, n_samples):
        X_list = []
        y_list = []
        metadata_list = []

        for _ in range(n_samples):
            X, y, meta = self.generate_sample()
            X_list.append(X)
            y_list.append(y)
            metadata_list.append(meta)

        return np.array(X_list), np.array(y_list), metadata_list


# ============================================================================
# PROBLEM 2: TRAJECTORY ENERGY MANIFOLD
# ============================================================================

class TrajectoryEnergySimulator:
    """
    Geometric encoding: Phase space trajectory as image
    Hidden physics: Energy manifold E(theta, omega) = const

    Converts pendulum dynamics into visual pattern.
    Model must discover energy contours from trajectory shape.
    """
    def __init__(self, trajectory_length=50, grid_size=16):
        self.trajectory_length = trajectory_length
        self.grid_size = grid_size
        self.name = "Trajectory Energy Manifold"

    def pendulum_dynamics(self, theta0, omega0, damping, time_points):
        """Simple damped pendulum."""
        trajectory = []
        theta = theta0
        omega = omega0
        dt = 0.05

        for _ in range(time_points):
            trajectory.append([theta, omega])

            # Dynamics: theta' = omega, omega' = -sin(theta) - damping*omega
            theta += omega * dt
            omega += (-np.sin(theta) - damping * omega) * dt

        return np.array(trajectory)

    def trajectory_to_image(self, trajectory):
        """Convert trajectory to 2D image (heat map)."""
        img = np.zeros((self.grid_size, self.grid_size))

        # Map theta to [-pi, pi] -> [0, grid_size-1]
        # Map omega to [-3, 3] -> [0, grid_size-1]
        theta_vals = trajectory[:, 0]
        omega_vals = trajectory[:, 1]

        theta_idx = ((theta_vals + np.pi) / (2*np.pi) * (self.grid_size - 1)).astype(int)
        omega_idx = ((omega_vals + 3) / 6 * (self.grid_size - 1)).astype(int)

        theta_idx = np.clip(theta_idx, 0, self.grid_size - 1)
        omega_idx = np.clip(omega_idx, 0, self.grid_size - 1)

        # Increment image at trajectory points
        for i, j in zip(theta_idx, omega_idx):
            img[j, i] += 1

        # Gaussian blur to make continuous field
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=0.8)

        return img

    def calculate_energy(self, theta, omega, damping):
        """Mechanical energy (approximate, ignoring damping loss)."""
        KE = 0.5 * omega**2
        PE = 1 - np.cos(theta)  # Potential for pendulum
        return KE + PE

    def generate_sample(self):
        """Generate trajectory image."""
        theta0 = np.random.uniform(-np.pi, np.pi)
        omega0 = np.random.uniform(-2, 2)
        damping = np.random.uniform(0.01, 0.3)

        trajectory = self.pendulum_dynamics(theta0, omega0, damping, self.trajectory_length)
        img = self.trajectory_to_image(trajectory)

        # Input: Flattened image (GEOMETRIC encoding)
        X_input = img.flatten()

        # Target: Initial energy
        energy = self.calculate_energy(theta0, omega0, damping)

        # Hidden variables
        hidden = {
            'theta0': theta0,
            'omega0': omega0,
            'damping': damping,
            'energy': energy
        }

        return X_input, energy, hidden

    def generate_dataset(self, n_samples):
        X_list = []
        y_list = []
        metadata_list = []

        for _ in range(n_samples):
            X, y, meta = self.generate_sample()
            X_list.append(X)
            y_list.append(y)
            metadata_list.append(meta)

        return np.array(X_list), np.array(y_list), metadata_list


# ============================================================================
# PROBLEM 3: TOPOLOGICAL INVARIANT (Velocity Field)
# ============================================================================

class TopologicalInvariantSimulator:
    """
    Geometric encoding: Velocity field on grid
    Hidden physics: Winding number (topological invariant)

    This is ALREADY well-designed (from D1 plan).
    Vortex dynamics with discrete topological classes.
    """
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.name = "Topological Invariant"

    def create_vortex_field(self, center_x, center_y, strength, winding):
        """Create velocity field with vortex."""
        x = np.linspace(-1, 1, self.grid_size)
        y = np.linspace(-1, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)

        # Shift to vortex center
        dx = X - center_x
        dy = Y - center_y
        r = np.sqrt(dx**2 + dy**2) + 0.01

        # Vortex velocity: tangential
        # v_theta = strength / r, convert to Cartesian
        vx = -strength * winding * dy / r**2
        vy = strength * winding * dx / r**2

        return vx, vy

    def calculate_winding_number(self, vx, vy):
        """
        Calculate topological winding number.
        Uses discrete circulation around boundary.
        """
        # Simple approximation: sum of curl over domain
        curl = np.gradient(vy, axis=1) - np.gradient(vx, axis=0)
        W = np.sum(curl) / (2 * np.pi)
        return int(np.round(W))

    def generate_sample(self):
        """Generate multi-vortex field."""
        # Random number of vortices (1-3)
        n_vortices = np.random.randint(1, 4)

        vx_total = np.zeros((self.grid_size, self.grid_size))
        vy_total = np.zeros((self.grid_size, self.grid_size))

        total_winding = 0

        for _ in range(n_vortices):
            center_x = np.random.uniform(-0.7, 0.7)
            center_y = np.random.uniform(-0.7, 0.7)
            strength = np.random.uniform(0.5, 2.0)
            winding = np.random.choice([-1, 1])  # Clockwise or counterclockwise

            vx, vy = self.create_vortex_field(center_x, center_y, strength, winding)
            vx_total += vx
            vy_total += vy
            total_winding += winding

        # Clamp winding to {-2, -1, 0, 1, 2}
        total_winding = np.clip(total_winding, -2, 2)

        # Input: Concatenate vx and vy fields (GEOMETRIC encoding)
        X_input = np.concatenate([vx_total.flatten(), vy_total.flatten()])

        # Target: Winding number (classification)
        winding_class = total_winding + 2  # Map {-2,-1,0,1,2} -> {0,1,2,3,4}

        # Hidden: Individual vortex parameters
        hidden = {
            'winding_number': total_winding,
            'n_vortices': n_vortices
        }

        return X_input, winding_class, hidden

    def generate_dataset(self, n_samples):
        X_list = []
        y_list = []
        metadata_list = []

        for _ in range(n_samples):
            X, y, meta = self.generate_sample()
            X_list.append(X)
            y_list.append(y)
            metadata_list.append(meta)

        return np.array(X_list), np.array(y_list), metadata_list


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_problem(simulator, problem_num, is_classification=False, n_train=2000, n_test=500):
    """Run single problem."""
    print(f"\n{'='*70}")
    print(f"PROBLEM {problem_num}: {simulator.name}")
    print(f"{'='*70}")

    # Generate data
    print(f"\n[1/5] Generating datasets...")
    X_train, y_train, meta_train = simulator.generate_dataset(n_train)
    X_test, y_test, meta_test = simulator.generate_dataset(n_test)

    print(f"  Input shape: {X_train.shape}")
    print(f"  Output range: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")

    # Train model
    print(f"\n[2/5] Training optical chaos model...")
    if is_classification:
        model = OpticalChaosClassifier(n_features=4096, brightness=0.001)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        metric_name = "Accuracy"
    else:
        model = OpticalChaosMachine(n_features=4096, brightness=0.001)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        metric_name = "R2"

    print(f"  {metric_name}: {score:.4f}")

    # Cage analysis
    print(f"\n[3/5] Performing cage analysis...")

    # Extract hidden variables for cage analysis
    if problem_num == 1:  # Spherical wave
        hidden_vars = {
            'k': [m['k'] for m in meta_test],
            'A': [m['A'] for m in meta_test],
            'r_center': [m['r_center'] for m in meta_test]
        }
    elif problem_num == 2:  # Trajectory
        hidden_vars = {
            'theta0': [m['theta0'] for m in meta_test],
            'omega0': [m['omega0'] for m in meta_test],
            'energy': [m['energy'] for m in meta_test]
        }
    else:  # Topological
        hidden_vars = {
            'winding': [m['winding_number'] for m in meta_test],
            'n_vortices': [m['n_vortices'] for m in meta_test]
        }

    cage_result = analyze_cage(model, X_test, hidden_vars)

    print(f"\n[CAGE ANALYSIS]")
    print(f"  Status: {cage_result['status']}")
    print(f"  Max Correlation: {cage_result['max_correlation']:.4f}")
    for var_name, corr in cage_result['correlations_by_variable'].items():
        print(f"    {var_name}: {corr:.4f}")

    # Visualization
    print(f"\n[4/5] Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Predictions
    if is_classification:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        axes[0].imshow(cm, cmap='Blues')
        axes[0].set_title(f'Confusion Matrix\n Acc={score:.3f}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
    else:
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0].set_xlabel('True')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(f'{metric_name}={score:.4f}')
        axes[0].grid(True, alpha=0.3)

    # Plot 2: Sample input visualization
    sample_input = X_test[0].reshape(int(np.sqrt(len(X_test[0]))), -1) if len(X_test[0]) == 256 else X_test[0].reshape(16, -1)
    axes[1].imshow(sample_input, cmap='viridis')
    axes[1].set_title('Sample Input\n(Geometric Encoding)')
    axes[1].axis('off')

    # Plot 3: Cage analysis
    var_names = list(cage_result['correlations_by_variable'].keys())
    corr_vals = list(cage_result['correlations_by_variable'].values())
    axes[2].bar(var_names, corr_vals, color='steelblue')
    axes[2].axhline(0.5, color='red', linestyle='--', label='Cage Threshold')
    axes[2].axhline(0.7, color='orange', linestyle='--', label='Locked Threshold')
    axes[2].set_ylabel('Max Correlation')
    axes[2].set_title(f"Cage: {cage_result['status']}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/problem_{problem_num}_{simulator.name.replace(" ", "_")}.png', dpi=150)
    plt.close()

    print(f"\n[5/5] Complete!")

    return {
        'problem': problem_num,
        'name': simulator.name,
        'score': score,
        'metric': metric_name,
        'cage_status': cage_result['status'],
        'max_correlation': cage_result['max_correlation'],
        'correlations': cage_result['correlations_by_variable']
    }


def run_all_problems():
    """Run all 3 geometric forcing problems."""
    print("\n" + "="*70)
    print("EXPERIMENT D2: FORCING EMERGENT REPRESENTATIONS")
    print("Via Geometric Encoding (D1 Insight Applied)")
    print("="*70)

    results = []

    # Problem 1: Spherical Wave
    print("\n" + "="*70)
    print("Testing geometric encoding: 2D wave field")
    print("="*70)
    sim1 = SphericalWaveSimulator(grid_size=16)
    result1 = run_problem(sim1, 1, is_classification=False)
    results.append(result1)

    # Problem 2: Trajectory Energy
    print("\n" + "="*70)
    print("Testing geometric encoding: Phase space trajectory image")
    print("="*70)
    sim2 = TrajectoryEnergySimulator(trajectory_length=50, grid_size=16)
    result2 = run_problem(sim2, 2, is_classification=False)
    results.append(result2)

    # Problem 3: Topological Invariant
    print("\n" + "="*70)
    print("Testing geometric encoding: Velocity field")
    print("="*70)
    sim3 = TopologicalInvariantSimulator(grid_size=16)
    result3 = run_problem(sim3, 3, is_classification=True)
    results.append(result3)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: GEOMETRIC ENCODING EFFECTIVENESS")
    print("="*70)

    print("\n{:<6} {:<30} {:<10} {:<10} {:<15}".format(
        "Prob", "Name", "Score", "Metric", "Cage Status"))
    print("-"*70)

    for r in results:
        print("{:<6} {:<30} {:<10.4f} {:<10} {:<15}".format(
            r['problem'], r['name'][:30], r['score'], r['metric'], r['cage_status']))

    # Analysis
    print("\n" + "="*70)
    print("CAGE-BREAKING ANALYSIS")
    print("="*70)

    broken_count = sum(1 for r in results if r['cage_status'] == 'BROKEN')
    transition_count = sum(1 for r in results if r['cage_status'] == 'TRANSITION')

    print(f"\nResults:")
    print(f"  BROKEN cages: {broken_count}/3")
    print(f"  TRANSITION: {transition_count}/3")
    print(f"  LOCKED: {3 - broken_count - transition_count}/3")

    if broken_count >= 2:
        print("\n[SUCCESS] Geometric encoding CONFIRMED to break cage!")
        print("D1 hypothesis validated: Representation type > Dimensionality")
    elif broken_count >= 1:
        print("\n[PARTIAL] Some geometric encodings break cage")
        print("Further investigation needed on encoding types")
    else:
        print("\n[UNEXPECTED] Geometric encoding did not break cage")
        print("Hypothesis requires revision")

    # Save results
    with open('results/D2_complete_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] Results: results/D2_complete_results.json")

    return results


if __name__ == "__main__":
    # Install scipy if needed for Gaussian blur
    try:
        import scipy
    except ImportError:
        print("Installing scipy for Gaussian blur...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy'])

    results = run_all_problems()
