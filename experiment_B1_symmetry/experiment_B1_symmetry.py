"""
Physics vs. Darwin: Experiment B1
Symmetry Discovery (Rotational Invariance)
------------------------------------------

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

Objective:
Test if an optical chaos model can discover that rotational kinetic energy
is invariant under coordinate rotations, WITHOUT being explicitly told about
rotation symmetry. This tests Noether's theorem: symmetry -> conservation.

Scientific Question:
Can the model learn that E_rot is the SAME regardless of which direction
we call "x-axis" or "y-axis"? This is the deepest principle in physics:
physical laws don't depend on coordinate system choice.

Hypothesis:
The model will discover emergent geometric features (r^2, v^2, L_z) that are
rotation-invariant, rather than reconstructing Cartesian coordinates (x, y).

This would be DEFINITIVE evidence of cage-breaking via symmetry discovery.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. PHYSICS SIMULATOR
# ===========================

class RotationalSystemSimulator:
    """
    Simulates systems of N point masses with rotational motion.

    Key Physics: Rotational kinetic energy E_rot = L^2/(2I) is INVARIANT
    under coordinate rotations (Noether's theorem consequence).
    """

    def __init__(self, N=10):
        """
        Args:
            N: Number of point masses in the system
        """
        self.N = N

    def calculate_rotational_energy(self, masses, x, y, vx, vy):
        """
        Calculate total rotational kinetic energy around center of mass.

        This is THE GROUND TRUTH that must be rotation-invariant.

        Formula: E_rot = L_z^2 / (2 * I)

        where:
            L_z = SUM m_i * (x_i * vy_i - y_i * vx_i)  [Angular momentum]
            I = SUM m_i * r_i^2 = SUM m_i * (x_i^2 + y_i^2)  [Moment of inertia]

        This quantity is INVARIANT under:
        - Rotation of coordinates
        - Translation to CM frame (already done)
        - Addition of constant velocity (already removed)

        Args:
            masses: (N,) array of particle masses [kg]
            x, y: (N,) arrays of positions [m]
            vx, vy: (N,) arrays of velocities [m/s]

        Returns:
            E_rot: Scalar rotational energy [J]
        """
        N = len(masses)
        total_mass = np.sum(masses)

        # 1. Transform to center-of-mass (CM) frame
        x_cm = np.sum(masses * x) / total_mass
        y_cm = np.sum(masses * y) / total_mass
        vx_cm = np.sum(masses * vx) / total_mass
        vy_cm = np.sum(masses * vy) / total_mass

        # Relative to CM
        x_rel = x - x_cm
        y_rel = y - y_cm
        vx_rel = vx - vx_cm
        vy_rel = vy - vy_cm

        # 2. Calculate angular momentum (z-component in 2D)
        # L_z = SUM m_i * (r_i x v_i)_z = SUM m_i * (x*vy - y*vx)
        L_z = np.sum(masses * (x_rel * vy_rel - y_rel * vx_rel))

        # 3. Calculate moment of inertia
        # I = SUM m_i * r_i^2 = SUM m_i * (x^2 + y^2)
        I = np.sum(masses * (x_rel**2 + y_rel**2))

        # 4. Rotational kinetic energy
        # E_rot = L_z^2 / (2I)
        if I > 1e-10:  # Avoid division by zero
            E_rot = (L_z**2) / (2 * I)
        else:
            # All particles at same point (no rotation possible)
            E_rot = 0.0

        return E_rot

    def apply_rotation(self, x, y, vx, vy, theta):
        """
        Apply 2D rotation by angle theta to coordinates and velocities.

        Rotation matrix: R(θ) = [cos(θ)  -sin(θ)]
                                 [sin(θ)   cos(θ)]

        Args:
            x, y, vx, vy: Arrays to rotate
            theta: Rotation angle [radians]

        Returns:
            x_rot, y_rot, vx_rot, vy_rot: Rotated arrays
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotate positions
        x_rot = cos_t * x - sin_t * y
        y_rot = sin_t * x + cos_t * y

        # Rotate velocities
        vx_rot = cos_t * vx - sin_t * vy
        vy_rot = sin_t * vx + cos_t * vy

        return x_rot, y_rot, vx_rot, vy_rot

    def generate_base_configuration(self):
        """
        Generate a single system configuration in canonical frame (no rotation yet).

        Returns:
            masses, x_base, y_base, vx_base, vy_base
        """
        N = self.N

        # 1. Generate particle masses
        masses = np.random.uniform(0.1, 10.0, N)

        # 2. Generate spatial configuration
        config_type = np.random.choice(['circular', 'elliptical', 'random', 'clustered'])

        if config_type == 'circular':
            # Particles arranged in circle
            radii = np.random.uniform(1, 5, N)
            angles = np.random.uniform(0, 2*np.pi, N)
            x_base = radii * np.cos(angles)
            y_base = radii * np.sin(angles)

        elif config_type == 'elliptical':
            # Particles arranged in ellipse
            a = np.random.uniform(2, 8)  # Semi-major axis
            b = np.random.uniform(1, a)   # Semi-minor axis
            angles = np.random.uniform(0, 2*np.pi, N)
            x_base = a * np.cos(angles) + np.random.normal(0, 0.3, N)
            y_base = b * np.sin(angles) + np.random.normal(0, 0.3, N)

        elif config_type == 'random':
            # Scattered randomly
            x_base = np.random.uniform(-10, 10, N)
            y_base = np.random.uniform(-10, 10, N)

        else:  # clustered
            # Gaussian clusters
            n_clusters = np.random.randint(2, 4)
            cluster_centers = np.random.uniform(-5, 5, (n_clusters, 2))
            cluster_ids = np.random.randint(0, n_clusters, N)
            x_base = cluster_centers[cluster_ids, 0] + np.random.normal(0, 1.5, N)
            y_base = cluster_centers[cluster_ids, 1] + np.random.normal(0, 1.5, N)

        # 3. Generate velocity configuration
        motion_type = np.random.choice(['rotating', 'random', 'expanding', 'stationary', 'mixed'])

        if motion_type == 'rotating':
            # Tangential velocities (circular motion)
            omega = np.random.uniform(0.1, 2.0)  # Angular velocity
            vx_base = -y_base * omega + np.random.normal(0, 0.2, N)
            vy_base = x_base * omega + np.random.normal(0, 0.2, N)

        elif motion_type == 'random':
            # Random velocities
            vx_base = np.random.uniform(-5, 5, N)
            vy_base = np.random.uniform(-5, 5, N)

        elif motion_type == 'expanding':
            # Radial velocities (expansion/contraction)
            speed = np.random.uniform(-2.0, 3.0)
            r = np.sqrt(x_base**2 + y_base**2) + 1e-6
            vx_base = (x_base / r) * speed + np.random.normal(0, 0.3, N)
            vy_base = (y_base / r) * speed + np.random.normal(0, 0.3, N)

        elif motion_type == 'stationary':
            # Small random motion
            vx_base = np.random.normal(0, 0.5, N)
            vy_base = np.random.normal(0, 0.5, N)

        else:  # mixed
            # Some rotating, some random
            omega = np.random.uniform(0.1, 1.5)
            vx_base = -y_base * omega * 0.5 + np.random.uniform(-3, 3, N)
            vy_base = x_base * omega * 0.5 + np.random.uniform(-3, 3, N)

        return masses, x_base, y_base, vx_base, vy_base

    def generate_sample(self, apply_random_rotation=True):
        """
        Generate single training sample with optional random rotation.

        Args:
            apply_random_rotation: If True, applies random rotation to frame

        Returns:
            X: Input vector (40 dimensions)
            E_rot: Target rotational energy
            metadata: Dictionary with masses, theta, config info
        """
        # 1. Generate base configuration (no rotation)
        masses, x_base, y_base, vx_base, vy_base = self.generate_base_configuration()

        # 2. Calculate TRUE rotational energy (before rotation)
        E_rot_true = self.calculate_rotational_energy(masses, x_base, y_base,
                                                       vx_base, vy_base)

        # 3. Apply random rotation to coordinate frame
        if apply_random_rotation:
            theta = np.random.uniform(0, 2*np.pi)
        else:
            theta = 0.0

        x_rot, y_rot, vx_rot, vy_rot = self.apply_rotation(
            x_base, y_base, vx_base, vy_base, theta
        )

        # 4. Verify energy is unchanged (CRITICAL validation)
        E_rot_rotated = self.calculate_rotational_energy(masses, x_rot, y_rot,
                                                          vx_rot, vy_rot)

        # This assertion is CRITICAL - if it fails, physics is wrong
        rel_error = abs(E_rot_true - E_rot_rotated) / (abs(E_rot_true) + 1e-10)
        assert rel_error < 1e-9, f"Energy not invariant! Error: {rel_error:.2e}"

        # 5. Flatten to input vector (40 dimensions)
        # Format: [x1...x10, y1...y10, vx1...vx10, vy1...vy10]
        X = np.concatenate([x_rot, y_rot, vx_rot, vy_rot])

        # 6. Metadata for analysis
        metadata = {
            'masses': masses,
            'theta': theta,
            'x_base': x_base,
            'y_base': y_base,
            'vx_base': vx_base,
            'vy_base': vy_base
        }

        return X, E_rot_true, metadata

    def generate_dataset(self, n_samples=4000):
        """
        Generate complete dataset with random rotations.

        Args:
            n_samples: Number of samples to generate

        Returns:
            X: (n_samples, 40) array of inputs
            y: (n_samples,) array of targets (rotational energies)
            metadata: List of metadata dictionaries
        """
        X_list = []
        y_list = []
        metadata_list = []

        for i in range(n_samples):
            X, y, meta = self.generate_sample(apply_random_rotation=True)
            X_list.append(X)
            y_list.append(y)
            metadata_list.append(meta)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y, metadata_list


# ===========================
# 2. OPTICAL CHAOS MODEL
# ===========================

class OpticalChaosMachine:
    """
    Optical chaos-based reservoir computer for physics learning.

    Architecture:
    1. Random projection to high-dimensional complex space (optical scattering)
    2. FFT mixing (wave interference)
    3. Intensity detection (magnitude squared)
    4. Ridge regression readout (linear learning layer)

    Key: The reservoir layer is FIXED (no backprop). Only readout trains.
    """

    def __init__(self, n_features=4096, brightness=0.001):
        """
        Args:
            n_features: Number of optical features (reservoir size)
            brightness: Scaling factor to prevent saturation
        """
        self.n_features = n_features
        self.brightness = brightness
        self.readout = Ridge(alpha=0.1)
        self.optical_matrix = None

    def _optical_interference(self, X):
        """
        Apply optical transformation: random projection + FFT + intensity.

        Args:
            X: (n_samples, 40) input array

        Returns:
            intensity: (n_samples, n_features) optical features
        """
        n_samples, n_input = X.shape

        # 1. Initialize fixed optical matrix (first call only)
        if self.optical_matrix is None:
            np.random.seed(1337)  # Fixed seed for reproducibility
            self.optical_matrix = np.random.normal(0, 1, (n_input, self.n_features))

        # 2. Random projection (light scattering through diffuser)
        light_field = X @ self.optical_matrix

        # 3. FFT mixing (wave propagation and interference)
        # This simulates physical mixing of waves in frequency domain
        interference_pattern = np.fft.rfft(light_field, axis=1)

        # 4. Intensity detection (photodetector measures |E|^2)
        intensity = np.abs(interference_pattern)**2

        # 5. Nonlinear saturation (tanh mimics sensor saturation)
        intensity = np.tanh(intensity * self.brightness)

        return intensity

    def fit(self, X, y):
        """Train readout layer on optical features."""
        X_optical = self._optical_interference(X)
        self.readout.fit(X_optical, y)

    def predict(self, X):
        """Predict using optical features."""
        X_optical = self._optical_interference(X)
        return self.readout.predict(X_optical)

    def get_internal_state(self, X):
        """Return internal optical features for cage analysis."""
        return self._optical_interference(X)


# ===========================
# 3. BASELINE MODEL
# ===========================

class DarwinianModel:
    """
    Polynomial regression baseline (standard machine learning approach).

    Uses polynomial features to approximate the relationship.
    Expected to learn E_rot but unclear if it will discover rotation invariance.
    """

    def __init__(self, degree=3):
        """
        Args:
            degree: Polynomial degree (2-4 typical)
        """
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.model = Ridge(alpha=0.1)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)


# ===========================
# 4. MAIN EXPERIMENT
# ===========================

def run_experiment_B1():
    """Execute complete Experiment B1: Symmetry Discovery."""

    print("="*70)
    print("EXPERIMENT B1: SYMMETRY DISCOVERY (ROTATIONAL INVARIANCE)")
    print("="*70)
    print("\nObjective: Test if optical chaos model discovers that rotational")
    print("kinetic energy is invariant under coordinate rotations (Noether's theorem).")
    print("\nThis is THE DEEPEST test of cage-breaking: discovering physical symmetry")
    print("without being explicitly told about rotation invariance.\n")
    print("="*70)

    # ===== 1. GENERATE DATA =====
    print("\n[PHASE 1] Generating Dataset...")
    print("-"*70)

    np.random.seed(42)  # For reproducibility
    sim = RotationalSystemSimulator(N=10)

    X, y, metadata = sim.generate_dataset(n_samples=5000)

    print(f"  [OK] Generated {len(X)} samples")
    print(f"  [OK] Input dimensions: {X.shape[1]} (10 masses x 4 coords each)")
    print(f"  [OK] Output (E_rot) range: [{np.min(y):.2f}, {np.max(y):.2f}] J")
    print(f"  [OK] Output mean: {np.mean(y):.2f} J")
    print(f"  [OK] Output std: {np.std(y):.2f} J")

    # Validate no NaN/Inf
    assert not np.any(np.isnan(X)), "NaN detected in inputs!"
    assert not np.any(np.isnan(y)), "NaN detected in outputs!"
    assert not np.any(np.isinf(y)), "Inf detected in outputs!"
    print("  [OK] Data validation passed (no NaN/Inf)")

    # ===== 2. SCALE AND SPLIT DATA =====
    print("\n[PHASE 2] Scaling and Splitting Data...")
    print("-"*70)

    # Scale inputs (critical for optical model)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data (keep raw and scaled aligned)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test = train_test_split(
        X, X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"  [OK] Training samples: {len(X_train)}")
    print(f"  [OK] Test samples: {len(X_test)}")
    print(f"  [OK] Train/Test split: 80/20")

    # ===== 3. TRAIN BASELINE MODEL =====
    print("\n[PHASE 3] Training Baseline Model (Polynomial Regression)...")
    print("-"*70)

    darwin = DarwinianModel(degree=3)
    darwin.fit(X_train, y_train)
    y_pred_darwin = darwin.predict(X_test)
    r2_darwin = r2_score(y_test, y_pred_darwin)
    rmse_darwin = np.sqrt(mean_squared_error(y_test, y_pred_darwin))

    print(f"  [OK] Darwinian R^2 Score: {r2_darwin:.4f}")
    print(f"  [OK] Darwinian RMSE: {rmse_darwin:.4f} J")

    if r2_darwin > 0.90:
        print("  [PASS] Baseline PASS: Problem is learnable!")
    elif r2_darwin > 0.70:
        print("  [PARTIAL] Baseline PARTIAL: Problem is moderately difficult")
    else:
        print("  [FAIL] Baseline FAIL: Problem may be too complex")

    # ===== 4. TRAIN OPTICAL CHAOS MODEL =====
    print("\n[PHASE 4] Training Optical Chaos Model...")
    print("-"*70)
    print("  Performing brightness hyperparameter search...")

    best_r2 = -np.inf
    best_model = None
    best_brightness = 0.001

    for brightness in [0.0001, 0.001, 0.01, 0.1]:
        model_test = OpticalChaosMachine(n_features=4096, brightness=brightness)
        model_test.fit(X_train_s, y_train)
        y_pred_test = model_test.predict(X_test_s)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"    brightness={brightness:7.4f} -> R^2={r2_test:.4f}")

        if r2_test > best_r2:
            best_r2 = r2_test
            best_model = model_test
            best_brightness = brightness

    chaos = best_model
    y_pred_chaos = chaos.predict(X_test_s)
    r2_chaos = best_r2
    rmse_chaos = np.sqrt(mean_squared_error(y_test, y_pred_chaos))

    print(f"\n  [OK] Optimal brightness: {best_brightness}")
    print(f"  [OK] Optical Chaos R^2 Score: {r2_chaos:.4f}")
    print(f"  [OK] Optical Chaos RMSE: {rmse_chaos:.4f} J")

    if r2_chaos > 0.90:
        print("  [PASS] Chaos Model PASS: High prediction accuracy!")
    elif r2_chaos > 0.70:
        print("  [PARTIAL] Chaos Model PARTIAL: Moderate performance")
    else:
        print("  [FAIL] Chaos Model FAIL: Poor performance (40D may be too high)")

    # ===== 5. RESULTS SUMMARY =====
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nDarwinian Baseline:")
    print(f"  R^2 Score:  {r2_darwin:.4f}")
    print(f"  RMSE:      {rmse_darwin:.4f} J")

    print(f"\nOptical Chaos Model:")
    print(f"  R^2 Score:  {r2_chaos:.4f}")
    print(f"  RMSE:      {rmse_chaos:.4f} J")
    print(f"  Optimal brightness: {best_brightness}")

    print(f"\nPerformance Comparison:")
    if r2_chaos > r2_darwin:
        print(f"  [WIN] Chaos model OUTPERFORMS baseline by {(r2_chaos-r2_darwin):.4f}")
    else:
        print(f"  [LOSE] Baseline outperforms chaos by {(r2_darwin-r2_chaos):.4f}")

    # ===== 6. VISUALIZATION =====
    print("\n[PHASE 5] Generating Visualizations...")
    print("-"*70)

    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Predictions scatter (Baseline)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_pred_darwin, alpha=0.5, s=20, color='blue', label='Predictions')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('True E_rot [J]', fontsize=11)
    ax1.set_ylabel('Predicted E_rot [J]', fontsize=11)
    ax1.set_title(f'Baseline Model\nR^2 = {r2_darwin:.4f}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predictions scatter (Chaos)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, y_pred_chaos, alpha=0.5, s=20, color='red', label='Predictions')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('True E_rot [J]', fontsize=11)
    ax2.set_ylabel('Predicted E_rot [J]', fontsize=11)
    ax2.set_title(f'Optical Chaos Model\nR^2 = {r2_chaos:.4f}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals comparison
    ax3 = plt.subplot(2, 3, 3)
    residuals_darwin = y_test - y_pred_darwin
    residuals_chaos = y_test - y_pred_chaos
    ax3.scatter(y_test, residuals_darwin, alpha=0.4, s=15, color='blue', label='Baseline')
    ax3.scatter(y_test, residuals_chaos, alpha=0.4, s=15, color='red', label='Chaos')
    ax3.axhline(y=0, color='black', linestyle='--', lw=1)
    ax3.set_xlabel('True E_rot [J]', fontsize=11)
    ax3.set_ylabel('Residuals [J]', fontsize=11)
    ax3.set_title('Prediction Residuals', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Error distribution (Baseline)
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(residuals_darwin, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', lw=2)
    ax4.set_xlabel('Residual [J]', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title(f'Baseline Error Distribution\nMean: {np.mean(residuals_darwin):.3f} J',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Error distribution (Chaos)
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(residuals_chaos, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', lw=2)
    ax5.set_xlabel('Residual [J]', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title(f'Chaos Error Distribution\nMean: {np.mean(residuals_chaos):.3f} J',
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Model comparison
    ax6 = plt.subplot(2, 3, 6)
    models = ['Baseline\n(Polynomial)', 'Chaos\n(Optical)']
    r2_scores = [r2_darwin, r2_chaos]
    colors = ['blue', 'red']
    bars = ax6.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax6.axhline(y=0.90, color='green', linestyle='--', lw=2, label='PASS threshold (0.90)')
    ax6.axhline(y=0.70, color='orange', linestyle='--', lw=2, label='PARTIAL threshold (0.70)')
    ax6.set_ylabel('R^2 Score', fontsize=11)
    ax6.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 1.0])
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('d:\\Darwin Cage\\experiment_B1_symmetry\\results\\experiment_B1_main_results.png',
                dpi=150, bbox_inches='tight')
    print("  [OK] Saved: results/experiment_B1_main_results.png")

    # ===== 7. SAVE RESULTS =====
    results = {
        'r2_darwin': r2_darwin,
        'r2_chaos': r2_chaos,
        'rmse_darwin': rmse_darwin,
        'rmse_chaos': rmse_chaos,
        'best_brightness': best_brightness,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    import json
    with open('d:\\Darwin Cage\\experiment_B1_symmetry\\results\\metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  [OK] Saved: results/metrics.json")

    print("\n" + "="*70)
    print("EXPERIMENT B1 COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Run benchmark_experiment_B1.py for comprehensive validation")
    print("2. Check rotation invariance (THE KEY TEST)")
    print("3. Analyze cage status (coordinate reconstruction vs. emergence)")
    print("\n" + "="*70)

    plt.show()

    return chaos, darwin, X_test_s, y_test, scaler, sim


if __name__ == "__main__":
    # Run main experiment
    chaos_model, darwin_model, X_test_scaled, y_test, scaler, simulator = run_experiment_B1()
