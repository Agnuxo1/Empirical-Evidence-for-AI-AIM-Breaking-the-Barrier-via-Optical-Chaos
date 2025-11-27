"""
EXPERIMENT D1: COMPLEXITY PHASE TRANSITION
==========================================

Objective: Systematically map the boundary where cage-breaking begins

This experiment implements a 5-level complexity ladder in orbital/dynamical systems:
1. Harmonic Oscillator (4D) - Expect LOCKED
2. Kepler 2-Body (3D) - Expect LOCKED
3. Restricted 3-Body (6D) - Expect TRANSITION
4. Unrestricted 3-Body (18D) - Expect BROKEN
5. N-Body (42-60D) - Expect STRONGLY BROKEN

Key Hypothesis: max_correlation decreases monotonically with complexity

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.integrate import odeint
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)

# ============================================================================
# UNIFIED OPTICAL CHAOS MACHINE (from B1, validated architecture)
# ============================================================================

class OpticalChaosMachine:
    """
    FFT-based optical chaos reservoir for physics learning.
    Fixed reservoir + trainable Ridge readout.
    """
    def __init__(self, n_features=4096, brightness=0.001, random_state=42):
        self.n_features = n_features
        self.brightness = brightness
        self.random_state = random_state
        self.optical_matrix = None
        self.readout = Ridge(alpha=0.1)
        self.scaler = MinMaxScaler()

    def _initialize_optical_matrix(self, n_inputs):
        """Create fixed random projection matrix (light scattering)."""
        rng = np.random.RandomState(self.random_state)
        self.optical_matrix = rng.randn(n_inputs, self.n_features) / np.sqrt(n_inputs)

    def _optical_interference(self, X):
        """
        Simulate optical interference pattern via FFT.

        Process:
        1. Random projection (light scattering)
        2. FFT (wave interference in frequency domain)
        3. Intensity detection |FFT|^2
        4. Nonlinear saturation tanh(brightness * intensity)
        """
        if self.optical_matrix is None:
            self._initialize_optical_matrix(X.shape[1])

        # 1. Light scattering
        light_field = X @ self.optical_matrix

        # 2. Wave interference (FFT)
        interference_pattern = np.fft.rfft(light_field, axis=1)

        # 3. Intensity detection
        intensity = np.abs(interference_pattern)**2

        # 4. Nonlinear saturation
        intensity = np.tanh(intensity * self.brightness)

        return intensity

    def get_features(self, X):
        """Extract internal optical features (for cage analysis)."""
        return self._optical_interference(X)

    def fit(self, X, y):
        """Train readout layer only (reservoir fixed)."""
        X_scaled = self.scaler.fit_transform(X)
        features = self._optical_interference(X_scaled)
        self.readout.fit(features, y)
        return self

    def predict(self, X):
        """Predict using trained readout."""
        X_scaled = self.scaler.transform(X)
        features = self._optical_interference(X_scaled)
        return self.readout.predict(features)


# ============================================================================
# PHYSICS DISCOVERY MODEL WITH AUTOMATED CAGE ANALYSIS
# ============================================================================

class PhysicsDiscoveryModel:
    """
    Unified model with integrated cage analysis.
    Automatically computes correlations with human variables.
    """
    def __init__(self, n_features=4096, brightness=0.001):
        self.chaos_core = OpticalChaosMachine(n_features, brightness)
        self.trained = False

    def fit(self, X, y):
        """Train the model."""
        self.chaos_core.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        """Make predictions."""
        return self.chaos_core.predict(X)

    def get_features(self, X):
        """Extract internal representations."""
        X_scaled = self.chaos_core.scaler.transform(X)
        return self.chaos_core.get_features(X_scaled)

    def cage_status(self, X, verbose=True):
        """
        Automated cage analysis.

        Returns:
            dict: {
                'max_correlation': float,
                'correlations': list,
                'status': 'BROKEN' | 'TRANSITION' | 'LOCKED'
            }
        """
        if not self.trained:
            raise ValueError("Model must be trained before cage analysis")

        features = self.get_features(X)

        # Compute correlation of each input variable with all features
        correlations = []
        for i in range(X.shape[1]):
            # Max absolute correlation with any feature
            corr_matrix = np.corrcoef(X[:, i], features.T)[0, 1:]
            max_corr = np.max(np.abs(corr_matrix))
            correlations.append(max_corr)

        max_correlation = np.max(correlations)

        # Determine cage status
        if max_correlation < 0.5:
            status = 'BROKEN'
        elif max_correlation < 0.7:
            status = 'TRANSITION'
        else:
            status = 'LOCKED'

        if verbose:
            print(f"\n[CAGE ANALYSIS]")
            print(f"  Max correlation: {max_correlation:.4f}")
            print(f"  Status: {status}")
            for i, corr in enumerate(correlations):
                print(f"    Input {i}: {corr:.4f}")

        return {
            'max_correlation': max_correlation,
            'correlations': correlations,
            'status': status
        }


# ============================================================================
# LEVEL 1: HARMONIC OSCILLATOR (4D - Expect LOCKED)
# ============================================================================

class HarmonicOscillatorSimulator:
    """
    Simple harmonic oscillator: x(t) = A * cos(omega*t + phi)

    Inputs: [omega, A, phi, t] (4D)
    Output: x(t) (1D)

    This is fully analytical and low-dimensional.
    Expected: LOCKED cage (model reconstructs variables)
    """
    def __init__(self):
        self.name = "Harmonic Oscillator"
        self.dim = 4

    def generate_sample(self):
        """Generate random sample."""
        omega = np.random.uniform(0.5, 5.0)  # Angular frequency
        A = np.random.uniform(0.5, 5.0)      # Amplitude
        phi = np.random.uniform(0, 2*np.pi)  # Phase
        t = np.random.uniform(0, 10.0)       # Time

        x = A * np.cos(omega * t + phi)

        return np.array([omega, A, phi, t]), x

    def generate_dataset(self, n_samples):
        """Generate dataset."""
        X = []
        y = []
        for _ in range(n_samples):
            x_sample, y_sample = self.generate_sample()
            X.append(x_sample)
            y.append(y_sample)
        return np.array(X), np.array(y)

    def generate_extrapolation_test(self, n_samples):
        """Generate test with extrapolated time values."""
        X = []
        y = []
        for _ in range(n_samples):
            omega = np.random.uniform(0.5, 5.0)
            A = np.random.uniform(0.5, 5.0)
            phi = np.random.uniform(0, 2*np.pi)
            t = np.random.uniform(10.0, 20.0)  # Extrapolated time range

            x = A * np.cos(omega * t + phi)
            X.append([omega, A, phi, t])
            y.append(x)
        return np.array(X), np.array(y)


# ============================================================================
# LEVEL 2: KEPLER 2-BODY (3D - Expect LOCKED)
# ============================================================================

class Kepler2BodySimulator:
    """
    Kepler 2-body orbital mechanics: r(theta) = a(1-e^2)/(1+e*cos(theta))

    Inputs: [a, e, theta] (3D)
    Output: r (1D)

    Known from Exp 10: R²=0.98, max_corr=0.98
    Expected: LOCKED cage
    """
    def __init__(self):
        self.name = "Kepler 2-Body"
        self.dim = 3

    def generate_sample(self):
        """Generate random sample."""
        a = np.random.uniform(0.5, 5.0)        # Semi-major axis
        e = np.random.uniform(0.0, 0.9)        # Eccentricity (< 1 for ellipse)
        theta = np.random.uniform(0, 2*np.pi)  # True anomaly

        r = a * (1 - e**2) / (1 + e * np.cos(theta))

        return np.array([a, e, theta]), r

    def generate_dataset(self, n_samples):
        """Generate dataset."""
        X = []
        y = []
        for _ in range(n_samples):
            x_sample, y_sample = self.generate_sample()
            X.append(x_sample)
            y.append(y_sample)
        return np.array(X), np.array(y)

    def generate_extrapolation_test(self, n_samples):
        """Generate test with larger orbits."""
        X = []
        y = []
        for _ in range(n_samples):
            a = np.random.uniform(5.0, 10.0)  # Larger orbits
            e = np.random.uniform(0.0, 0.9)
            theta = np.random.uniform(0, 2*np.pi)

            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            X.append([a, e, theta])
            y.append(r)
        return np.array(X), np.array(y)


# ============================================================================
# LEVEL 3: RESTRICTED 3-BODY (6D - Expect TRANSITION)
# ============================================================================

class Restricted3BodySimulator:
    """
    Circular restricted 3-body problem (CR3BP).

    Two massive bodies orbit their barycenter, test particle moves in their field.
    Some chaotic regions, no general analytical solution.

    Inputs: [x0, y0, vx0, vy0, mu, t] (6D)
    Output: x(t) (position at time t)

    Expected: TRANSITION (max_corr ~ 0.5-0.7)
    """
    def __init__(self):
        self.name = "Restricted 3-Body"
        self.dim = 6

    def equations(self, state, t, mu):
        """CR3BP equations of motion in rotating frame."""
        x, y, vx, vy = state

        # Distances to primaries
        r1 = np.sqrt((x + mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2)

        # Equations of motion
        ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3

        return [vx, vy, ax, ay]

    def generate_sample(self):
        """Generate random sample."""
        # Mass parameter (mu = m2/(m1+m2))
        mu = np.random.uniform(0.01, 0.3)

        # Initial conditions (avoid primaries)
        x0 = np.random.uniform(-0.5, 0.5)
        y0 = np.random.uniform(-0.5, 0.5)
        vx0 = np.random.uniform(-1.0, 1.0)
        vy0 = np.random.uniform(-1.0, 1.0)

        # Integration time
        t = np.random.uniform(0.1, 2.0)

        # Integrate
        state0 = [x0, y0, vx0, vy0]
        t_eval = np.linspace(0, t, 100)

        try:
            solution = odeint(self.equations, state0, t_eval, args=(mu,))
            x_final = solution[-1, 0]
        except:
            # If integration fails (escape/collision), return initial position
            x_final = x0

        return np.array([x0, y0, vx0, vy0, mu, t]), x_final

    def generate_dataset(self, n_samples):
        """Generate dataset."""
        X = []
        y = []
        for _ in range(n_samples):
            x_sample, y_sample = self.generate_sample()
            X.append(x_sample)
            y.append(y_sample)
        return np.array(X), np.array(y)

    def generate_extrapolation_test(self, n_samples):
        """Generate test with longer integration times."""
        X = []
        y = []
        for _ in range(n_samples):
            mu = np.random.uniform(0.01, 0.3)
            x0 = np.random.uniform(-0.5, 0.5)
            y0 = np.random.uniform(-0.5, 0.5)
            vx0 = np.random.uniform(-1.0, 1.0)
            vy0 = np.random.uniform(-1.0, 1.0)
            t = np.random.uniform(2.0, 4.0)  # Longer time

            state0 = [x0, y0, vx0, vy0]
            t_eval = np.linspace(0, t, 100)

            try:
                solution = odeint(self.equations, state0, t_eval, args=(mu,))
                x_final = solution[-1, 0]
            except:
                x_final = x0

            X.append([x0, y0, vx0, vy0, mu, t])
            y.append(x_final)
        return np.array(X), np.array(y)


# ============================================================================
# LEVEL 4: UNRESTRICTED 3-BODY (18D - Expect BROKEN)
# ============================================================================

class Unrestricted3BodySimulator:
    """
    Full 3-body problem with all masses free.

    Fully chaotic, no general analytical solution.

    Inputs: [m1, m2, m3, x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3, G, t, target_body] (18D)
    Output: x_target(t) (position of target body at time t)

    Expected: BROKEN (max_corr < 0.4)
    """
    def __init__(self):
        self.name = "Unrestricted 3-Body"
        self.dim = 18

    def equations(self, state, t, masses, G):
        """3-body equations of motion."""
        m1, m2, m3 = masses

        # Unpack state
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state

        # Distances
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2) + 1e-10
        r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2) + 1e-10
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2) + 1e-10

        # Accelerations on body 1
        ax1 = G * m2 * (x2-x1) / r12**3 + G * m3 * (x3-x1) / r13**3
        ay1 = G * m2 * (y2-y1) / r12**3 + G * m3 * (y3-y1) / r13**3

        # Accelerations on body 2
        ax2 = G * m1 * (x1-x2) / r12**3 + G * m3 * (x3-x2) / r23**3
        ay2 = G * m1 * (y1-y2) / r12**3 + G * m3 * (y3-y2) / r23**3

        # Accelerations on body 3
        ax3 = G * m1 * (x1-x3) / r13**3 + G * m2 * (x2-x3) / r23**3
        ay3 = G * m1 * (y1-y3) / r13**3 + G * m2 * (y2-y3) / r23**3

        return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

    def generate_sample(self):
        """Generate random sample."""
        # Masses
        m1 = np.random.uniform(0.5, 2.0)
        m2 = np.random.uniform(0.5, 2.0)
        m3 = np.random.uniform(0.5, 2.0)
        masses = [m1, m2, m3]

        # Initial positions (spread out)
        x1, y1 = np.random.uniform(-2, 2, 2)
        x2, y2 = np.random.uniform(-2, 2, 2)
        x3, y3 = np.random.uniform(-2, 2, 2)

        # Initial velocities
        vx1, vy1 = np.random.uniform(-0.5, 0.5, 2)
        vx2, vy2 = np.random.uniform(-0.5, 0.5, 2)
        vx3, vy3 = np.random.uniform(-0.5, 0.5, 2)

        # Physical constants
        G = 1.0  # Normalized
        t = np.random.uniform(0.1, 1.0)

        # Which body to track
        target_body = np.random.randint(0, 3)

        # Integrate
        state0 = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        t_eval = np.linspace(0, t, 50)

        try:
            solution = odeint(self.equations, state0, t_eval, args=(masses, G))
            final_state = solution[-1]
            # Extract target body position
            x_target = final_state[target_body * 2]
        except:
            x_target = state0[target_body * 2]

        # Input vector
        X = np.array([m1, m2, m3, x1, y1, x2, y2, x3, y3,
                      vx1, vy1, vx2, vy2, vx3, vy3, G, t, target_body])

        return X, x_target

    def generate_dataset(self, n_samples):
        """Generate dataset."""
        X = []
        y = []
        for _ in range(n_samples):
            x_sample, y_sample = self.generate_sample()
            X.append(x_sample)
            y.append(y_sample)
        return np.array(X), np.array(y)

    def generate_extrapolation_test(self, n_samples):
        """Generate test with longer times and larger masses."""
        X = []
        y = []
        for _ in range(n_samples):
            m1 = np.random.uniform(2.0, 4.0)  # Larger masses
            m2 = np.random.uniform(2.0, 4.0)
            m3 = np.random.uniform(2.0, 4.0)
            masses = [m1, m2, m3]

            x1, y1 = np.random.uniform(-2, 2, 2)
            x2, y2 = np.random.uniform(-2, 2, 2)
            x3, y3 = np.random.uniform(-2, 2, 2)
            vx1, vy1 = np.random.uniform(-0.5, 0.5, 2)
            vx2, vy2 = np.random.uniform(-0.5, 0.5, 2)
            vx3, vy3 = np.random.uniform(-0.5, 0.5, 2)

            G = 1.0
            t = np.random.uniform(1.0, 2.0)  # Longer time
            target_body = np.random.randint(0, 3)

            state0 = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
            t_eval = np.linspace(0, t, 50)

            try:
                solution = odeint(self.equations, state0, t_eval, args=(masses, G))
                final_state = solution[-1]
                x_target = final_state[target_body * 2]
            except:
                x_target = state0[target_body * 2]

            X_vec = np.array([m1, m2, m3, x1, y1, x2, y2, x3, y3,
                             vx1, vy1, vx2, vy2, vx3, vy3, G, t, target_body])
            X.append(X_vec)
            y.append(x_target)

        return np.array(X), np.array(y)


# ============================================================================
# LEVEL 5: N-BODY (42-60D - Expect STRONGLY BROKEN)
# ============================================================================

class NBodySimulator:
    """
    N-body gravitational system.

    Known from Exp 10: At 36D (N=6), max_corr=0.13, R²=-0.17
    This tests N=7 (42D) to confirm strongly broken cage.

    Inputs: [m1..mN, x1..xN, y1..yN, vx1..vxN, vy1..vyN, G, t] (6N+2 dimensions)
    Output: Total energy at time t

    Expected: STRONGLY BROKEN (max_corr < 0.2)
    """
    def __init__(self, N=7):
        self.N = N
        self.name = f"{N}-Body"
        self.dim = 6*N + 2

    def equations(self, state, t, masses, G):
        """N-body equations of motion."""
        N = len(masses)
        positions = state[:2*N].reshape(N, 2)
        velocities = state[2*N:].reshape(N, 2)

        accelerations = np.zeros_like(positions)

        for i in range(N):
            for j in range(N):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r = np.linalg.norm(r_vec) + 1e-10
                    accelerations[i] += G * masses[j] * r_vec / r**3

        derivatives = np.concatenate([velocities.flatten(), accelerations.flatten()])
        return derivatives

    def calculate_energy(self, masses, positions, velocities, G):
        """Calculate total energy (kinetic + potential)."""
        N = len(masses)

        # Kinetic energy
        KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

        # Potential energy
        PE = 0.0
        for i in range(N):
            for j in range(i+1, N):
                r = np.linalg.norm(positions[i] - positions[j]) + 1e-10
                PE -= G * masses[i] * masses[j] / r

        return KE + PE

    def generate_sample(self):
        """Generate random sample."""
        # Masses
        masses = np.random.uniform(0.5, 2.0, self.N)

        # Initial positions
        positions = np.random.uniform(-3, 3, (self.N, 2))

        # Initial velocities
        velocities = np.random.uniform(-0.3, 0.3, (self.N, 2))

        # Constants
        G = 1.0
        t = np.random.uniform(0.05, 0.5)  # Short time to avoid divergence

        # Integrate
        state0 = np.concatenate([positions.flatten(), velocities.flatten()])
        t_eval = np.linspace(0, t, 30)

        try:
            solution = odeint(self.equations, state0, t_eval, args=(masses, G))
            final_state = solution[-1]
            final_positions = final_state[:2*self.N].reshape(self.N, 2)
            final_velocities = final_state[2*self.N:].reshape(self.N, 2)

            # Calculate total energy
            energy = self.calculate_energy(masses, final_positions, final_velocities, G)
        except:
            # Fallback
            energy = self.calculate_energy(masses, positions, velocities, G)

        # Input vector: [masses, positions_x, positions_y, velocities_x, velocities_y, G, t]
        X = np.concatenate([
            masses,
            positions[:, 0],
            positions[:, 1],
            velocities[:, 0],
            velocities[:, 1],
            [G, t]
        ])

        return X, energy

    def generate_dataset(self, n_samples):
        """Generate dataset."""
        X = []
        y = []
        for _ in range(n_samples):
            x_sample, y_sample = self.generate_sample()
            X.append(x_sample)
            y.append(y_sample)
        return np.array(X), np.array(y)

    def generate_extrapolation_test(self, n_samples):
        """Generate test with more particles or longer times."""
        X = []
        y = []
        for _ in range(n_samples):
            masses = np.random.uniform(0.5, 2.0, self.N)
            positions = np.random.uniform(-3, 3, (self.N, 2))
            velocities = np.random.uniform(-0.3, 0.3, (self.N, 2))
            G = 1.0
            t = np.random.uniform(0.5, 1.0)  # Longer time

            state0 = np.concatenate([positions.flatten(), velocities.flatten()])
            t_eval = np.linspace(0, t, 30)

            try:
                solution = odeint(self.equations, state0, t_eval, args=(masses, G))
                final_state = solution[-1]
                final_positions = final_state[:2*self.N].reshape(self.N, 2)
                final_velocities = final_state[2*self.N:].reshape(self.N, 2)
                energy = self.calculate_energy(masses, final_positions, final_velocities, G)
            except:
                energy = self.calculate_energy(masses, positions, velocities, G)

            X_vec = np.concatenate([
                masses,
                positions[:, 0],
                positions[:, 1],
                velocities[:, 0],
                velocities[:, 1],
                [G, t]
            ])
            X.append(X_vec)
            y.append(energy)

        return np.array(X), np.array(y)


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_level(simulator, level_num, brightness=0.001, n_train=3000, n_test=500):
    """
    Run single complexity level.

    Returns:
        dict: Complete results including cage status
    """
    print(f"\n{'='*70}")
    print(f"LEVEL {level_num}: {simulator.name} ({simulator.dim}D)")
    print(f"{'='*70}")

    # Generate datasets
    print(f"\n[1/5] Generating datasets...")
    X_train, y_train = simulator.generate_dataset(n_train)
    X_test, y_test = simulator.generate_dataset(n_test)
    X_extrap, y_extrap = simulator.generate_extrapolation_test(n_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Extrap: {X_extrap.shape}")
    print(f"  Target range: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")

    # Train model
    print(f"\n[2/5] Training optical chaos model...")
    model = PhysicsDiscoveryModel(n_features=4096, brightness=brightness)
    model.fit(X_train, y_train)

    # Evaluate
    print(f"\n[3/5] Evaluating performance...")
    y_pred_test = model.predict(X_test)
    y_pred_extrap = model.predict(X_extrap)

    r2_test = r2_score(y_test, y_pred_test)
    r2_extrap = r2_score(y_extrap, y_pred_extrap)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"  R2 (test): {r2_test:.4f}")
    print(f"  R2 (extrapolation): {r2_extrap:.4f}")
    print(f"  RMSE (test): {rmse_test:.4f}")

    # Cage analysis
    print(f"\n[4/5] Performing cage analysis...")
    cage_result = model.cage_status(X_test, verbose=True)

    # Visualization
    print(f"\n[5/5] Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Predictions vs Truth
    axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel('True Value')
    axes[0].set_ylabel('Predicted Value')
    axes[0].set_title(f'Test Set (R2={r2_test:.4f})')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Extrapolation
    axes[1].scatter(y_extrap, y_pred_extrap, alpha=0.5, s=10, color='orange')
    axes[1].plot([y_extrap.min(), y_extrap.max()], [y_extrap.min(), y_extrap.max()], 'r--')
    axes[1].set_xlabel('True Value')
    axes[1].set_ylabel('Predicted Value')
    axes[1].set_title(f'Extrapolation (R2={r2_extrap:.4f})')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Correlation Bar Chart
    correlations = cage_result['correlations']
    axes[2].bar(range(len(correlations)), correlations, color='steelblue')
    axes[2].axhline(y=0.5, color='red', linestyle='--', label='Cage Threshold (0.5)')
    axes[2].axhline(y=0.7, color='orange', linestyle='--', label='Locked Threshold (0.7)')
    axes[2].set_xlabel('Input Variable Index')
    axes[2].set_ylabel('Max Correlation with Features')
    axes[2].set_title(f"Cage Status: {cage_result['status']}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/level_{level_num}_{simulator.name.replace(" ", "_").replace("-", "")}.png', dpi=150)
    plt.close()

    print(f"\n[COMPLETE] Level {level_num} finished.")
    print(f"  Status: {cage_result['status']}")
    print(f"  Max Correlation: {cage_result['max_correlation']:.4f}")
    print(f"  R2 Test: {r2_test:.4f}")

    return {
        'level': level_num,
        'name': simulator.name,
        'dimensionality': simulator.dim,
        'r2_test': r2_test,
        'r2_extrapolation': r2_extrap,
        'rmse_test': rmse_test,
        'cage_status': cage_result['status'],
        'max_correlation': cage_result['max_correlation'],
        'correlations': correlations
    }


def run_all_levels():
    """Run all 5 complexity levels."""
    print("\n" + "="*70)
    print("EXPERIMENT D1: COMPLEXITY PHASE TRANSITION")
    print("Systematically mapping the cage-breaking boundary")
    print("="*70)

    # Define simulators
    simulators = [
        (HarmonicOscillatorSimulator(), 1),
        (Kepler2BodySimulator(), 2),
        (Restricted3BodySimulator(), 3),
        (Unrestricted3BodySimulator(), 4),
        (NBodySimulator(N=7), 5)
    ]

    results = []

    for simulator, level_num in simulators:
        result = run_level(simulator, level_num)
        results.append(result)

    # Summary analysis
    print("\n" + "="*70)
    print("SUMMARY: CAGE-BREAKING BOUNDARY ANALYSIS")
    print("="*70)

    print("\n{:<6} {:<25} {:<8} {:<10} {:<15} {:<10}".format(
        "Level", "System", "Dim", "R2", "Max Corr", "Status"))
    print("-"*70)

    for r in results:
        print("{:<6} {:<25} {:<8} {:<10.4f} {:<15.4f} {:<10}".format(
            r['level'], r['name'], r['dimensionality'],
            r['r2_test'], r['max_correlation'], r['cage_status']))

    # Save results
    with open('results/D1_complete_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n[SAVED] Complete results: results/D1_complete_results.json")

    # Analyze transition
    print("\n" + "="*70)
    print("BOUNDARY ANALYSIS")
    print("="*70)

    dims = [r['dimensionality'] for r in results]
    corrs = [r['max_correlation'] for r in results]

    # Find transition point
    for i in range(len(results)-1):
        if results[i]['cage_status'] == 'LOCKED' and results[i+1]['cage_status'] != 'LOCKED':
            print(f"\n[KEY FINDING] Cage breaks between:")
            print(f"  Level {results[i]['level']}: {results[i]['name']} ({results[i]['dimensionality']}D)")
            print(f"  Level {results[i+1]['level']}: {results[i+1]['name']} ({results[i+1]['dimensionality']}D)")
            print(f"\n  Transition occurs in range: {results[i]['dimensionality']}-{results[i+1]['dimensionality']} dimensions")
            break

    # Plot boundary curve
    plt.figure(figsize=(10, 6))
    plt.plot(dims, corrs, 'o-', markersize=10, linewidth=2, label='Observed')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Cage Breaking Threshold (0.5)')
    plt.axhline(y=0.7, color='orange', linestyle='--', label='Cage Locking Threshold (0.7)')
    plt.xlabel('Dimensionality', fontsize=12)
    plt.ylabel('Max Correlation with Human Variables', fontsize=12)
    plt.title('Cage-Breaking Phase Transition', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    for i, r in enumerate(results):
        plt.annotate(r['name'], (dims[i], corrs[i]),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/D1_phase_transition_curve.png', dpi=150)
    plt.close()

    print("\n[SAVED] Phase transition curve: results/D1_phase_transition_curve.png")
    print("\n" + "="*70)
    print("EXPERIMENT D1 COMPLETE")
    print("="*70)

    return results


if __name__ == "__main__":
    results = run_all_levels()
