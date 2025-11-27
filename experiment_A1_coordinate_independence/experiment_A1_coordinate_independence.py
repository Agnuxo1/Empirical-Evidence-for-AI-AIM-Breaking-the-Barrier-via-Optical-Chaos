"""
Experiment A1: Coordinate Independence (The Twisted Cage)
---------------------------------------------------------
Testing if AI can learn physics in a 'twisted' coordinate system
where human-derived mathematical simplicity is destroyed.

System: Double Pendulum (Chaotic)
Transform: Non-linear diffeomorphism (The Twist)
Models: Darwinian (Polynomial) vs Chaos (Optical Reservoir)

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
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time

# --- 1. PHYSICAL SIMULATOR: DOUBLE PENDULUM ---
class DoublePendulumSimulator:
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g

    def derivatives(self, t, state):
        theta1, theta2, p1, p2 = state
        
        # Hamiltonian equations of motion (canonical coordinates)
        # Denominator terms
        delta = theta1 - theta2
        den1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * np.cos(delta)**2
        den2 = (self.l2 / self.l1) * den1

        # Angular velocities (theta_dot)
        theta1_dot = (self.l2 * p1 - self.l1 * p2 * np.cos(delta)) / (self.l1**2 * self.l2 * (self.m1 + self.m2 * np.sin(delta)**2))
        theta2_dot = (self.l1 * (self.m1 + self.m2) * p2 - self.m2 * self.l1 * p1 * np.cos(delta)) / (self.m2 * self.l1 * self.l2**2 * (self.m1 + self.m2 * np.sin(delta)**2))
        
        # Simplified Lagrangian formulation for robustness (standard implementation)
        # Using standard equations for d(theta)/dt and d(omega)/dt
        # State: [theta1, theta2, omega1, omega2]
        # BUT we want canonical [theta1, theta2, p1, p2] for Hamiltonian structure?
        # Let's stick to the standard Lagrangian form [theta1, theta2, w1, w2] for simulation stability
        # and then map to whatever we want.
        
        return self._lagrangian_derivatives(t, state)

    def _lagrangian_derivatives(self, t, state):
        th1, th2, w1, w2 = state
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        dth1 = w1
        dth2 = w2
        
        delta = th1 - th2
        den = (2*m1 + m2 - m2*np.cos(2*th1 - 2*th2))
        
        dw1 = (-g*(2*m1 + m2)*np.sin(th1) - m2*g*np.sin(th1 - 2*th2) - 2*np.sin(delta)*m2*(w2**2*l2 + w1**2*l1*np.cos(delta))) / (l1*den)
        dw2 = (2*np.sin(delta)*(w1**2*l1*(m1 + m2) + g*(m1 + m2)*np.cos(th1) + w2**2*l2*m2*np.cos(delta))) / (l2*den)
        
        return [dth1, dth2, dw1, dw2]

    def generate_trajectory(self, t_max=10, dt=0.01):
        # Random initial conditions
        th1 = np.random.uniform(-np.pi, np.pi)
        th2 = np.random.uniform(-np.pi, np.pi)
        w1 = np.random.uniform(-2, 2)
        w2 = np.random.uniform(-2, 2)
        
        t_eval = np.arange(0, t_max, dt)
        sol = solve_ivp(self._lagrangian_derivatives, [0, t_max], [th1, th2, w1, w2], t_eval=t_eval, rtol=1e-8)
        
        return sol.y.T  # Shape: (n_steps, 4)

    def generate_dataset(self, n_trajectories=50, t_max=5, dt=0.05):
        X = []
        Y = []
        
        print(f"  Generating {n_trajectories} trajectories...")
        for _ in range(n_trajectories):
            traj = self.generate_trajectory(t_max, dt)
            # Input: State at t
            # Output: State at t+1 (Next Step Prediction)
            X.append(traj[:-1])
            Y.append(traj[1:])
            
        X = np.vstack(X)
        Y = np.vstack(Y)
        return X, Y

# --- 2. THE TWIST (COORDINATE TRANSFORMATION) ---
class TwistedCoordinateSystem:
    def __init__(self):
        # Twist parameters
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.5
        
    def forward(self, state):
        """Standard -> Twisted"""
        # State: [th1, th2, w1, w2]
        th1, th2, w1, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        
        u1 = th1 + self.alpha * np.sin(th2)
        u2 = th2 + self.beta * np.cos(th1)
        v1 = w1 + self.gamma * np.tanh(w2) # Non-linear mixing
        v2 = w2 + 0.2 * th1 * th2          # Mixing position and momentum
        
        return np.column_stack([u1, u2, v1, v2])
    
    def inverse(self, twisted_state):
        """Twisted -> Standard (Numerical Inverse if needed, but we train on twisted directly)"""
        # We don't strictly need the inverse for the learning task, 
        # as we predict u_{t+1} from u_t.
        pass

# --- 3. MODELS ---
class DarwinianModel:
    """Polynomial Regression"""
    def __init__(self, degree=3):
        self.poly = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=0.1)
        
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

class ChaosModel:
    """Optical Reservoir Computing"""
    def __init__(self, n_features=4096, brightness=0.01):
        self.n_features = n_features
        self.brightness = brightness
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=0.01)
        self.reservoir = None
        
    def _transform(self, X):
        if self.reservoir is None:
            np.random.seed(42)
            self.reservoir = np.random.randn(X.shape[1], self.n_features)
            
        # Optical mixing
        field = X @ self.reservoir
        
        # Non-linear encoding (simulating optical interference/intensity)
        # Complex-valued transformation to allow phase interactions
        field_complex = field * (1 + 1j) 
        spectrum = np.fft.fft(field_complex, axis=1)
        intensity = np.abs(spectrum)**2
        
        # Activation
        return np.tanh(intensity * self.brightness)
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        features = self._transform(X_scaled)
        self.readout.fit(features, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        features = self._transform(X_scaled)
        return self.readout.predict(features)

# --- 4. EXPERIMENT EXECUTION ---
def run_experiment_A1():
    print("🌀 STARTING EXPERIMENT A1: COORDINATE INDEPENDENCE")
    print("="*70)
    
    # 1. Generate Data (Standard Frame)
    print("\n[1] Generating Data (Double Pendulum)...")
    sim = DoublePendulumSimulator()
    X_std, Y_std = sim.generate_dataset(n_trajectories=100, t_max=10, dt=0.05)
    print(f"    Standard Data Shape: {X_std.shape}")
    
    # 2. Apply Twist
    print("\n[2] Applying Twisted Coordinate Transformation...")
    twist = TwistedCoordinateSystem()
    X_twist = twist.forward(X_std)
    Y_twist = twist.forward(Y_std)
    print(f"    Twisted Data Shape: {X_twist.shape}")
    
    # Split Data
    test_size = 0.2
    # Standard
    X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(X_std, Y_std, test_size=test_size, random_state=42)
    # Twisted
    X_train_twi, X_test_twi, Y_train_twi, Y_test_twi = train_test_split(X_twist, Y_twist, test_size=test_size, random_state=42)
    
    # 3. Train & Evaluate Models
    results = {}
    
    print("\n[3] Training Models...")
    
    # --- Standard Frame ---
    print("\n    --- Standard Frame (Human-Readable) ---")
    
    # Darwinian
    print("    Training Darwinian Model (Poly Degree 3)...")
    darwin_std = DarwinianModel(degree=3)
    t0 = time.time()
    darwin_std.fit(X_train_std, Y_train_std)
    print(f"    Time: {time.time()-t0:.2f}s")
    r2_darwin_std = r2_score(Y_test_std, darwin_std.predict(X_test_std))
    print(f"    R²: {r2_darwin_std:.4f}")
    
    # Chaos
    print("    Training Chaos Model (Optical Reservoir)...")
    chaos_std = ChaosModel(n_features=4096)
    t0 = time.time()
    chaos_std.fit(X_train_std, Y_train_std)
    print(f"    Time: {time.time()-t0:.2f}s")
    r2_chaos_std = r2_score(Y_test_std, chaos_std.predict(X_test_std))
    print(f"    R²: {r2_chaos_std:.4f}")
    
    # --- Twisted Frame ---
    print("\n    --- Twisted Frame (The 'Ugly' Coordinates) ---")
    
    # Darwinian
    print("    Training Darwinian Model...")
    darwin_twi = DarwinianModel(degree=3)
    darwin_twi.fit(X_train_twi, Y_train_twi)
    r2_darwin_twi = r2_score(Y_test_twi, darwin_twi.predict(X_test_twi))
    print(f"    R²: {r2_darwin_twi:.4f}")
    
    # Chaos
    print("    Training Chaos Model...")
    chaos_twi = ChaosModel(n_features=4096)
    chaos_twi.fit(X_train_twi, Y_train_twi)
    r2_chaos_twi = r2_score(Y_test_twi, chaos_twi.predict(X_test_twi))
    print(f"    R²: {r2_chaos_twi:.4f}")
    
    # 4. Analysis
    print("\n[4] Analysis & Verdict")
    print("="*70)
    
    gap_darwin = r2_darwin_std - r2_darwin_twi
    gap_chaos = r2_chaos_std - r2_chaos_twi
    
    print(f"Darwinian Gap (Std - Twist): {gap_darwin:.4f}")
    print(f"Chaos Gap     (Std - Twist): {gap_chaos:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Standard Prediction (Chaos)
    plt.subplot(1, 2, 1)
    pred_std = chaos_std.predict(X_test_std)
    plt.scatter(Y_test_std[:, 0], pred_std[:, 0], alpha=0.5, s=10, label='Pred')
    plt.plot([Y_test_std.min(), Y_test_std.max()], [Y_test_std.min(), Y_test_std.max()], 'r--', lw=2)
    plt.title(f"Standard Frame (Chaos)\nR² = {r2_chaos_std:.4f}")
    plt.xlabel("True State (Theta1)")
    plt.ylabel("Predicted")
    
    # Plot 2: Twisted Prediction (Chaos)
    plt.subplot(1, 2, 2)
    pred_twi = chaos_twi.predict(X_test_twi)
    plt.scatter(Y_test_twi[:, 0], pred_twi[:, 0], alpha=0.5, s=10, label='Pred', color='orange')
    plt.plot([Y_test_twi.min(), Y_test_twi.max()], [Y_test_twi.min(), Y_test_twi.max()], 'r--', lw=2)
    plt.title(f"Twisted Frame (Chaos)\nR² = {r2_chaos_twi:.4f}")
    plt.xlabel("True Twisted State (u1)")
    plt.ylabel("Predicted")
    
    plt.tight_layout()
    plt.savefig('experiment_A1_results.png')
    print("    Graph saved as 'experiment_A1_results.png'")
    
    # Verdict
    print("\nVERDICT:")
    if gap_chaos < 0.05 and r2_chaos_twi > 0.9:
        print("🔓 CAGE BROKEN: Chaos model is Coordinate Independent!")
        print("   It learned the physics equally well in the twisted frame.")
    elif gap_chaos > 0.1:
        print("🔒 CAGE LOCKED: Chaos model relies on 'nice' coordinates.")
        print("   Performance dropped significantly in the twisted frame.")
    else:
        print("🟡 INCONCLUSIVE: Mixed results.")

if __name__ == "__main__":
    run_experiment_A1()
