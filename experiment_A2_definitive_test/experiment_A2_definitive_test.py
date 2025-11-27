"""
Experiment A2: The Definitive Coordinate Independence Test
-----------------------------------------------------------
Testing if LSTM (proper temporal architecture) can learn chaotic dynamics
in twisted coordinates as well as in standard coordinates.

System: Double Pendulum (Chaotic)
Models: Polynomial (Darwinian) vs LSTM (AI)
Test: Performance gap between standard and twisted coordinates

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# --- 1. PHYSICAL SIMULATOR: DOUBLE PENDULUM ---
class DoublePendulumSimulator:
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g

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

    def generate_trajectory(self, t_max=10, dt=0.05):
        th1 = np.random.uniform(-np.pi, np.pi)
        th2 = np.random.uniform(-np.pi, np.pi)
        w1 = np.random.uniform(-2, 2)
        w2 = np.random.uniform(-2, 2)
        
        t_eval = np.arange(0, t_max, dt)
        sol = solve_ivp(self._lagrangian_derivatives, [0, t_max], [th1, th2, w1, w2], t_eval=t_eval, rtol=1e-8)
        
        return sol.y.T

    def generate_sequences(self, n_trajectories=100, t_max=10, dt=0.05, seq_length=20):
        """Generate sequences for LSTM training"""
        sequences_X = []
        sequences_Y = []
        
        print(f"  Generating {n_trajectories} trajectories...")
        for i in range(n_trajectories):
            if (i+1) % 20 == 0:
                print(f"    Progress: {i+1}/{n_trajectories}")
            traj = self.generate_trajectory(t_max, dt)
            
            # Create overlapping sequences
            for start_idx in range(0, len(traj) - seq_length - 1, 5):
                seq_x = traj[start_idx:start_idx + seq_length]
                seq_y = traj[start_idx + 1:start_idx + seq_length + 1]
                sequences_X.append(seq_x)
                sequences_Y.append(seq_y)
        
        return np.array(sequences_X), np.array(sequences_Y)

# --- 2. THE TWIST ---
class TwistedCoordinateSystem:
    def __init__(self):
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.5
        
    def forward(self, state):
        """Standard -> Twisted"""
        if len(state.shape) == 2:
            th1, th2, w1, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        elif len(state.shape) == 3:
            th1, th2, w1, w2 = state[:, :, 0], state[:, :, 1], state[:, :, 2], state[:, :, 3]
        
        u1 = th1 + self.alpha * np.sin(th2)
        u2 = th2 + self.beta * np.cos(th1)
        v1 = w1 + self.gamma * np.tanh(w2)
        v2 = w2 + 0.2 * th1 * th2
        
        return np.stack([u1, u2, v1, v2], axis=-1)

# --- 3. MODELS ---
class PolynomialModel:
    """Polynomial Regression for 1-step prediction"""
    def __init__(self, degree=3):
        self.poly = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=0.1)
        
    def fit(self, X_seq, Y_seq):
        # Use only last state of sequence for prediction
        X = X_seq[:, -1, :]
        Y = Y_seq[:, -1, :]
        
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, Y)
        
    def predict(self, X_seq):
        X = X_seq[:, -1, :]
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)
    
    def rollout(self, initial_state, n_steps):
        """Multi-step prediction"""
        predictions = [initial_state]
        current = initial_state.reshape(1, -1)
        
        for _ in range(n_steps):
            X_poly = self.poly.transform(current)
            X_scaled = self.scaler.transform(X_poly)
            next_state = self.model.predict(X_scaled)
            predictions.append(next_state[0])
            current = next_state
            
        return np.array(predictions)

class LSTMModel(nn.Module):
    """LSTM for multi-step prediction"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)
        predictions = self.fc(lstm_out)
        return predictions

class LSTMWrapper:
    """Wrapper for LSTM to match sklearn interface"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, epochs=50, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size, hidden_size, num_layers).to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        
    def fit(self, X_seq, Y_seq):
        # Normalize
        X_flat = X_seq.reshape(-1, X_seq.shape[-1])
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(X_seq.shape)
        Y_scaled = self.scaler.transform(Y_seq.reshape(-1, Y_seq.shape[-1])).reshape(Y_seq.shape)
        
        # Convert to torch
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        Y_tensor = torch.FloatTensor(Y_scaled).to(self.device)
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.6f}")
    
    def predict(self, X_seq):
        self.model.eval()
        X_flat = X_seq.reshape(-1, X_seq.shape[-1])
        X_scaled = self.scaler.transform(X_flat).reshape(X_seq.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        pred_np = predictions.cpu().numpy()
        # Denormalize
        pred_flat = pred_np.reshape(-1, pred_np.shape[-1])
        pred_denorm = self.scaler.inverse_transform(pred_flat).reshape(pred_np.shape)
        
        return pred_denorm[:, -1, :]  # Return last step prediction
    
    def rollout(self, initial_state, n_steps):
        """Multi-step prediction"""
        self.model.eval()
        predictions = [initial_state]
        
        # Start with a sequence (repeat initial state)
        current_seq = np.tile(initial_state, (20, 1))
        
        for _ in range(n_steps):
            current_seq_scaled = self.scaler.transform(current_seq)
            X_tensor = torch.FloatTensor(current_seq_scaled).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(X_tensor)
            
            next_state_scaled = pred[0, -1, :].cpu().numpy()
            next_state = self.scaler.inverse_transform(next_state_scaled.reshape(1, -1))[0]
            
            predictions.append(next_state)
            # Update sequence
            current_seq = np.vstack([current_seq[1:], next_state])
        
        return np.array(predictions)

# --- 4. EXPERIMENT EXECUTION ---
def run_experiment_A2():
    print("🎯 STARTING EXPERIMENT A2: THE DEFINITIVE TEST")
    print("="*70)
    print("Testing LSTM (proper architecture) vs Polynomial in twisted coordinates")
    print("="*70)
    
    # 1. Generate Data
    print("\n[1] Generating Sequential Data (Double Pendulum)...")
    sim = DoublePendulumSimulator()
    X_seq_std, Y_seq_std = sim.generate_sequences(n_trajectories=100, seq_length=20)
    print(f"    Sequence Shape: {X_seq_std.shape}")
    
    # 2. Apply Twist
    print("\n[2] Applying Twisted Coordinate Transformation...")
    twist = TwistedCoordinateSystem()
    X_seq_twist = twist.forward(X_seq_std)
    Y_seq_twist = twist.forward(Y_seq_std)
    
    # Split Data
    test_size = 0.2
    X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(X_seq_std, Y_seq_std, test_size=test_size, random_state=42)
    X_train_twi, X_test_twi, Y_train_twi, Y_test_twi = train_test_split(X_seq_twist, Y_seq_twist, test_size=test_size, random_state=42)
    
    results = {}
    
    # 3. Train & Evaluate
    print("\n[3] Training Models...")
    
    # --- STANDARD FRAME ---
    print("\n    === STANDARD FRAME ===")
    
    # Polynomial
    print("    [Polynomial Model]")
    poly_std = PolynomialModel(degree=3)
    t0 = time.time()
    poly_std.fit(X_train_std, Y_train_std)
    print(f"      Training time: {time.time()-t0:.2f}s")
    pred_poly_std = poly_std.predict(X_test_std)
    r2_poly_std = r2_score(Y_test_std[:, -1, :], pred_poly_std)
    print(f"      1-step R²: {r2_poly_std:.4f}")
    
    # LSTM
    print("    [LSTM Model]")
    lstm_std = LSTMWrapper(hidden_size=128, num_layers=2, epochs=30, lr=0.001)
    t0 = time.time()
    lstm_std.fit(X_train_std, Y_train_std)
    print(f"      Training time: {time.time()-t0:.2f}s")
    pred_lstm_std = lstm_std.predict(X_test_std)
    r2_lstm_std = r2_score(Y_test_std[:, -1, :], pred_lstm_std)
    print(f"      1-step R²: {r2_lstm_std:.4f}")
    
    # --- TWISTED FRAME ---
    print("\n    === TWISTED FRAME ===")
    
    # Polynomial
    print("    [Polynomial Model]")
    poly_twi = PolynomialModel(degree=3)
    poly_twi.fit(X_train_twi, Y_train_twi)
    pred_poly_twi = poly_twi.predict(X_test_twi)
    r2_poly_twi = r2_score(Y_test_twi[:, -1, :], pred_poly_twi)
    print(f"      1-step R²: {r2_poly_twi:.4f}")
    
    # LSTM
    print("    [LSTM Model]")
    lstm_twi = LSTMWrapper(hidden_size=128, num_layers=2, epochs=30, lr=0.001)
    lstm_twi.fit(X_train_twi, Y_train_twi)
    pred_lstm_twi = lstm_twi.predict(X_test_twi)
    r2_lstm_twi = r2_score(Y_test_twi[:, -1, :], pred_lstm_twi)
    print(f"      1-step R²: {r2_lstm_twi:.4f}")
    
    # 4. Analysis
    print("\n[4] ANALYSIS & VERDICT")
    print("="*70)
    
    gap_poly = r2_poly_std - r2_poly_twi
    gap_lstm = r2_lstm_std - r2_lstm_twi
    
    print(f"Polynomial Gap (Std - Twist): {gap_poly:+.4f}")
    print(f"LSTM Gap       (Std - Twist): {gap_lstm:+.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Polynomial Standard
    axes[0, 0].scatter(Y_test_std[:, -1, 0], pred_poly_std[:, 0], alpha=0.5, s=10)
    axes[0, 0].plot([Y_test_std[:, -1, 0].min(), Y_test_std[:, -1, 0].max()], 
                     [Y_test_std[:, -1, 0].min(), Y_test_std[:, -1, 0].max()], 'r--', lw=2)
    axes[0, 0].set_title(f"Polynomial - Standard\nR² = {r2_poly_std:.4f}")
    axes[0, 0].set_xlabel("True θ₁")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Polynomial Twisted
    axes[0, 1].scatter(Y_test_twi[:, -1, 0], pred_poly_twi[:, 0], alpha=0.5, s=10, color='orange')
    axes[0, 1].plot([Y_test_twi[:, -1, 0].min(), Y_test_twi[:, -1, 0].max()], 
                     [Y_test_twi[:, -1, 0].min(), Y_test_twi[:, -1, 0].max()], 'r--', lw=2)
    axes[0, 1].set_title(f"Polynomial - Twisted\nR² = {r2_poly_twi:.4f}")
    axes[0, 1].set_xlabel("True u₁")
    axes[0, 1].set_ylabel("Predicted")
    axes[0, 1].grid(True, alpha=0.3)
    
    # LSTM Standard
    axes[1, 0].scatter(Y_test_std[:, -1, 0], pred_lstm_std[:, 0], alpha=0.5, s=10, color='green')
    axes[1, 0].plot([Y_test_std[:, -1, 0].min(), Y_test_std[:, -1, 0].max()], 
                     [Y_test_std[:, -1, 0].min(), Y_test_std[:, -1, 0].max()], 'r--', lw=2)
    axes[1, 0].set_title(f"LSTM - Standard\nR² = {r2_lstm_std:.4f}")
    axes[1, 0].set_xlabel("True θ₁")
    axes[1, 0].set_ylabel("Predicted")
    axes[1, 0].grid(True, alpha=0.3)
    
    # LSTM Twisted
    axes[1, 1].scatter(Y_test_twi[:, -1, 0], pred_lstm_twi[:, 0], alpha=0.5, s=10, color='purple')
    axes[1, 1].plot([Y_test_twi[:, -1, 0].min(), Y_test_twi[:, -1, 0].max()], 
                     [Y_test_twi[:, -1, 0].min(), Y_test_twi[:, -1, 0].max()], 'r--', lw=2)
    axes[1, 1].set_title(f"LSTM - Twisted\nR² = {r2_lstm_twi:.4f}")
    axes[1, 1].set_xlabel("True u₁")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_A2_results.png', dpi=150)
    print("\n📊 Graph saved as 'experiment_A2_results.png'")
    
    # VERDICT
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if abs(gap_lstm) < 0.1 and abs(gap_poly) > 0.2:
        print("✅ HYPOTHESIS CONFIRMED: LSTM is Coordinate Independent!")
        print("   LSTM maintains performance in twisted frame.")
        print("   Polynomial fails in twisted frame.")
        print("\n   🔓 THE CAGE IS BROKEN (for LSTM)")
    elif abs(gap_lstm) > 0.2:
        print("❌ HYPOTHESIS REFUTED: LSTM is also Coordinate Dependent")
        print("   Both models fail in twisted coordinates.")
        print("\n   🔒 THE CAGE REMAINS LOCKED")
    else:
        print("🟡 INCONCLUSIVE: Mixed or unclear results")
        print(f"   LSTM gap: {gap_lstm:.4f}")
        print(f"   Poly gap: {gap_poly:.4f}")
    
    return {
        'poly_std': r2_poly_std,
        'poly_twi': r2_poly_twi,
        'lstm_std': r2_lstm_std,
        'lstm_twi': r2_lstm_twi,
        'gap_poly': gap_poly,
        'gap_lstm': gap_lstm
    }

if __name__ == "__main__":
    results = run_experiment_A2()
