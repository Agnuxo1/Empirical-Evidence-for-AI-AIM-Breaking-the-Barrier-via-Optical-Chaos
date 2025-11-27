"""
Physics vs. Darwin: Experiment 5
Conservation Laws Discovery (The Hidden Symmetry)
-------------------------------------------------

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
Determine if a chaotic optical system can discover physical conservation laws
(energy and momentum conservation) without explicit knowledge of these concepts,
and evaluate whether it does so by reconstructing human variables or finding
distributed representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- 1. COLLISION SIMULATOR ---
class CollisionSimulator:
    """
    Simulates elastic and inelastic collisions in 1D and 2D.
    For simplicity, we'll focus on 1D collisions first (can extend to 2D).
    """
    def __init__(self):
        pass
    
    def elastic_collision_1d(self, m1, m2, v1, v2):
        """
        Elastic collision in 1D.
        Conserves both momentum and kinetic energy.
        
        Returns: v1_final, v2_final
        """
        # Conservation of momentum: m1*v1 + m2*v2 = m1*v1' + m2*v2'
        # Conservation of energy: 0.5*m1*v1^2 + 0.5*m2*v2^2 = 0.5*m1*v1'^2 + 0.5*m2*v2'^2
        
        # Solution for elastic collision:
        v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
        
        return v1_final, v2_final
    
    def inelastic_collision_1d(self, m1, m2, v1, v2, e):
        """
        Inelastic collision in 1D with coefficient of restitution e.
        - e = 1: Perfectly elastic (energy conserved)
        - e = 0: Perfectly inelastic (objects stick together)
        - 0 < e < 1: Partially inelastic
        
        Momentum is always conserved.
        Energy is conserved only if e = 1.
        """
        # Relative velocity after collision: v2' - v1' = -e * (v2 - v1)
        # Conservation of momentum: m1*v1 + m2*v2 = m1*v1' + m2*v2'
        
        # Solution:
        v1_final = (m1 * v1 + m2 * v2 - m2 * e * (v2 - v1)) / (m1 + m2)
        v2_final = (m1 * v1 + m2 * v2 + m1 * e * (v2 - v1)) / (m1 + m2)
        
        return v1_final, v2_final
    
    def generate_elastic_dataset(self, n_samples=3000):
        """Generate dataset of elastic collisions (e = 1.0)"""
        np.random.seed(42)
        
        m1 = np.random.uniform(0.1, 10.0, n_samples)
        m2 = np.random.uniform(0.1, 10.0, n_samples)
        v1 = np.random.uniform(-50, 50, n_samples)
        v2 = np.random.uniform(-50, 50, n_samples)
        e = np.ones(n_samples)  # All elastic
        
        v1_final, v2_final = self.elastic_collision_1d(m1, m2, v1, v2)
        
        # Input: [m1, m2, v1, v2, e]
        X = np.column_stack((m1, m2, v1, v2, e))
        # Output: [v1_final, v2_final]
        y = np.column_stack((v1_final, v2_final))
        
        return X, y
    
    def generate_inelastic_dataset(self, n_samples=2000):
        """Generate dataset of inelastic collisions (e in [0, 0.9])"""
        np.random.seed(123)
        
        m1 = np.random.uniform(0.1, 10.0, n_samples)
        m2 = np.random.uniform(0.1, 10.0, n_samples)
        v1 = np.random.uniform(-50, 50, n_samples)
        v2 = np.random.uniform(-50, 50, n_samples)
        e = np.random.uniform(0.0, 0.9, n_samples)  # Inelastic
        
        v1_final, v2_final = self.inelastic_collision_1d(m1, m2, v1, v2, e)
        
        X = np.column_stack((m1, m2, v1, v2, e))
        y = np.column_stack((v1_final, v2_final))
        
        return X, y
    
    def generate_mixed_dataset(self, n_samples=1000):
        """Generate mixed dataset for transfer testing"""
        np.random.seed(456)
        
        m1 = np.random.uniform(0.1, 10.0, n_samples)
        m2 = np.random.uniform(0.1, 10.0, n_samples)
        v1 = np.random.uniform(-50, 50, n_samples)
        v2 = np.random.uniform(-50, 50, n_samples)
        e = np.random.uniform(0.0, 1.0, n_samples)  # Full range
        
        # Use appropriate collision function based on e
        v1_final = np.zeros(n_samples)
        v2_final = np.zeros(n_samples)
        
        for i in range(n_samples):
            if abs(e[i] - 1.0) < 0.01:  # Elastic
                v1_final[i], v2_final[i] = self.elastic_collision_1d(m1[i], m2[i], v1[i], v2[i])
            else:  # Inelastic
                v1_final[i], v2_final[i] = self.inelastic_collision_1d(m1[i], m2[i], v1[i], v2[i], e[i])
        
        X = np.column_stack((m1, m2, v1, v2, e))
        y = np.column_stack((v1_final, v2_final))
        
        return X, y
    
    def verify_conservation(self, m1, m2, v1, v2, v1_final, v2_final, e):
        """Verify conservation laws"""
        # Momentum conservation (always true)
        p_initial = m1 * v1 + m2 * v2
        p_final = m1 * v1_final + m2 * v2_final
        momentum_error = np.abs(p_final - p_initial)
        
        # Energy conservation (only for elastic)
        E_initial = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
        E_final = 0.5 * m1 * v1_final**2 + 0.5 * m2 * v2_final**2
        energy_error = np.abs(E_final - E_initial)
        
        return momentum_error, energy_error

# --- 2. DARWINIAN BASELINE MODEL ---
class DarwinianModel:
    """Baseline model using polynomial features"""
    def __init__(self, degree=4):
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

# --- 3. CONSERVATION CHAOS MODEL ---
class ConservationChaosModel:
    """Optical chaos model for discovering conservation laws"""
    def __init__(self, n_features=4096, brightness=0.001):
        self.n_features = n_features
        self.brightness = brightness
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=0.1)
        self.reservoir = None
        
    def _chaos_transform(self, X):
        """Chaotic optical transformation"""
        n_samples = X.shape[0]
        
        if self.reservoir is None:
            np.random.seed(999)
            self.reservoir = np.random.randn(X.shape[1], self.n_features)
            
        # Optical mixing
        optical_field = X @ self.reservoir
        optical_field *= self.brightness
        
        # FFT interference
        spectrum = np.fft.rfft(optical_field, axis=1)
        intensity = np.abs(spectrum) ** 2
        
        # Normalize
        intensity = np.tanh(intensity)
        
        return intensity
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        features = self._chaos_transform(X_scaled)
        self.readout.fit(features, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        features = self._chaos_transform(X_scaled)
        return self.readout.predict(features)
    
    def get_internal_state(self, X):
        """Get internal features for cage analysis"""
        X_scaled = self.scaler.transform(X)
        return self._chaos_transform(X_scaled)

# --- 4. MAIN EXPERIMENT ---
def run_experiment_5():
    print("⚖️ STARTING EXPERIMENT 5: CONSERVATION LAWS DISCOVERY")
    print("=" * 60)
    print("Testing if chaos can discover energy/momentum conservation")
    print("without reconstructing human variables")
    print("=" * 60)
    
    # 1. Generate Data
    print("\n[Generating Datasets]")
    sim = CollisionSimulator()
    
    print("  - Elastic collisions (e=1.0)...")
    X_elastic, y_elastic = sim.generate_elastic_dataset(n_samples=3000)
    
    print("  - Inelastic collisions (e in [0, 0.9])...")
    X_inelastic, y_inelastic = sim.generate_inelastic_dataset(n_samples=2000)
    
    print("  - Mixed dataset (e in [0, 1])...")
    X_mixed, y_mixed = sim.generate_mixed_dataset(n_samples=1000)
    
    # Verify conservation laws
    print("\n[Verifying Conservation Laws]")
    m1_el, m2_el, v1_el, v2_el, e_el = X_elastic.T
    v1_f_el, v2_f_el = y_elastic.T
    mom_err, en_err = sim.verify_conservation(m1_el, m2_el, v1_el, v2_el, v1_f_el, v2_f_el, e_el)
    print(f"  Elastic - Mean momentum error: {np.mean(mom_err):.2e}")
    print(f"  Elastic - Mean energy error: {np.mean(en_err):.2e}")
    
    m1_in, m2_in, v1_in, v2_in, e_in = X_inelastic.T
    v1_f_in, v2_f_in = y_inelastic.T
    mom_err_in, en_err_in = sim.verify_conservation(m1_in, m2_in, v1_in, v2_in, v1_f_in, v2_f_in, e_in)
    print(f"  Inelastic - Mean momentum error: {np.mean(mom_err_in):.2e}")
    print(f"  Inelastic - Mean energy error: {np.mean(en_err_in):.2e} (should be > 0)")
    
    # 2. Train on Elastic Collisions
    print("\n[Training] Both models learn from Elastic Collisions ONLY...")
    
    # Split elastic data
    X_train, X_test, y_train, y_test = train_test_split(
        X_elastic, y_elastic, test_size=0.2, random_state=42
    )
    
    # Darwinian Model
    darwin_model = DarwinianModel(degree=4)
    darwin_model.fit(X_train, y_train)
    y_pred_darwin = darwin_model.predict(X_test)
    
    # Chaos Model
    chaos_model = ConservationChaosModel(n_features=4096, brightness=0.001)
    chaos_model.fit(X_train, y_train)
    y_pred_chaos = chaos_model.predict(X_test)
    
    # 3. Evaluate on Elastic (Within-Domain)
    print("\n[Within-Domain Test: Elastic Collisions]")
    r2_darwin_el = r2_score(y_test, y_pred_darwin)
    r2_chaos_el = r2_score(y_test, y_pred_chaos)
    
    print(f"  Darwinian R²: {r2_darwin_el:.4f}")
    print(f"  Chaos Model R²: {r2_chaos_el:.4f}")
    
    # 4. Transfer Test: Elastic → Inelastic
    print("\n[Transfer Test: Elastic → Inelastic]")
    y_pred_darwin_transfer = darwin_model.predict(X_inelastic)
    y_pred_chaos_transfer = chaos_model.predict(X_inelastic)
    
    r2_darwin_transfer = r2_score(y_inelastic, y_pred_darwin_transfer)
    r2_chaos_transfer = r2_score(y_inelastic, y_pred_chaos_transfer)
    
    print(f"  Darwinian Transfer R²: {r2_darwin_transfer:.4f}")
    print(f"  Chaos Transfer R²: {r2_chaos_transfer:.4f}")
    
    # 5. Conservation Error Analysis
    print("\n[Conservation Error Analysis]")
    
    # For chaos model predictions on elastic
    m1_t, m2_t, v1_t, v2_t, e_t = X_test.T
    v1_pred_c, v2_pred_c = y_pred_chaos.T
    mom_err_pred, en_err_pred = sim.verify_conservation(m1_t, m2_t, v1_t, v2_t, v1_pred_c, v2_pred_c, e_t)
    
    print(f"  Chaos Model (on elastic test):")
    print(f"    Mean momentum error: {np.mean(mom_err_pred):.4f}")
    print(f"    Mean energy error: {np.mean(en_err_pred):.4f}")
    
    # 6. Cage Analysis
    print("\n[Cage Analysis: Internal Feature Correlations]")
    print("  Checking if internal features correlate with energy/momentum...")
    
    # Get internal states
    internal_states = chaos_model.get_internal_state(X_test)
    
    # Calculate energy and momentum for test set
    m1_t, m2_t, v1_t, v2_t, e_t = X_test.T
    energy_total = 0.5 * m1_t * v1_t**2 + 0.5 * m2_t * v2_t**2
    momentum_total = m1_t * v1_t + m2_t * v2_t
    
    # Correlations
    corrs_energy = []
    corrs_momentum = []
    
    for i in range(min(1000, internal_states.shape[1])):  # Sample features
        corr_e = np.abs(np.corrcoef(internal_states[:, i], energy_total)[0, 1])
        corr_p = np.abs(np.corrcoef(internal_states[:, i], momentum_total)[0, 1])
        corrs_energy.append(corr_e)
        corrs_momentum.append(corr_p)
    
    max_corr_energy = np.max(corrs_energy)
    max_corr_momentum = np.max(corrs_momentum)
    mean_corr_energy = np.mean(corrs_energy)
    mean_corr_momentum = np.mean(corrs_momentum)
    
    print(f"  Max correlation with Energy: {max_corr_energy:.4f}")
    print(f"  Max correlation with Momentum: {max_corr_momentum:.4f}")
    print(f"  Mean correlation with Energy: {mean_corr_energy:.4f}")
    print(f"  Mean correlation with Momentum: {mean_corr_momentum:.4f}")
    
    # Cage verdict
    if max_corr_energy > 0.8 or max_corr_momentum > 0.8:
        print("\n🔒 CAGE STATUS: LOCKED")
        print("   The model reconstructed human variables (energy/momentum)")
    elif max_corr_energy < 0.5 and max_corr_momentum < 0.5 and r2_chaos_el > 0.9:
        print("\n🔓 CAGE STATUS: BROKEN")
        print("   The model discovered conservation without reconstructing variables")
    else:
        print("\n🟡 CAGE STATUS: UNCLEAR")
        print("   Intermediate correlation levels")
    
    # 7. Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Within-Domain Predictions (Chaos)
    plt.subplot(2, 3, 1)
    plt.scatter(y_test[:, 0], y_pred_chaos[:, 0], alpha=0.3, s=5, label='v1_final')
    plt.scatter(y_test[:, 1], y_pred_chaos[:, 1], alpha=0.3, s=5, label='v2_final')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
    plt.title(f"Within-Domain: Chaos Model\nR² = {r2_chaos_el:.3f}")
    plt.xlabel("True Velocity")
    plt.ylabel("Predicted Velocity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Transfer Test (Chaos)
    plt.subplot(2, 3, 2)
    plt.scatter(y_inelastic[:, 0], y_pred_chaos_transfer[:, 0], alpha=0.3, s=5, label='v1_final')
    plt.scatter(y_inelastic[:, 1], y_pred_chaos_transfer[:, 1], alpha=0.3, s=5, label='v2_final')
    plt.plot([y_inelastic.min(), y_inelastic.max()], [y_inelastic.min(), y_inelastic.max()], 'k--', lw=1)
    plt.title(f"Transfer: Chaos Model\nR² = {r2_chaos_transfer:.3f}")
    plt.xlabel("True Velocity")
    plt.ylabel("Predicted Velocity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Conservation Errors
    plt.subplot(2, 3, 3)
    plt.hist(mom_err_pred, bins=50, alpha=0.7, label='Momentum Error', color='blue')
    plt.hist(en_err_pred, bins=50, alpha=0.7, label='Energy Error', color='red')
    plt.xlabel("Conservation Error")
    plt.ylabel("Frequency")
    plt.title("Conservation Law Violations\n(Chaos Model Predictions)")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cage Analysis - Energy Correlations
    plt.subplot(2, 3, 4)
    plt.hist(corrs_energy, bins=50, alpha=0.7, color='green')
    plt.axvline(max_corr_energy, color='red', linestyle='--', label=f'Max: {max_corr_energy:.3f}')
    plt.xlabel("Correlation with Energy")
    plt.ylabel("Count")
    plt.title("Internal Feature Correlations\nwith Total Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Cage Analysis - Momentum Correlations
    plt.subplot(2, 3, 5)
    plt.hist(corrs_momentum, bins=50, alpha=0.7, color='purple')
    plt.axvline(max_corr_momentum, color='red', linestyle='--', label=f'Max: {max_corr_momentum:.3f}')
    plt.xlabel("Correlation with Momentum")
    plt.ylabel("Count")
    plt.title("Internal Feature Correlations\nwith Total Momentum")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Transfer Comparison
    plt.subplot(2, 3, 6)
    models = ['Darwinian', 'Chaos']
    r2_scores = [r2_darwin_transfer, r2_chaos_transfer]
    colors = ['blue', 'red']
    plt.bar(models, r2_scores, color=colors, alpha=0.7)
    plt.ylabel("R² Score")
    plt.title("Transfer Performance\n(Elastic → Inelastic)")
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiment_5_conservation.png', dpi=150)
    print("\n📊 Graph saved as 'experiment_5_conservation.png'")
    plt.show()
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Within-Domain (Elastic):")
    print(f"  Darwinian R²: {r2_darwin_el:.4f}")
    print(f"  Chaos R²: {r2_chaos_el:.4f}")
    print(f"\nTransfer (Elastic → Inelastic):")
    print(f"  Darwinian R²: {r2_darwin_transfer:.4f}")
    print(f"  Chaos R²: {r2_chaos_transfer:.4f}")
    print(f"\nCage Analysis:")
    print(f"  Max Energy Correlation: {max_corr_energy:.4f}")
    print(f"  Max Momentum Correlation: {max_corr_momentum:.4f}")

if __name__ == "__main__":
    run_experiment_5()

