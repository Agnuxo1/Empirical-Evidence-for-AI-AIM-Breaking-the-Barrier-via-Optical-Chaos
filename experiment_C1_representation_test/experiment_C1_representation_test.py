"""
Physics vs. Darwin: Experiment C1
The Representation Test - Direct Falsification of Darwin's Cage Theory
-----------------------------------------------------------------------

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
Direct falsification test of Darwin's Cage theory by comparing two representations
of the SAME physical problem with IDENTICAL model architecture and hyperparameters.

Core Hypothesis (Falsifiable):
If Darwin's Cage theory is correct: Non-anthropomorphic representation should show
LOWER correlation with human variables compared to anthropomorphic representation.

If theory is false: Both representations should show similar correlation patterns.

This is a controlled experiment where ONLY the input representation varies.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- 1. PHYSICS SIMULATOR (Same as Experiment 1) ---
class PhysicsSimulator:
    """Generates projectile motion data - same as Experiment 1"""
    def __init__(self, g=9.81):
        self.g = g
        
    def calculate_trajectory(self, v0, angle_deg):
        """Returns the landing distance (R) based on Newtonian Physics."""
        theta = np.radians(angle_deg)
        # Formula: R = (v^2 * sin(2*theta)) / g
        distance = (v0**2 * np.sin(2*theta)) / self.g
        return distance

    def generate_dataset(self, n_samples=2000, random_seed=42):
        """Generates random throws."""
        np.random.seed(random_seed)
        # Random velocities between 10 and 100 m/s
        v0 = np.random.uniform(10, 100, n_samples)
        # Random angles between 5 and 85 degrees
        angle = np.random.uniform(5, 85, n_samples)
        
        # Calculate Truth
        y = self.calculate_trajectory(v0, angle)
        
        return v0, angle, y

# --- 2. DATA CONVERTER (Between Representations) ---
class DataConverter:
    """
    Converts between anthropomorphic and non-anthropomorphic representations.
    Ensures information equivalence.
    """
    @staticmethod
    def to_anthropomorphic(v0, angle):
        """
        Anthropomorphic representation: [v0, angle]
        This is how humans naturally think about the problem.
        """
        return np.column_stack((v0, angle))
    
    @staticmethod
    def to_non_anthropomorphic(v0, angle, x0=0.0, y0=0.0):
        """
        Non-anthropomorphic representation: [x0, y0, vx, vy]
        Raw coordinates without human concepts like 'velocity' or 'angle'.
        Contains the SAME information but in different form.
        """
        theta = np.radians(angle)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        return np.column_stack((np.full_like(v0, x0), np.full_like(v0, y0), vx, vy))
    
    @staticmethod
    def verify_information_equivalence(X_anthro, X_non_anthro, v0, angle):
        """
        Verify that both representations contain the same information.
        We can recover v0 and angle from non-anthropomorphic representation.
        """
        # Recover from non-anthropomorphic
        vx = X_non_anthro[:, 2]
        vy = X_non_anthro[:, 3]
        recovered_v0 = np.sqrt(vx**2 + vy**2)
        recovered_angle = np.degrees(np.arctan2(vy, vx))
        
        # Check equivalence
        v0_error = np.abs(recovered_v0 - v0).max()
        angle_error = np.abs(recovered_angle - angle).max()
        
        return v0_error < 1e-10 and angle_error < 1e-10

# --- 3. OPTICAL CHAOS MACHINE (Same Architecture) ---
class OpticalChaosMachine:
    """
    Identical to Experiment 1's OpticalChaosMachine.
    Same architecture, same hyperparameters.
    """
    def __init__(self, n_features=4096, brightness=0.001, random_seed=1337):
        """
        n_features: Number of optical paths (simulated neurons/pixels).
        brightness: Scaling factor for the signal.
        random_seed: For reproducibility of the optical matrix.
        """
        self.n_features = n_features
        self.brightness = brightness
        self.random_seed = random_seed
        self.readout = Ridge(alpha=0.1)  # Linear readout (cheap training)
        self.optical_matrix = None  # This will be our fixed "Diffuser"
        
    def _optical_interference(self, X):
        """
        Simulates light passing through a chaotic medium (Random Matrix)
        and interfering (FFT).
        """
        n_samples, n_input = X.shape
        
        # 1. Initialize the chaotic medium (The Diffuser) if not exists
        if self.optical_matrix is None:
            np.random.seed(self.random_seed)  # Fixed seed for reproducibility
            # Random complex weights to simulate phase shifts
            self.optical_matrix = np.random.normal(0, 1, (n_input, self.n_features))
            
        # 2. Projection (Light enters the medium)
        light_field = X @ self.optical_matrix
        
        # 3. Wave Propagation (FFT)
        interference_pattern = np.fft.rfft(light_field, axis=1)
        
        # 4. Detection (Intensity)
        intensity = np.abs(interference_pattern)**2
        
        # Normalize (simulating sensor saturation)
        intensity = np.tanh(intensity * self.brightness)
        
        return intensity

    def fit(self, X, y):
        # Transform inputs into "Optical Speckle Patterns"
        X_optical = self._optical_interference(X)
        # Train ONLY the readout (The brain interpreting the pattern)
        self.readout.fit(X_optical, y)
        
    def predict(self, X):
        X_optical = self._optical_interference(X)
        return self.readout.predict(X_optical)

    def get_internal_state(self, X):
        """Helper to get internal features for cage analysis"""
        return self._optical_interference(X)

# --- 4. CAGE ANALYZER (Statistical Comparison) ---
class CageAnalyzer:
    """
    Analyzes cage status with statistical rigor.
    Compares correlation distributions between representations.
    """
    def __init__(self):
        pass
    
    def analyze_cage(self, model, X_test, v0_test, angle_test, representation_type='anthropomorphic'):
        """
        Analyze cage status by checking correlation of ALL internal features
        with human variables.
        
        NOTE: v0 and angle are derivable from both representations:
        - Anthropomorphic: directly in inputs [v0, angle]
        - Non-anthropomorphic: derivable from [x0, y0, vx, vy] as v0=sqrt(vx²+vy²), angle=arctan2(vy,vx)
        
        This is intentional - we want to see if models reconstruct these human concepts.
        
        Returns:
        - max_correlations: Dict with max correlation for each variable
        - all_correlations: Dict with all correlations for statistical testing
        """
        # Get internal states
        internal_states = model.get_internal_state(X_test)
        n_features = internal_states.shape[1]
        
        # Human variables to check
        # These are the "human concepts" we want to see if models reconstruct
        human_vars = {
            'v0': v0_test,
            'angle': angle_test,
            'v0_squared': v0_test**2,
            'sin_2theta': np.sin(2 * np.radians(angle_test))
        }
        
        # For non-anthropomorphic, also check direct correlations with input components
        # This helps understand if the model uses vx, vy directly
        if representation_type == 'non_anthropomorphic':
            # Extract vx, vy from test data (we need to reconstruct from test indices)
            # Actually, we can't easily get vx, vy here without passing more info
            # So we'll focus on the human variables which are the key test
            pass
        
        max_correlations = {}
        all_correlations = {}
        
        for var_name, var_values in human_vars.items():
            correlations = []
            
            # Check correlation with ALL internal features
            for i in range(n_features):
                # Check for constant features (would cause division by zero)
                if np.std(internal_states[:, i]) < 1e-10:
                    continue
                if np.std(var_values) < 1e-10:
                    continue
                    
                corr = np.corrcoef(internal_states[:, i], var_values)[0, 1]
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations.append(np.abs(corr))
            
            if correlations:
                max_correlations[var_name] = np.max(correlations)
                all_correlations[var_name] = np.array(correlations)
            else:
                max_correlations[var_name] = 0.0
                all_correlations[var_name] = np.array([])
        
        return max_correlations, all_correlations
    
    def compare_representations(self, corr_anthro, corr_non_anthro):
        """
        Statistically compare correlation distributions between representations.
        
        Returns:
        - comparison_results: Dict with statistical test results
        """
        comparison_results = {}
        
        # Compare each human variable
        for var_name in corr_anthro.keys():
            if var_name not in corr_non_anthro:
                continue
                
            anthro_corrs = corr_anthro[var_name]
            non_anthro_corrs = corr_non_anthro[var_name]
            
            if len(anthro_corrs) == 0 or len(non_anthro_corrs) == 0:
                continue
            
            # Statistical tests
            # 1. T-test for difference in means
            t_stat, p_value = stats.ttest_ind(anthro_corrs, non_anthro_corrs)
            
            # 2. Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(anthro_corrs, non_anthro_corrs, 
                                                   alternative='two-sided')
            
            # 3. Effect size (Cohen's d)
            mean_diff = np.mean(anthro_corrs) - np.mean(non_anthro_corrs)
            pooled_std = np.sqrt((np.var(anthro_corrs) + np.var(non_anthro_corrs)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            comparison_results[var_name] = {
                'mean_anthro': np.mean(anthro_corrs),
                'mean_non_anthro': np.mean(non_anthro_corrs),
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value_t': p_value,
                'u_statistic': u_stat,
                'p_value_u': u_p_value,
                'cohens_d': cohens_d,
                'max_anthro': np.max(anthro_corrs),
                'max_non_anthro': np.max(non_anthro_corrs),
                'max_diff': np.max(anthro_corrs) - np.max(non_anthro_corrs)
            }
        
        return comparison_results

# --- 5. MAIN EXPERIMENT ---
def run_experiment_C1():
    print("=" * 80)
    print("EXPERIMENT C1: THE REPRESENTATION TEST")
    print("Direct Falsification of Darwin's Cage Theory")
    print("=" * 80)
    print("\nThis experiment tests if input representation alone affects cage status.")
    print("Same physics, same model, same hyperparameters - ONLY representation differs.")
    print("\nKNOWN LIMITATIONS:")
    print("  1. Dimensionality differs: 2D (anthro) vs 4D (non-anthro)")
    print("  2. v0/angle are derivable from both representations (intentional)")
    print("  3. Different random seeds used to avoid dimensional bias\n")
    
    # Fixed random seeds for reproducibility
    DATA_SEED = 42
    MODEL_SEED = 1337
    SPLIT_SEED = 42
    
    # --- 1. GENERATE DATA ---
    print("[Step 1] Generating projectile motion data...")
    sim = PhysicsSimulator()
    v0, angle, y = sim.generate_dataset(n_samples=2000, random_seed=DATA_SEED)
    
    # Convert to both representations
    converter = DataConverter()
    X_anthro = converter.to_anthropomorphic(v0, angle)
    X_non_anthro = converter.to_non_anthropomorphic(v0, angle)
    
    # Verify information equivalence
    is_equivalent = converter.verify_information_equivalence(X_anthro, X_non_anthro, v0, angle)
    if not is_equivalent:
        raise ValueError("CRITICAL: Representations are not information-equivalent!")
    print("   ✓ Information equivalence verified")
    print(f"   Anthropomorphic shape: {X_anthro.shape}")
    print(f"   Non-anthropomorphic shape: {X_non_anthro.shape}")
    
    # --- 2. SCALE DATA ---
    print("\n[Step 2] Scaling data...")
    scaler_anthro = MinMaxScaler()
    scaler_non_anthro = MinMaxScaler()
    
    X_anthro_scaled = scaler_anthro.fit_transform(X_anthro)
    X_non_anthro_scaled = scaler_non_anthro.fit_transform(X_non_anthro)
    
    # --- 3. SPLIT DATA (Same split for both) ---
    print("\n[Step 3] Splitting data (same split for both representations)...")
    # Create indices for consistent splitting
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=SPLIT_SEED
    )
    
    # Split anthropomorphic representation
    X_anthro_train = X_anthro_scaled[train_indices]
    X_anthro_test = X_anthro_scaled[test_indices]
    X_anthro_test_raw = X_anthro[test_indices]  # Raw (unscaled) for cage analysis
    
    # Split non-anthropomorphic representation
    X_non_anthro_train = X_non_anthro_scaled[train_indices]
    X_non_anthro_test = X_non_anthro_scaled[test_indices]
    
    # Split targets and raw values
    y_train = y[train_indices]
    y_test = y[test_indices]
    v0_test = v0[test_indices]
    angle_test = angle[test_indices]
    
    print(f"   Training samples: {len(X_anthro_train)}")
    print(f"   Test samples: {len(X_anthro_test)}")
    
    # --- 4. TRAIN MODELS (Identical hyperparameters) ---
    print("\n[Step 4] Training models with IDENTICAL hyperparameters...")
    print("   Model architecture: OpticalChaosMachine")
    print("   n_features: 4096")
    print("   brightness: 0.001")
    print("   random_seed: 1337 (anthro), 1338 (non-anthro) - different to avoid dimensional bias")
    
    # Anthropomorphic model
    model_anthro = OpticalChaosMachine(
        n_features=4096, 
        brightness=0.001, 
        random_seed=MODEL_SEED
    )
    model_anthro.fit(X_anthro_train, y_train)
    y_pred_anthro = model_anthro.predict(X_anthro_test)
    r2_anthro = r2_score(y_test, y_pred_anthro)
    
    # Non-anthropomorphic model
    # CRITICAL FIX: Use different seed because input dimensions differ (2 vs 4)
    # Using same seed would create bias (first 2 rows would be identical)
    # We use MODEL_SEED + 1 to ensure independence while maintaining reproducibility
    model_non_anthro = OpticalChaosMachine(
        n_features=4096, 
        brightness=0.001, 
        random_seed=MODEL_SEED + 1  # Different seed to avoid dimensional bias
    )
    model_non_anthro.fit(X_non_anthro_train, y_train)
    y_pred_non_anthro = model_non_anthro.predict(X_non_anthro_test)
    r2_non_anthro = r2_score(y_test, y_pred_non_anthro)
    
    print(f"\n   Anthropomorphic R²: {r2_anthro:.6f}")
    print(f"   Non-anthropomorphic R²: {r2_non_anthro:.6f}")
    print(f"   R² difference: {abs(r2_anthro - r2_non_anthro):.6f}")
    
    if abs(r2_anthro - r2_non_anthro) > 0.01:
        print("   ⚠️ WARNING: Large R² difference - models may not be learning same physics")
    else:
        print("   ✓ Both models achieve similar accuracy (same physics learned)")
    
    # --- 5. CAGE ANALYSIS ---
    print("\n[Step 5] Cage Analysis: Comparing internal feature correlations...")
    analyzer = CageAnalyzer()
    
    max_corr_anthro, all_corr_anthro = analyzer.analyze_cage(
        model_anthro, X_anthro_test, v0_test, angle_test, representation_type='anthropomorphic'
    )
    max_corr_non_anthro, all_corr_non_anthro = analyzer.analyze_cage(
        model_non_anthro, X_non_anthro_test, v0_test, angle_test, representation_type='non_anthropomorphic'
    )
    
    print("\n   Max Correlations (Anthropomorphic):")
    for var, corr in max_corr_anthro.items():
        print(f"     {var:15s}: {corr:.6f}")
    
    print("\n   Max Correlations (Non-anthropomorphic):")
    for var, corr in max_corr_non_anthro.items():
        print(f"     {var:15s}: {corr:.6f}")
    
    # --- 6. STATISTICAL COMPARISON ---
    print("\n[Step 6] Statistical Comparison...")
    comparison = analyzer.compare_representations(all_corr_anthro, all_corr_non_anthro)
    
    print("\n   Statistical Test Results:")
    for var_name, results in comparison.items():
        print(f"\n   Variable: {var_name}")
        print(f"     Mean correlation (Anthro):    {results['mean_anthro']:.6f}")
        print(f"     Mean correlation (Non-anthro): {results['mean_non_anthro']:.6f}")
        print(f"     Mean difference:               {results['mean_diff']:.6f}")
        print(f"     Max correlation (Anthro):      {results['max_anthro']:.6f}")
        print(f"     Max correlation (Non-anthro):  {results['max_non_anthro']:.6f}")
        print(f"     Max difference:                {results['max_diff']:.6f}")
        print(f"     T-test p-value:                {results['p_value_t']:.6f}")
        print(f"     Mann-Whitney p-value:          {results['p_value_u']:.6f}")
        print(f"     Cohen's d (effect size):        {results['cohens_d']:.6f}")
        
        # Interpret effect size
        if abs(results['cohens_d']) < 0.2:
            effect = "negligible"
        elif abs(results['cohens_d']) < 0.5:
            effect = "small"
        elif abs(results['cohens_d']) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        print(f"     Effect size interpretation:     {effect}")
    
    # --- 7. VERDICT ---
    print("\n" + "=" * 80)
    print("VERDICT: FALSIFICATION TEST RESULTS")
    print("=" * 80)
    
    # Key metric: Max correlation with v0 (primary human variable)
    max_v0_anthro = max_corr_anthro['v0']
    max_v0_non_anthro = max_corr_non_anthro['v0']
    max_v0_diff = max_v0_anthro - max_v0_non_anthro
    
    # Statistical significance
    v0_p_value = comparison['v0']['p_value_t']
    
    print(f"\nPrimary Metric: Max correlation with velocity (v0)")
    print(f"  Anthropomorphic:    {max_v0_anthro:.6f}")
    print(f"  Non-anthropomorphic: {max_v0_non_anthro:.6f}")
    print(f"  Difference:         {max_v0_diff:.6f}")
    print(f"  Statistical test:   p = {v0_p_value:.6f}")
    
    # Falsification criteria
    # Note: We're looking at mean correlation difference, not max correlation difference
    mean_v0_diff = comparison['v0']['mean_diff'] if 'v0' in comparison else 0
    mean_v0_anthro = comparison['v0']['mean_anthro'] if 'v0' in comparison else 0
    mean_v0_non_anthro = comparison['v0']['mean_non_anthro'] if 'v0' in comparison else 0
    
    if v0_p_value < 0.05:
        # Significant difference detected
        if abs(mean_v0_diff) > 0.2:  # Meaningful difference in mean correlations
            if max_v0_anthro > 0.9 and max_v0_non_anthro < 0.5:
                print("\n✅ THEORY SUPPORTED:")
                print("   - Significant difference in correlation patterns")
                print("   - Anthropomorphic: Cage LOCKED (reconstructs human variables)")
                print("   - Non-anthropomorphic: Cage BROKEN (distributed representation)")
                print("   - Representation alone affects cage status")
            elif max_v0_anthro > 0.9:
                print("\n⚠️ PARTIAL SUPPORT:")
                print("   - Significant difference detected in mean correlations")
                print(f"   - Mean correlation difference: {mean_v0_diff:.4f}")
                print("   - Anthropomorphic shows high max correlation (cage locked)")
                print("   - Non-anthropomorphic shows lower mean correlation (more distributed)")
                print("   - But max correlation is still high - mixed result")
            else:
                print("\n⚠️ INCONCLUSIVE:")
                print("   - Significant difference detected")
                print("   - But pattern doesn't match expected (both show high max correlation)")
                print("   - May indicate representation doesn't affect cage status as predicted")
        else:
            print("\n⚠️ INCONCLUSIVE:")
            print("   - Significant difference detected but effect is small")
            print(f"   - Mean correlation difference: {mean_v0_diff:.4f}")
            print("   - May need more data or different analysis")
    else:
        print("\n❌ THEORY FALSIFIED:")
        print("   - No significant difference between representations")
        print("   - Representation does NOT affect cage status")
        print("   - Correlation patterns are similar")
    
    # --- 8. VISUALIZATION ---
    print("\n[Step 7] Generating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Prediction accuracy comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_pred_anthro, alpha=0.5, s=10, label='Anthropomorphic', color='blue')
    ax1.scatter(y_test, y_pred_non_anthro, alpha=0.5, s=10, label='Non-anthropomorphic', color='red')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax1.set_xlabel('True Distance (m)')
    ax1.set_ylabel('Predicted Distance (m)')
    ax1.set_title(f'Prediction Accuracy\nAnthro R²={r2_anthro:.4f}, Non-anthro R²={r2_non_anthro:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max correlations comparison
    ax2 = plt.subplot(2, 3, 2)
    vars_list = list(max_corr_anthro.keys())
    corr_anthro_list = [max_corr_anthro[v] for v in vars_list]
    corr_non_anthro_list = [max_corr_non_anthro[v] for v in vars_list]
    x = np.arange(len(vars_list))
    width = 0.35
    ax2.bar(x - width/2, corr_anthro_list, width, label='Anthropomorphic', color='blue', alpha=0.7)
    ax2.bar(x + width/2, corr_non_anthro_list, width, label='Non-anthropomorphic', color='red', alpha=0.7)
    ax2.set_xlabel('Human Variable')
    ax2.set_ylabel('Max Correlation')
    ax2.set_title('Max Correlations: Cage Analysis')
    ax2.set_xticks(x)
    ax2.set_xticklabels(vars_list, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Locked threshold')
    ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Broken threshold')
    
    # Plot 3: Correlation distribution for v0
    ax3 = plt.subplot(2, 3, 3)
    if len(all_corr_anthro['v0']) > 0 and len(all_corr_non_anthro['v0']) > 0:
        ax3.hist(all_corr_anthro['v0'], bins=50, alpha=0.6, label='Anthropomorphic', color='blue', density=True)
        ax3.hist(all_corr_non_anthro['v0'], bins=50, alpha=0.6, label='Non-anthropomorphic', color='red', density=True)
        ax3.axvline(max_v0_anthro, color='blue', linestyle='--', linewidth=2, label=f'Max Anthro: {max_v0_anthro:.3f}')
        ax3.axvline(max_v0_non_anthro, color='red', linestyle='--', linewidth=2, label=f'Max Non-anthro: {max_v0_non_anthro:.3f}')
        ax3.set_xlabel('Correlation with v0')
        ax3.set_ylabel('Density')
        ax3.set_title('Correlation Distribution: v0')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation distribution for angle
    ax4 = plt.subplot(2, 3, 4)
    if len(all_corr_anthro['angle']) > 0 and len(all_corr_non_anthro['angle']) > 0:
        ax4.hist(all_corr_anthro['angle'], bins=50, alpha=0.6, label='Anthropomorphic', color='blue', density=True)
        ax4.hist(all_corr_non_anthro['angle'], bins=50, alpha=0.6, label='Non-anthropomorphic', color='red', density=True)
        max_angle_anthro = max_corr_anthro['angle']
        max_angle_non_anthro = max_corr_non_anthro['angle']
        ax4.axvline(max_angle_anthro, color='blue', linestyle='--', linewidth=2, label=f'Max Anthro: {max_angle_anthro:.3f}')
        ax4.axvline(max_angle_non_anthro, color='red', linestyle='--', linewidth=2, label=f'Max Non-anthro: {max_angle_non_anthro:.3f}')
        ax4.set_xlabel('Correlation with angle')
        ax4.set_ylabel('Density')
        ax4.set_title('Correlation Distribution: angle')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Effect sizes
    ax5 = plt.subplot(2, 3, 5)
    effect_sizes = [comparison[v]['cohens_d'] for v in vars_list if v in comparison]
    vars_with_effects = [v for v in vars_list if v in comparison]
    colors = ['green' if abs(d) > 0.8 else 'orange' if abs(d) > 0.5 else 'blue' for d in effect_sizes]
    ax5.barh(vars_with_effects, effect_sizes, color=colors, alpha=0.7)
    ax5.axvline(0, color='black', linestyle='-', linewidth=1)
    ax5.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel("Cohen's d (Effect Size)")
    ax5.set_title('Effect Sizes: Representation Difference')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Plot 6: P-values
    ax6 = plt.subplot(2, 3, 6)
    p_values = [comparison[v]['p_value_t'] for v in vars_list if v in comparison]
    vars_with_p = [v for v in vars_list if v in comparison]
    colors_p = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'blue' for p in p_values]
    ax6.barh(vars_with_p, p_values, color=colors_p, alpha=0.7)
    ax6.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax6.set_xlabel('P-value (T-test)')
    ax6.set_title('Statistical Significance')
    ax6.set_xlim(0, 1)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    import os
    results_file = os.path.join(os.path.dirname(__file__), 'experiment_C1_results.png')
    plt.savefig(results_file, dpi=150, bbox_inches='tight')
    print("   ✓ Saved: experiment_C1_results.png")
    
    # --- 9. SAVE RESULTS ---
    results_summary = {
        'r2_anthro': r2_anthro,
        'r2_non_anthro': r2_non_anthro,
        'r2_difference': abs(r2_anthro - r2_non_anthro),
        'max_correlations_anthro': max_corr_anthro,
        'max_correlations_non_anthro': max_corr_non_anthro,
        'statistical_comparison': comparison,
        'verdict': {
            'max_v0_anthro': max_v0_anthro,
            'max_v0_non_anthro': max_v0_non_anthro,
            'max_v0_diff': max_v0_diff,
            'p_value': v0_p_value,
            'significant': v0_p_value < 0.05,
            'effect_size': comparison['v0']['cohens_d'] if 'v0' in comparison else None
        }
    }
    
    # Save to file
    import json
    import os
    results_file = os.path.join(os.path.dirname(__file__), 'results_summary.json')
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print("   ✓ Saved: results_summary.json")
    
    plt.show()
    
    return results_summary

if __name__ == "__main__":
    results = run_experiment_C1()
    print("\n" + "=" * 80)
    print("EXPERIMENT C1 COMPLETE")
    print("=" * 80)

