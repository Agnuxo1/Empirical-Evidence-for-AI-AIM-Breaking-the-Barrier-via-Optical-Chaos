import numpy as np
import matplotlib.pyplot as plt
from quantum_entanglement import BellPairSystem

def run_benchmark():
    print("=== Experiment B3 Benchmark: Verifying Quantum Correlations ===")
    
    sys = BellPairSystem()
    num_pairs_per_angle = 2000
    angles = np.linspace(0, 2*np.pi, 50)
    correlations = []
    theoretical_correlations = -np.cos(angles)
    
    print(f"1. Correlation Sweep: Testing {len(angles)} angles with {num_pairs_per_angle} pairs each...")
    
    # Sweep the relative angle theta between Detector A and Detector B
    # We fix Axis A at 0, and rotate Axis B from 0 to 2pi
    axis_A = 0.0
    
    for theta in angles:
        axis_B = theta
        res_A, res_B = sys.measure(axis_A, axis_B, num_pairs=num_pairs_per_angle)
        
        # Correlation E = <A * B>
        # Outcomes are +1 or -1.
        corr = np.mean(res_A * res_B)
        correlations.append(corr)
        
    # 2. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(angles, correlations, 'o', label='Simulated Data', alpha=0.6)
    plt.plot(angles, theoretical_correlations, 'r-', label='Quantum Theory (-cos(theta))', linewidth=2)
    plt.title('Bell Pair Correlation: Simulation vs Theory')
    plt.xlabel('Relative Angle (radians)')
    plt.ylabel('Correlation E(a,b)')
    plt.grid(True)
    plt.legend()
    
    output_img = "bell_correlation_plot.png"
    plt.savefig(output_img)
    print(f"   [Visual] Correlation plot saved to {output_img}")
    
    # 3. CHSH Inequality Test
    # S = |E(a, b) - E(a, b')| + |E(a', b) + E(a', b')|
    # Classical Limit: S <= 2
    # Quantum Limit: S <= 2*sqrt(2) ~ 2.828
    # Optimal Angles: a=0, a'=pi/2, b=pi/4, b'=3pi/4
    
    print("\n2. CHSH Inequality Test (The 'Cage' Check)")
    print("   Testing for violation of Local Realism (S > 2)...")
    
    a = 0
    a_prime = np.pi / 2
    b = np.pi / 4
    b_prime = 3 * np.pi / 4
    
    N = 10000
    
    def get_corr(ax1, ax2):
        rA, rB = sys.measure(ax1, ax2, N)
        return np.mean(rA * rB)
        
    E_ab = get_corr(a, b)
    E_ab_prime = get_corr(a, b_prime)
    E_a_prime_b = get_corr(a_prime, b)
    E_a_prime_b_prime = get_corr(a_prime, b_prime)
    
    S = abs(E_ab - E_ab_prime) + abs(E_a_prime_b + E_a_prime_b_prime)
    
    print(f"   E(a, b)       = {E_ab:.4f} (Theory: {-np.cos(a-b):.4f})")
    print(f"   E(a, b')      = {E_ab_prime:.4f} (Theory: {-np.cos(a-b_prime):.4f})")
    print(f"   E(a', b)      = {E_a_prime_b:.4f} (Theory: {-np.cos(a_prime-b):.4f})")
    print(f"   E(a', b')     = {E_a_prime_b_prime:.4f} (Theory: {-np.cos(a_prime-b_prime):.4f})")
    
    print(f"\n   Measured CHSH Parameter S = {S:.4f}")
    print(f"   Classical Limit         S <= 2.0000")
    print(f"   Quantum Theory          S ~= 2.8284")
    
    is_quantum = S > 2.0
    print(f"\n   RESULT: {'VIOLATION CONFIRMED' if is_quantum else 'FAILED'}")
    if is_quantum:
        print("   The data exhibits genuine quantum correlations. The 'Cage' of Local Realism is broken.")
    else:
        print("   The data is classical. Something is wrong with the simulation.")

if __name__ == "__main__":
    run_benchmark()
