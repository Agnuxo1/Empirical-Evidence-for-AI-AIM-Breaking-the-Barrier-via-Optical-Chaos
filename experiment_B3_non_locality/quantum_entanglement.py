import numpy as np

class BellPairSystem:
    def __init__(self):
        # Singlet state is rotationally invariant.
        # Correlation E(a, b) = -cos(theta_ab)
        pass

    def measure(self, axis_A, axis_B, num_pairs=1000):
        """
        Simulate measurement of 'num_pairs' entangled particles.
        axis_A: Angle of detector A (in radians)
        axis_B: Angle of detector B (in radians)
        
        Returns: (outcomes_A, outcomes_B)
        Outcomes are +1 (Up) or -1 (Down)
        """
        # For the singlet state, the probability of getting (+, +) or (-, -) is (1 - cos(theta))/2
        # The probability of getting (+, -) or (-, +) is (1 + cos(theta))/2
        # Wait, standard result for spin 1/2:
        # P(up, up) = P(down, down) = (1/2) * sin^2(theta/2)
        # P(up, down) = P(down, up) = (1/2) * cos^2(theta/2)
        # where theta is angle between axes.
        # Let's check: if theta=0 (aligned), P(up, down)=1/2 * 1 = 0.5. Total anti-correlation?
        # Singlet is |ud> - |du>. 
        # If axes aligned: always opposite. P(u,d)=0.5, P(d,u)=0.5. P(u,u)=0. 
        # Correct.
        
        theta = axis_A - axis_B
        
        # Probabilities for outcomes given the first measurement
        # We can simulate this by:
        # 1. Measure A (random 50/50)
        # 2. Collapse state for B
        # 3. Measure B
        
        outcomes_A = np.random.choice([1, -1], size=num_pairs)
        outcomes_B = np.zeros(num_pairs)
        
        for i in range(num_pairs):
            # If A is +1 (Up along axis_A), B is state |Down> along axis_A.
            # We measure B along axis_B.
            # The angle between |Down>_A and |Up>_B is (pi - theta)?
            # Let's use the correlation directly.
            # P(B = -A) = cos^2(theta/2)
            # P(B = A)  = sin^2(theta/2)
            
            p_same = np.sin(theta/2)**2
            
            if np.random.random() < p_same:
                outcomes_B[i] = outcomes_A[i]
            else:
                outcomes_B[i] = -outcomes_A[i]
                
        return outcomes_A, outcomes_B

if __name__ == "__main__":
    sys = BellPairSystem()
    # Test: Aligned axes (theta=0). Should be perfectly anti-correlated.
    A, B = sys.measure(0, 0, 10)
    print("Aligned (0,0):")
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"Correlation: {np.mean(A*B)}") # Should be -1
    
    # Test: Opposite axes (theta=pi). Should be perfectly correlated.
    A, B = sys.measure(0, np.pi, 10)
    print("\nOpposite (0,pi):")
    print(f"Correlation: {np.mean(A*B)}") # Should be +1
