import numpy as np
import os
from quantum_entanglement import BellPairSystem

def run_experiment():
    print("=== Experiment B3: The Non-Local Link ===")
    
    sys = BellPairSystem()
    num_samples = 5000
    
    # Generate data with RANDOM detector settings for each pair
    # This makes it harder for the model to just learn a single correlation function.
    # It must learn the relationship E(theta).
    
    print(f"Generating {num_samples} Bell pairs with random axes...")
    
    data = []
    
    for _ in range(num_samples):
        axis_A = np.random.uniform(0, 2*np.pi)
        axis_B = np.random.uniform(0, 2*np.pi)
        
        # Measure 1 pair
        res_A, res_B = sys.measure(axis_A, axis_B, num_pairs=1)
        
        # Store: [axis_A, axis_B, outcome_A, outcome_B]
        data.append([axis_A, axis_B, res_A[0], res_B[0]])
        
    data = np.array(data)
    
    # Save Data
    output_file = "entanglement_data.npy"
    np.save(output_file, data)
    print(f"Data saved to {output_file}. Shape: {data.shape}")
    
    # Verification: Check CHSH Inequality?
    # Or just check simple correlation statistics.
    # Let's check correlation for cases where |axis_A - axis_B| is small.
    
    diffs = np.abs(data[:, 0] - data[:, 1])
    # Wrap around pi
    diffs = np.minimum(diffs, 2*np.pi - diffs)
    
    mask_aligned = diffs < 0.1
    if np.any(mask_aligned):
        subset_A = data[mask_aligned, 2]
        subset_B = data[mask_aligned, 3]
        corr = np.mean(subset_A * subset_B)
        print(f"\nVerification (Aligned axes < 0.1 rad): Correlation = {corr:.4f} (Expected ~ -1.0)")
    else:
        print("\nVerification: No aligned axes found in random sample.")

if __name__ == "__main__":
    run_experiment()
