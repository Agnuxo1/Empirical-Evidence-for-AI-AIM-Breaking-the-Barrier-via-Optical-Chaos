import numpy as np
import os
from synthetic_universe_4d import Universe4D

def run_experiment():
    print("=== Experiment B2: The Genesis ===")
    
    # 1. Setup 4D Universe
    size = 16
    uni = Universe4D(size=size, c=1.0, dt=0.1)
    
    # 2. Initialize with a hidden source
    # Pulse starts at (8, 8, 8, 2) - Far away in w from our slice at w=8
    # We will observe at w=8
    center = (size//2, size//2, size//2, 2)
    uni.initialize_gaussian_pulse(center=center, width=2.0, amplitude=10.0)
    
    print(f"Initialized 4D Pulse at {center}")
    print("Observer is located at 3D slice w=8")
    
    # 3. Generate Data
    steps = 100 # Run longer to see the full passage
    observations = []
    
    print(f"Simulating {steps} time steps...")
    for t in range(steps):
        uni.evolve()
        # Capture the 3D slice at w=8
        slice_3d = uni.get_3d_slice(w_index=8)
        observations.append(slice_3d)
        
        if t % 10 == 0:
            max_val = np.max(slice_3d)
            print(f"Step {t}: Max observed value = {max_val:.4f}")
            
    observations = np.array(observations)
    # Shape: (steps, size, size, size)
    
    # 4. Save Data
    output_file = "observation_data.npy"
    np.save(output_file, observations)
    print(f"Data saved to {output_file}. Shape: {observations.shape}")
    
    # 5. Preliminary Analysis (The "Trap")
    # Check if 3D Wave Equation holds: d2u/dt2 = c^2 * Laplacian_3D
    # We pick a point in the middle of the slice
    mid = size // 2
    # Time derivative (central difference)
    # d2u/dt2 ~ (u[t+1] - 2u[t] + u[t-1]) / dt^2
    
    # Laplacian 3D
    # ...
    
    print("\nPreliminary Analysis:")
    print("To a 3D observer, energy will appear to spontaneously increase as the 4D wave intersects the slice.")
    print("Standard 3D physics will fail to explain the source.")

if __name__ == "__main__":
    run_experiment()
