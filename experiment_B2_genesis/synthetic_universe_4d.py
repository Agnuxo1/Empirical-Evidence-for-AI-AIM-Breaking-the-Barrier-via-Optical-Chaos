import numpy as np
import matplotlib.pyplot as plt

class Universe4D:
    def __init__(self, size=20, c=1.0, dt=0.1, dx=1.0):
        self.size = size
        self.c = c
        self.dt = dt
        self.dx = dx
        # Grid: [x, y, z, w]
        # We use 2 time steps for wave equation (current and previous)
        self.u = np.zeros((size, size, size, size))
        self.u_prev = np.zeros((size, size, size, size))
        self.time_step = 0

    def initialize_gaussian_pulse(self, center, width, amplitude):
        """
        Initialize a Gaussian pulse in 4D.
        center: (cx, cy, cz, cw)
        """
        x = np.arange(self.size)
        X, Y, Z, W = np.meshgrid(x, x, x, x, indexing='ij')
        
        dist_sq = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2 + (W - center[3])**2
        self.u = amplitude * np.exp(-dist_sq / (2 * width**2))
        self.u_prev = self.u.copy() # Start with zero velocity

    def evolve(self):
        """
        Evolve the universe one time step using Finite Difference Method for Wave Equation.
        u_new = 2*u - u_prev + (c*dt/dx)^2 * Laplacian(u)
        """
        # Laplacian in 4D
        # np.roll is slow but easy to read. For performance, we might optimize later.
        laplacian = (
            np.roll(self.u, 1, axis=0) + np.roll(self.u, -1, axis=0) +
            np.roll(self.u, 1, axis=1) + np.roll(self.u, -1, axis=1) +
            np.roll(self.u, 1, axis=2) + np.roll(self.u, -1, axis=2) +
            np.roll(self.u, 1, axis=3) + np.roll(self.u, -1, axis=3) -
            8 * self.u
        )
        
        # Courant number squared
        C2 = (self.c * self.dt / self.dx)**2
        
        u_new = 2 * self.u - self.u_prev + C2 * laplacian
        
        # Update state
        self.u_prev = self.u
        self.u = u_new
        self.time_step += 1

    def get_3d_slice(self, w_index):
        """
        Get a 3D slice of the universe at a specific w coordinate.
        """
        return self.u[:, :, :, w_index]

if __name__ == "__main__":
    # Test: Pulse starting at w=10, observed at w=10 (should look normal)
    # vs Pulse starting at w=5, observed at w=10 (should appear later out of nowhere)
    
    uni = Universe4D(size=16, c=1.0, dt=0.1)
    
    # Pulse centered at (8,8,8, 8)
    uni.initialize_gaussian_pulse(center=(8,8,8,8), width=2.0, amplitude=1.0)
    
    print("Simulating 4D Wave Equation...")
    center_vals = []
    
    for t in range(50):
        uni.evolve()
        # Observe at center of 3D slice w=8
        val = uni.u[8, 8, 8, 8]
        center_vals.append(val)
        
    print(f"Simulation complete. Max value at center: {max(center_vals):.4f}")
