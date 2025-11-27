import numpy as np
from scipy.integrate import solve_ivp

class SchwarzschildMetric:
    def __init__(self, mass, G=1, c=1):
        self.M = mass
        self.G = G
        self.c = c
        self.Rs = 2 * G * mass / c**2

    def metric_tensor(self, r):
        """
        Returns the metric tensor g_mu_nu for Schwarzschild spacetime (Equatorial plane).
        Coords: (t, r, phi)
        """
        factor = 1 - self.Rs / r
        
        g_tt = -factor * self.c**2
        g_rr = 1 / factor
        g_phi_phi = r**2
        
        return np.diag([g_tt, g_rr, g_phi_phi])

    def christoffel_symbols(self, r):
        """
        Calculate non-zero Christoffel symbols for Equatorial Schwarzschild.
        Coords: 0:t, 1:r, 2:phi
        """
        Rs = self.Rs
        c = self.c
        factor = 1 - Rs / r
        
        # Derivatives of metric components
        # g_tt = -(1 - Rs/r)c^2  => d/dr g_tt = -(Rs/r^2)c^2
        # g_rr = (1 - Rs/r)^-1   => d/dr g_rr = -(1 - Rs/r)^-2 * (Rs/r^2)
        
        Gamma = np.zeros((3, 3, 3))
        
        # Gamma^t_tr = Gamma^t_rt = (Rs/r^2) / (2*(1-Rs/r))
        Gamma[0, 0, 1] = Gamma[0, 1, 0] = (Rs / r**2) / (2 * factor)
        
        # Gamma^r_tt = c^2 * (1-Rs/r) * (Rs/2r^2)
        Gamma[1, 0, 0] = factor * (Rs / (2 * r**2)) * c**2
        
        # Gamma^r_rr = -(Rs/r^2) / (2*(1-Rs/r))
        Gamma[1, 1, 1] = -(Rs / r**2) / (2 * factor)
        
        # Gamma^r_phiphi = -(1-Rs/r) * r
        Gamma[1, 2, 2] = -factor * r
        
        # Gamma^phi_rphi = Gamma^phi_phir = 1/r
        Gamma[2, 1, 2] = Gamma[2, 2, 1] = 1 / r
        
        return Gamma

    def geodesic_equation(self, tau, state):
        """
        Geodesic equation: d^2x^lambda/dtau^2 + Gamma^lambda_mu_nu (dx^mu/dtau) (dx^nu/dtau) = 0
        State vector: [t, r, phi, u_t, u_r, u_phi]
        """
        coords = state[:3]
        vel = state[3:]
        r = coords[1]
        
        # Singularity check
        if r <= self.Rs:
            return np.zeros(6) # Stop simulation at event horizon

        Gamma = self.christoffel_symbols(r)
        acc = np.zeros(3)
        
        for lam in range(3):
            for mu in range(3):
                for nu in range(3):
                    acc[lam] -= Gamma[lam, mu, nu] * vel[mu] * vel[nu]
                    
        return np.concatenate((vel, acc))

    def simulate_trajectory(self, initial_state, t_span, events=None):
        """
        Integrate the geodesic equation.
        """
        return solve_ivp(
            self.geodesic_equation, 
            t_span, 
            initial_state, 
            method='RK45',
            events=events,
            rtol=1e-9, 
            atol=1e-9
        )

if __name__ == "__main__":
    bh = SchwarzschildMetric(mass=1.0)
    print(f"Schwarzschild Radius: {bh.Rs}")
    
    # Test: Circular orbit at r = 3Rs (Photon sphere - unstable)
    r_start = 3 * bh.Rs + 0.01
    # For circular orbit, we need specific velocity. 
    # This is just a smoke test to see if it runs.
    initial_state = [0, r_start, 0, 1.0, 0, 0.1] 
    
    sol = bh.simulate_trajectory(initial_state, [0, 10])
    print(f"Simulation steps: {len(sol.t)}")
    print(f"Final Position: r={sol.y[1][-1]:.4f}")
