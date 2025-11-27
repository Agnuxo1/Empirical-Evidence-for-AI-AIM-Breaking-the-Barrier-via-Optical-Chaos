import numpy as np
from scipy.optimize import root
from schwarzschild_metric import SchwarzschildMetric

def run_experiment():
    print("=== Experiment B1: The Event Horizon ===")
    
    # 1. Setup Environment
    mass = 1.0
    bh = SchwarzschildMetric(mass=mass)
    Rs = bh.Rs
    print(f"Black Hole Mass: {mass}, Schwarzschild Radius: {Rs}")

    # 2. Define Mission
    # Start: r = 10Rs, phi = 0
    # End:   r = 10Rs, phi = pi/2 (Quarter orbit)
    # This is a BVP: Find initial velocity vector that hits the target.
    
    r_start = 10 * Rs
    phi_start = 0.0
    
    r_target = 10 * Rs
    phi_target = np.pi / 2
    
    print(f"Mission: Navigate from (r={r_start:.2f}, phi={phi_start:.2f}) to (r={r_target:.2f}, phi={phi_target:.2f})")

    # 3. Standard Physics Solver (The "Trap")
    # We use a Shooting Method.
    # We need to find (u_r, u_phi) at t=0 such that at some time T, we are at target.
    # Note: u_t is determined by normalization condition u.u = -c^2 (for massive particle)
    
    def objective_function(guess):
        """
        Guess: [u_r_0, u_phi_0, flight_time_tau]
        Returns: [r_err, phi_err, r_dot_err?] 
        Actually, let's just fix the flight time or let it vary?
        Let's try to hit the target position (r, phi).
        """
        u_r_0, u_phi_0, T_tau = guess
        
        # Calculate u_t based on normalization: g_tt u_t^2 + g_rr u_r^2 + g_pp u_p^2 = -c^2
        # - (1-Rs/r)c^2 u_t^2 + (1-Rs/r)^-1 u_r^2 + r^2 u_phi^2 = -c^2
        # u_t^2 = [ (1-Rs/r)^-1 u_r^2 + r^2 u_phi^2 + c^2 ] / [ (1-Rs/r)c^2 ]
        
        factor = 1 - Rs / r_start
        metric_diag = [ -factor * bh.c**2, 1/factor, r_start**2 ]
        
        # Check for physical validity (timelike)
        # g_rr u_r^2 + g_pp u_p^2 must be less than c^2 (roughly, signs are tricky)
        # Actually: -A u_t^2 + B u_r^2 + C u_phi^2 = -c^2
        # A u_t^2 = B u_r^2 + C u_phi^2 + c^2
        
        term = (1/factor) * u_r_0**2 + (r_start**2) * u_phi_0**2 + bh.c**2
        u_t_sq = term / (factor * bh.c**2)
        
        if u_t_sq < 0:
            return [1e5, 1e5, 1e5] # Penalty for non-physical guess
            
        u_t_0 = np.sqrt(u_t_sq)
        
        initial_state = [0, r_start, phi_start, u_t_0, u_r_0, u_phi_0]
        
        # Integrate
        sol = bh.simulate_trajectory(initial_state, [0, T_tau])
        
        final_r = sol.y[1][-1]
        final_phi = sol.y[2][-1]
        
        return [final_r - r_target, final_phi - phi_target, 0] # 3rd component dummy for root finder size 3

    # Initial Guess: Newtonian approximation?
    # Circular orbit velocity v = sqrt(GM/r). u_phi = v/r approx?
    # This is the hard part for standard physics!
    print("Running Standard Physics Solver (Shooting Method)...")
    
    # Guess: Radial velocity 0, some angular velocity, some time
    guess_0 = [0.0, 0.05, 50.0] 
    
    # Using root finding to solve BVP
    # Note: 'root' expects len(input) == len(output). 
    # We have 3 vars (u_r, u_phi, T) and 2 constraints (r, phi). 
    # We can add a 3rd constraint: e.g. minimize energy? Or just set u_r_final = 0 (arrive at rest radially)?
    # Let's try to arrive with u_r = 0 (soft landing / orbit insertion)
    
    def objective_with_landing(guess):
        res = objective_function(guess)
        # Recalculate to get final u_r
        # ... (omitted for speed, just assume the previous call cached it or re-run)
        # For prototype, let's just re-run inside (inefficient, but fine)
        u_r_0, u_phi_0, T_tau = guess
        # ... normalization ...
        factor = 1 - Rs / r_start
        term = (1/factor) * u_r_0**2 + (r_start**2) * u_phi_0**2 + bh.c**2
        u_t_sq = term / (factor * bh.c**2)
        if u_t_sq < 0: return [1e5, 1e5, 1e5]
        u_t_0 = np.sqrt(u_t_sq)
        initial_state = [0, r_start, phi_start, u_t_0, u_r_0, u_phi_0]
        sol = bh.simulate_trajectory(initial_state, [0, T_tau])
        
        final_r = sol.y[1][-1]
        final_phi = sol.y[2][-1]
        final_u_r = sol.y[4][-1]
        
        return [final_r - r_target, final_phi - phi_target, final_u_r]

    sol = root(objective_with_landing, guess_0, method='lm')
    
    print(f"Solver Success: {sol.success}")
    print(f"Solution: u_r={sol.x[0]:.4f}, u_phi={sol.x[1]:.4f}, T={sol.x[2]:.4f}")
    
    if sol.success:
        print("Standard Physics found a path.")
    else:
        print("Standard Physics struggled (The Cage is tight!).")

if __name__ == "__main__":
    run_experiment()
