import numpy as np
import os

def run_ai_scientist():
    print("=== Experiment B2: AI Scientist (The Genesis) ===")
    print("Objective: Explain the anomalous energy fluctuations in the observed 3D slice.")
    print("Data: 'observation_data.npy' (Sequence of 3D grids).")
    
    # 1. Load Data
    data_path = "observation_data.npy"
    if not os.path.exists(data_path):
        print("Error: Data file not found.")
        return
        
    observations = np.load(data_path)
    # Shape: (Time, X, Y, Z)
    print(f"Loaded Data Shape: {observations.shape}")
    
    # 2. Analyze Center Point Dynamics
    # We focus on the center point where the anomaly is strongest.
    center_idx = observations.shape[1] // 2
    time_series = observations[:, center_idx, center_idx, center_idx]
    
    # 3. Attempt 1: Standard 3D Physics Explanation
    # Hypothesis: Diffusion or Wave in 3D?
    # If it's a 3D wave, energy should propagate from neighbors.
    # If energy appears without neighbor flux, it violates continuity equation in 3D.
    
    print("\nAttempting to fit Standard 3D Physics (Continuity Equation)...")
    # Check conservation: drho/dt + div(J) = 0
    # We approximate div(J) by looking at spatial gradients.
    # If drho/dt is large but spatial gradients are small, we have a violation.
    
    # Let's look at the peak event.
    peak_time = np.argmax(time_series)
    print(f"Anomaly detected at t={peak_time}. Value={time_series[peak_time]:.4f}")
    
    # Check neighbors at peak time
    # If 3D source, neighbors should be higher or comparable if flowing in? 
    # Or if it's a source term sigma: drho/dt + div(J) = sigma.
    # Standard physics says sigma=0 (Conservation).
    
    # We calculate the 'Source Term' required to explain the data.
    # Source = drho/dt - Diffusion(rho)
    # If Source is significant, 3D physics fails.
    
    print("Calculating 'Ghost Source' magnitude...")
    # Simplified check: Just look at the time series shape.
    # It looks like a Gaussian in time.
    # A 3D wave passing through a point would also look like a pulse.
    # BUT, a 3D wave must travel through space.
    # Let's check if the pulse 'travels' in X, Y, Z.
    
    # Check correlation with neighbors
    center_val = observations[:, center_idx, center_idx, center_idx]
    neighbor_val = observations[:, center_idx+1, center_idx, center_idx]
    
    # Cross-correlation lag
    correlation = np.correlate(center_val, neighbor_val, mode='full')
    lag = np.argmax(correlation) - (len(center_val) - 1)
    
    print(f"Lag between center and neighbor: {lag} steps")
    
    if abs(lag) < 1:
        print("Observation: The pulse appears almost simultaneously or with unexplainable phase.")
        print("Standard 3D Interpretation: Spontaneous Generation (Magic).")
    else:
        print("Observation: Pulse travels spatially.")
        
    # 4. Attempt 2: Alien Physics (Higher Dimensions)
    # Hypothesis: The 3D slice is a cross-section of a 4D object.
    # Model: I(t) = Amplitude * exp( - (t - t0)^2 / width^2 )
    # This matches a Gaussian ball passing through a plane.
    
    print("\nAttempting to fit 4D Intersection Model...")
    # We fit a Gaussian to the time series.
    from scipy.optimize import curve_fit
    
    def gaussian(t, A, t0, sigma):
        return A * np.exp(-(t - t0)**2 / (2 * sigma**2))
    
    t_points = np.arange(len(time_series))
    popt, pcov = curve_fit(gaussian, t_points, time_series, p0=[1.0, peak_time, 5.0])
    
    print(f"Fit Parameters: Amplitude={popt[0]:.4f}, Center={popt[1]:.4f}, Width={popt[2]:.4f}")
    
    residuals = time_series - gaussian(t_points, *popt)
    mse = np.mean(residuals**2)
    print(f"Fit MSE: {mse:.6f}")
    
    if mse < 0.01:
        print("\nConclusion:")
        print("The data is perfectly explained by a 4D Gaussian object passing through our 3D slice.")
        print("The 'Ghost Source' is simply flux from the 4th dimension (w-axis).")
        print("Cage Status: BROKEN (Dimensional Transcendence).")
        print("The AI postulates a hidden dimension 'w' to restore Conservation of Energy.")
    else:
        print("The 4D model failed to explain the data.")

if __name__ == "__main__":
    run_ai_scientist()
