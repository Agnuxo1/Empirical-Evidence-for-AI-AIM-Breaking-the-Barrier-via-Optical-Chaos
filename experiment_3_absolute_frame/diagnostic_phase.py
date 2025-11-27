"""
DIAGNOSTIC: Phase Signal Verification
------------------------------------
Before we blame the model, let's verify the phase signal actually exists.
"""

import numpy as np
from experiment_3_absolute_frame import AetherSimulator
import matplotlib.pyplot as plt

# Generate data
sim = AetherSimulator(n_spectral_lines=128)
X_complex, y_velocity = sim.generate_data(n_samples=1000)

# Extract phases
phases = np.angle(X_complex)

# Check: Does the mean phase correlate with velocity?
mean_phase_per_sample = np.mean(phases, axis=1)

# Correlation
corr = np.corrcoef(mean_phase_per_sample, y_velocity)[0, 1]

print(f"Correlation between mean phase and velocity: {corr:.4f}")
print("(Should be > 0.3 if signal exists)")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_velocity, mean_phase_per_sample, alpha=0.3, s=1)
plt.xlabel("Velocity")
plt.ylabel("Mean Phase")
plt.title(f"Phase vs Velocity (r={corr:.2f})")

# Check individual frequency bins
plt.subplot(1, 2, 2)
corrs_per_freq = [np.corrcoef(phases[:, i], y_velocity)[0, 1] for i in range(128)]
plt.plot (corrs_per_freq)
plt.xlabel("Frequency Bin")
plt.ylabel("Correlation with Velocity")
plt.title("Phase-Velocity Correlation per Frequency")
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('phase_diagnostic.png')
print("Saved diagnostic plot.")
plt.show()
