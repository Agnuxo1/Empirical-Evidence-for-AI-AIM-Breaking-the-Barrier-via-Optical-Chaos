# Experiment 2: Einstein's Train (The Photon Clock)

## Credits and References

**Darwin's Cage Theory:**
- **Theory Creator**: Gideon Samid
- **Reference**: Samid, G. (2025). Negotiating Darwin's Barrier: Evolution Limits Our View of Reality, AI Breaks Through. *Applied Physics Research*, 17(2), 102. https://doi.org/10.5539/apr.v17n2p102
- **Publication**: Applied Physics Research; Vol. 17, No. 2; 2025. ISSN 1916-9639 E-ISSN 1916-9647. Published by Canadian Center of Science and Education
- **Available at**: https://www.researchgate.net/publication/396377476_Negotiating_Darwin's_Barrier_Evolution_Limits_Our_View_of_Reality_AI_Breaks_Through

**Experiments, AI Models, Architectures, and Reports:**
- **Author**: Francisco Angulo de Lafuente
- **Responsibilities**: Experimental design, AI model creation, architecture development, results analysis, and report writing

---

## Abstract
This experiment investigates whether a chaotic optical network can predict Relativistic Time Dilation ($\gamma$) purely from geometric interference patterns, without knowledge of the Lorentz Transformation formula. We further subject the model to critical stress tests to distinguish between true geometric learning and mere curve fitting.

## Objective
To demonstrate that the non-linear phase shifts in a complex optical reservoir can map the observed zigzag path of a photon to its time dilation factor, effectively solving $ \gamma = \frac{1}{\sqrt{1 - v^2/c^2}} $ through analog wave interference.

## Methodology

### 1. The Relativistic Simulator
We simulate a "Light Clock" moving at velocity $v$.
*   **Proper Time ($\Delta t_0$)**: The time for a photon to bounce vertically in the train's frame.
*   **Observed Path**: An external observer sees the photon travel a diagonal path.
*   **Input**: The geometric components of the photon's path (Horizontal Distance $d_x$, Vertical Distance $L$).
*   **Target**: The Lorentz Factor $\gamma$.

### 2. The Optical Interference Net
The model mimics a coherent optical processor:
1.  **Input Encoding**: Path geometry is normalized to $[0, 1]$.
2.  **Optical Scattering**: Inputs are projected into a high-dimensional complex space ($N=5000$) via a fixed random scattering matrix.
3.  **Phase Encoding**: The network encodes information in the *phase* of the complex signal, simulating how light accumulates phase delay over distance.
4.  **Interference**: A Holographic FFT mixes the signals.
5.  **Detection**: Intensity is measured ($|z|^2$) and passed to a linear readout.

## Results

### Standard Performance
On a standard random split of velocities ($v \in [0, 0.99c]$):

| Model | R² Score | Note |
| :--- | :--- | :--- |
| **Darwinian (Polynomial)** | **0.9999** | High accuracy, but relies on Taylor approximation. |
| **Optical AI** | **1.0000** | **Perfect prediction.** |

### Critical Analysis & Stress Tests
To verify the validity of the results, we performed a "Stress Test" (`stress_test_relativity.py`) to check for overfitting and robustness.

#### Test 1: Extrapolation (The "Learning" Check)
*   **Protocol**: Train ONLY on low speeds ($v < 0.75c$). Test on unseen high speeds ($v > 0.75c$).
*   **Result**: **R² = 0.944** (Passed)
*   **Conclusion**: The model successfully predicted the asymptotic behavior of time dilation near the speed of light without ever seeing it during training. This confirms the model **learned the underlying geometric relationship** rather than memorizing data points.

#### Test 2: Noise Robustness (The "Fragility" Check)
*   **Protocol**: Add 5% random noise to the input observations.
*   **Result**: **R² = 0.396** (Failed)
*   **Conclusion**: The system is highly sensitive to noise. Like a physical interferometer, small perturbations in the input phase destroy the coherence of the output pattern. While the theoretical model is sound, a physical implementation would require significant error correction.

## Internal "Cage" Analysis
We analyzed the internal features of the optical network to see if it reconstructed the variable $v^2$ (the key term in the human formula).
*   **Max Correlation with $v^2$**: ~0.01
*   **Interpretation**: The AI did **not** calculate $v^2$. It found an alternative, purely geometric pathway to the solution, validating the "Cage-Free" hypothesis.

## Files
*   `experiment_2_einstein_train.py`: Main experiment script.
*   `stress_test_relativity.py`: Critical audit script.
*   `experiment_2_relativity.png`: Visualization of predictions.
*   `stress_test_results.png`: Visualization of extrapolation and noise tests.

## Reproduction
```bash
python experiment_2_einstein_train.py
python stress_test_relativity.py
```
