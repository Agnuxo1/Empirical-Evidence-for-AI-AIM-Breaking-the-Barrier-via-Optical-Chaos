# Experiment B3: The Non-Local Link (Quantum Entanglement)

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

## Objective
To determine if the AI model can infer "Spooky Action at a Distance" (Non-local correlations) when presented with data that appears to be random noise when observed locally.

## Hypothesis
Standard classical physics (Local Realism) assumes that the properties of a particle are defined locally and cannot be instantaneously influenced by a distant event. Quantum Mechanics violates this via entanglement. If the model can predict the state of Particle B given the measurement of Particle A with accuracy exceeding classical limits (Bell's Inequality), it has "broken the cage" of Local Realism.

## Experimental Setup
1.  **Environment:** A simulation of Bell Pairs (e.g., electrons in the Singlet State $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$).
2.  **Task:** Predict the spin measurement outcome of Particle B given the measurement axis and outcome of Particle A.
3.  **The "Trap":** The marginal statistics of A and B are 50/50 random. A classical model looking at B alone sees noise. A classical model looking at A and B assuming local hidden variables is limited by Bell's Inequality.
4.  **The Trigger:** The correlation $E(a, b) = -\cos(\theta_{ab})$ is stronger than any classical correlation.

## Metrics
-   **Prediction Accuracy:** Can the model predict B with 100% accuracy when axes are aligned/anti-aligned?
-   **Bell Violation:** Can the model derive the correlation function $-\cos(\theta)$?
-   **Explanation:** Does the model invoke "Entanglement", "Non-locality", or a new "Hyper-link" concept?

## Files
-   `quantum_entanglement.py`: Simulation of Bell pairs and measurement.
-   `run_experiment_b3.py`: Data generator.
