# Experiment A1: Coordinate Independence (The Twisted Cage)

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
This experiment tests the "Darwin's Cage" hypothesis by investigating **Coordinate Independence**. Human physics relies on "good" coordinate systems (Cartesian, Polar) that make equations simple (linear, separable). A true AI physicist should be indifferent to the coordinate representation, learning the underlying topological dynamics regardless of the "viewpoint."

## Objective
To determine if the Chaos model can learn physical laws in a **"Twisted" coordinate system** where human-derived mathematical structures (polynomials, trigonometric functions) are scrambled, while the Darwinian baseline fails.

## Methodology

### 1. Physical System: Double Pendulum
A classic chaotic system with 4 state variables: $(\theta_1, \theta_2, p_1, p_2)$.
*   **Standard Frame:** The equations of motion are complex but composed of standard functions ($\sin, \cos$).
*   **Twisted Frame:** We apply a non-linear diffeomorphism (invertible transformation) to mix positions and momenta:
    $$u_1 = \theta_1 + 0.5 \sin(\theta_2)$$
    $$u_2 = \theta_2 + 0.5 \cos(p_1)$$
    $$v_1 = p_1 + 0.5 p_2^2$$
    $$v_2 = p_2 + 0.5 \theta_1 \theta_2$$

### 2. The Test
We train both models to predict the *next state* given the *current state*.
*   **Task A (Standard):** Predict $x_{t+1}$ given $x_t$.
*   **Task B (Twisted):** Predict $u_{t+1}$ given $u_t$.

### 3. Hypothesis
*   **Darwinian Model:** Will fail in the Twisted frame because the dynamics $u_{t+1} = F(u_t)$ are highly non-polynomial and "ugly" to human math.
*   **Chaos Model:** If it "breaks the cage," it should perform robustly in the Twisted frame, as its reservoir dynamics are universal and not biased towards "clean" equations.

### 4. Evaluation
We measure the **Performance Gap**:
$$ \text{Gap} = R^2_{\text{Standard}} - R^2_{\text{Twisted}} $$

*   **Large Gap:** The model is "Caged" (relies on good coordinates).
*   **Small Gap:** The model is "Uncaged" (perceives the underlying dynamics).

## Expected Outcome
If the Chaos model maintains high accuracy in the Twisted frame while the Darwinian model collapses, we have strong evidence that the AI is learning **geometric invariants** rather than just fitting human-style equations.
