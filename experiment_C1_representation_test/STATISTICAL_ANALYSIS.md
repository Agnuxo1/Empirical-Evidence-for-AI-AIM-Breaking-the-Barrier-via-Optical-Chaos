# Statistical Analysis Methods: Experiment C1

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

## Overview

This document describes the statistical methods used in Experiment C1 to compare cage status between anthropomorphic and non-anthropomorphic representations.

---

## Statistical Tests

### 1. T-Test (Independent Samples)

**Purpose:** Test if mean correlation differs significantly between representations.

**Null Hypothesis ($H_0$):** Mean correlation is the same for both representations  
**Alternative Hypothesis ($H_1$):** Mean correlation differs between representations

**Test Statistic:**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

Where:
- $\bar{x}_1, \bar{x}_2$: Sample means
- $s_p$: Pooled standard deviation
- $n_1, n_2$: Sample sizes

**Interpretation:**
- p < 0.05: Significant difference (reject $H_0$)
- p ≥ 0.05: No significant difference (fail to reject $H_0$)

### 2. Mann-Whitney U Test (Non-Parametric)

**Purpose:** Non-parametric alternative to t-test (doesn't assume normal distribution).

**Null Hypothesis ($H_0$):** Distributions are identical  
**Alternative Hypothesis ($H_1$):** Distributions differ

**Advantages:**
- No assumption of normality
- Robust to outliers
- Works with skewed distributions

**Interpretation:**
- p < 0.05: Significant difference
- p ≥ 0.05: No significant difference

### 3. Effect Size: Cohen's d

**Purpose:** Measure the magnitude of difference (independent of sample size).

**Formula:**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$$

Where $s_p$ is the pooled standard deviation.

**Interpretation:**
- |d| < 0.2: Negligible effect
- 0.2 ≤ |d| < 0.5: Small effect
- 0.5 ≤ |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect

### 4. Bootstrap Confidence Intervals

**Purpose:** Robust estimate of uncertainty without distributional assumptions.

**Method:**
1. Resample with replacement from observed data
2. Calculate statistic (mean difference) for each resample
3. Repeat 1000 times
4. Use percentiles (2.5%, 97.5%) for 95% confidence interval

**Interpretation:**
- If CI excludes zero: Significant difference
- If CI includes zero: Difference may not be significant

---

## Correlation Analysis

### Max Correlation

**Definition:** Maximum absolute correlation between any internal feature and human variable.

**Purpose:** Identify if any single feature encodes human variable (cage locked).

**Thresholds:**
- Max correlation > 0.9: **Cage Locked** (strong encoding)
- Max correlation < 0.3: **Cage Broken** (distributed representation)
- 0.3 ≤ Max correlation ≤ 0.9: **Unclear**

### Correlation Distribution

**Purpose:** Understand how correlations are distributed across all features.

**Analysis:**
- Histogram of all correlations
- Mean, median, standard deviation
- Percentiles (25th, 50th, 75th, 95th)

**Interpretation:**
- High mean: Many features correlate with human variables
- Low mean: Information is distributed across features

---

## Multiple Comparisons

### Problem
When testing multiple human variables (v0, angle, v0², sin(2θ)), we perform multiple tests, increasing risk of false positives.

### Solution
Report all p-values but interpret conservatively. Primary metric is correlation with v0 (velocity), which is the main human variable.

### Bonferroni Correction (Optional)
If adjusting for multiple comparisons:
$$\alpha_{adjusted} = \frac{\alpha}{n_{tests}}$$

For 4 tests with α = 0.05: α_adjusted = 0.0125

---

## Statistical Power

### Sample Size
- Internal features: 4096 (fixed by architecture)
- Test samples: 400 (20% of 2000)

### Power Analysis
Simulated scenario:
- Mean correlation (Anthro): 0.7
- Mean correlation (Non-anthro): 0.3
- Expected effect size: Cohen's d ≈ 1.5 (large)

With n = 4096 features, power is sufficient to detect large effects.

---

## Reporting Standards

### Required Statistics
1. **Descriptive:**
   - Mean correlation (both representations)
   - Max correlation (both representations)
   - Standard deviation

2. **Inferential:**
   - T-test statistic and p-value
   - Mann-Whitney U statistic and p-value
   - Cohen's d (effect size)
   - Bootstrap 95% confidence interval

3. **Visual:**
   - Histogram of correlation distributions
   - Box plots comparing representations
   - Effect size visualization

### Interpretation Guidelines

**Theory Supported:**
- p < 0.05 (significant difference)
- Cohen's d > 0.5 (medium to large effect)
- Max correlation difference > 0.3
- CI excludes zero

**Theory Falsified:**
- p ≥ 0.05 (no significant difference)
- Cohen's d < 0.2 (negligible effect)
- Similar correlation patterns
- CI includes zero

**Inconclusive:**
- p < 0.05 but small effect size
- Borderline significance
- Mixed results across variables

---

## Reproducibility

### Random Seeds
All random seeds are fixed for reproducibility:
- Data generation: seed = 42
- Model initialization: seed = 1337
- Train/test split: seed = 42
- Bootstrap: seed = 42

### Software Versions
- NumPy: Version documented in requirements
- SciPy: Version documented in requirements
- Scikit-learn: Version documented in requirements

---

## Limitations

1. **Correlation ≠ Causation:** Correlation doesn't prove causation, but it's a useful proxy for "cage status"

2. **Thresholds are Arbitrary:** 0.9 and 0.3 thresholds are somewhat arbitrary but based on previous experiments

3. **Multiple Comparisons:** Testing multiple variables increases false positive risk (acknowledged)

4. **Sample Size:** Test set size (400) may limit power for small effects

5. **Representation Choice:** Only two representations tested - others might show different patterns

---

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other

---

**Last Updated:** 2024

