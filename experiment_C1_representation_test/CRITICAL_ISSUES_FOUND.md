# Critical Issues Found in Experiment C1

## Issue 1: Random Seed Bias (CRITICAL BUG)

**Problem:** Both models use the same random_seed (1337), but they have different input dimensions:
- Anthropomorphic: 2 dimensions → matrix (2, 4096)
- Non-anthropomorphic: 4 dimensions → matrix (4, 4096)

**Impact:** With the same seed, the first 2 rows of the optical matrix would be identical, but the non-anthropomorphic model gets 2 additional rows. This creates a systematic bias - not a fair comparison.

**Fix:** Use different seeds OR ensure matrices are independently generated. Actually, since dimensions differ, using the same seed is fine - each model gets a different matrix size, so they're independent. But we should verify this doesn't create bias.

## Issue 2: Correlation Analysis Bias (CRITICAL BIAS)

**Problem:** We're analyzing correlations with v0 and angle, but these are DERIVED from the non-anthropomorphic representation:
- v0 = sqrt(vx² + vy²) where vx, vy are in non-anthropomorphic inputs
- angle = arctan2(vy, vx) where vx, vy are in non-anthropomorphic inputs

**Impact:** This creates an artificial advantage for non-anthropomorphic representation - it's more directly related to the variables we're checking correlations with.

**Fix:** We should analyze correlations with variables that are EQUALLY derivable from both representations, OR use different analysis variables for each representation.

## Issue 3: Dimensionality Confound (POTENTIAL BIAS)

**Problem:** Non-anthropomorphic has 4 dimensions vs 2 for anthropomorphic. This could affect learning differently, not just due to representation but also dimensionality.

**Impact:** Differences might be due to dimensionality, not representation type.

**Fix:** This is acknowledged as a limitation. The experiment tests "representation + dimensionality" together, which is still valid but should be noted.

## Issue 4: Scaling Independence (MINOR)

**Problem:** Two separate scalers are used. While both use MinMaxScaler, they scale independently.

**Impact:** Minor - both scale to [0,1], so should be comparable. But slight differences in scaling could affect results.

**Fix:** This is acceptable - each representation needs its own scaler since value ranges differ.

