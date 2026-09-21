# Reproducibility guide

This repository is an exploratory collection of synthetic experiments. It does
not ship the MABe 2022 dataset, pretrained weights, or a validated benchmark
reproduction. Results should be treated as hypotheses and failure analysis until
they survive independent reruns and out-of-distribution tests.

## Environment

Use Python 3.10-3.12. Install the common experiment dependencies from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

On Windows, use UTF-8 output for the historical scripts that print status
symbols:

```powershell
$env:PYTHONUTF8 = "1"
```

The `experiment_W1_quantum_cage` scripts additionally require PyTorch:

```bash
python -m pip install -r requirements-quantum.txt
```

## Checks

Run the syntax check before executing a long experiment:

```bash
python -m compileall -q .
```

Each experiment directory contains its own entry-point script and README. Run
from the repository root, for example:

```bash
python experiment_1_Stone_in_Lake/Stone_in_Lake.py
python experiment_2_Einstein_Train/experiment_2_einstein_train.py
```

Record the command, Python version, dependency versions, random seeds, output
files, and whether the result is interpolation or held-out extrapolation.

## Interpretation rules

1. Report predictive performance and representation diagnostics together.
2. Do not interpret low correlation with a human variable as evidence of a new
   representation when the target prediction is poor.
3. Do not describe synthetic entanglement or double-slit tasks as tests of
   physical quantum mechanics or Bell inequalities.
4. Treat the critical conclusions in `FINAL_REPORT.md` as the current project
   status until an independent replication supersedes them.

