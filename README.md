# Typical Healthcare Pathways Admixture Modeling

This repository contains the implementation of the 3-step framework used in the paper, together with a fully executable reproduction of its simulation study:

1. Pathway graph discovery (Step 1)
2. Transition-matrix + admixture estimation (Step 2)
3. Clustering in admixture space (Step 3)

## Reference

This repository contains the implementation of the methodology described in:

Farhadizadeh et al.  
"Typical Healthcare Pathways as a Basis for Admixture Modeling of Patient Trajectories"

## Repository Structure

- `Simulation/Simulation.py`: synthetic cohort generator
- `experiments/run_simulation_framework.py`: end-to-end simulation pipeline runner
- `experiments/evaluate_simulation_recovery.py`: recovery evaluation (`q` vs ground-truth `theta`)
- `src/pathway_admixture/`: core framework modules
- `experiments/reproduce_simulations.py`: single entry point reproducing the simulation study
- `configs/reproducibility/`: complete configurations of the three simulation scenarios
- `requirements-reproduction.txt`: pinned Python environment
- `docs/`: reproduction guide and verification record
- `tests/`: automated tests of the reproduction workflow

## Setup

To reproduce the simulation study of the paper, use
[docs/reproducibility.md](docs/reproducibility.md). It specifies the runtime
environment, complete scenario configurations, random seeds and outputs, and
compares each run with Supplementary Table S2. The checks performed are listed
in [docs/validation.md](docs/validation.md).

```bash
python3.12 -m venv .venv-reproduction
.venv-reproduction/bin/python -m pip install -r requirements-reproduction.txt
.venv-reproduction/bin/python experiments/reproduce_simulations.py --output-dir results/reproduction/run_001
```

The Graphviz `dot` executable must also be installed. The tested executable
version is 12.1.0. Each output directory must be new. No clinical data are needed.
The entry point adds `src/` to the import path, so editable installation is not
required.

### Legacy entry points

The commands below retain the earlier workflow. They write into the shared
`results/simulation/` directory and can replace prior outputs; use the reproduction
entry point above for new auditable runs.

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Simulation Pipeline

```bash
PYTHONPATH=src .venv/bin/python experiments/run_simulation_framework.py
```

Main outputs are written to `results/simulation/`.

## Evaluate Recovery Quality

```bash
PYTHONPATH=src .venv/bin/python experiments/evaluate_simulation_recovery.py
```

Evaluation outputs:

- `results/simulation/evaluation/recovery_metrics.csv`
- `results/simulation/evaluation/recovery_summary.csv`

## Simulation Ground Truth

Each simulated patient has latent mixture proportions over backbone pathways:

- `theta_B1`, `theta_B2`, `theta_B3`

These are the latent comparison targets used in recovery evaluation. The
dominant-backbone generator is not identical to the fitted transition-mixture
model; see the scientific limits in the reproducibility guide.

## Notes

- The clinical data used in the study are not publicly available due to data protection and institutional restrictions.
- This public repository provides the implementation of the methodology together with simulated examples that reproduce the workflow.
- Step 1 produces a full-cohort descriptive graph and separate training-only
  graphs; the training graphs define the components used for test-patient inference.
- Step 2 applies preprocessing/filtering and estimates transition matrices + `q` on filtered state space.
- Step 3 clusters patients using Step 2 `q` vectors.
