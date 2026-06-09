# Typical Healthcare Pathways Admixture Modeling

This repository contains a simulation-focused implementation of a 3-step framework:

1. Pathway graph discovery (Step 1)
2. Transition-matrix + admixture estimation (Step 2)
3. Clustering in admixture space (Step 3)

## Reference

This repository contains the simulation-focused implementation of the methodology described in:

Farhadizadeh et al.  
"Typical Healthcare Pathways as a Basis for Admixture Modeling of Patient Trajectories"

## Repository Structure

- `Simulation/Simulation.py`: synthetic cohort generator
- `experiments/run_simulation_framework.py`: end-to-end simulation pipeline runner
- `experiments/evaluate_simulation_recovery.py`: recovery evaluation (`q` vs ground-truth `theta`)
- `src/pathway_admixture/`: core framework modules

## Setup

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

These are the ground-truth admixture targets used in recovery evaluation.

## Notes

- The clinical data used in the study are not publicly available due to data protection and institutional restrictions.
- This public repository provides the implementation of the methodology together with simulated examples that reproduce the workflow.
- Step 1 uses full simulated framework data for branch discovery.
- Step 2 applies preprocessing/filtering and estimates transition matrices + `q` on filtered state space.
- Step 3 clusters patients using Step 2 `q` vectors.
