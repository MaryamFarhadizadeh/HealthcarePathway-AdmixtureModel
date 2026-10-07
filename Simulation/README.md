# Applying the Framework to Simulated Data

For new runs, use the [reproducibility guide](../docs/reproducibility.md) and
`experiments/reproduce_simulations.py`. Complete low/moderate/high configurations
are under `configs/reproducibility/`. This entry point saves each scenario
separately and refuses to overwrite existing outputs. The older commands below
write to a shared output directory and are retained for reference.

This guide explains how to run the full pathway-admixture framework on simulated data only.

Current simulation logic: each patient has a Dirichlet-sampled admixture vector (`theta`), a dominant backbone label, and controlled within-trajectory backbone switching.

## What this runs

The simulation pipeline runs:

1. Step 1: graph construction and simplification
2. Step 2: transition matrices and admixture estimation (`EM`, `SLSQP`)
3. Step 3: clustering and admixture plots
4. Recovery evaluation: estimated `q` vs simulated ground-truth `theta`

All simulation outputs are written under:

- `results/simulation/`


## Prerequisites

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you already use the existing project venv, just activate it.

## Configure the simulation

Main generator config:

- `Simulation/Simulation.py`

Key controls:

- `N_PATIENTS`
- `SIMULATION_SCENARIO` (`step1_easy`, `step1_medium`, `step1_hard`)
- `BACKBONE_MODE` (`two_source`, `three_source`)

Framework profile for simulation runs:

- `src/pathway_admixture/settings_profiles.py`
- `SIMULATION_SETTINGS` controls:
  - `min_state_count`
  - `n_clusters`
  - `code_map`
  - simplification thresholds (`importance_keep_threshold`, `prune_abs_threshold`, etc.)

## Run the full simulation framework

From repo root:

```bash
PYTHONPATH=src .venv/bin/python experiments/run_simulation_framework.py
```

## Evaluate recovery quality

Compare estimated admixture (`q`) to simulated truth (`theta`):

```bash
PYTHONPATH=src .venv/bin/python experiments/evaluate_simulation_recovery.py
```

Outputs:

- `results/simulation/evaluation/recovery_metrics.csv`
- `results/simulation/evaluation/recovery_summary.csv`

## Main output files

- `results/simulation/step1/full_data/graph.pdf`
- `results/simulation/step3/cluster_validation_metrics.csv`
- `results/simulation/step3/cluster_validation_em.pdf`
- `results/simulation/step3/cluster_validation_slsqp.pdf`
- `results/simulation/evaluation/recovery_summary.csv`

Per-seed Step 3 figures are in:

- `results/simulation/step3/seed_*_*/`

## Optional outputs (disabled by default)

In `experiments/run_simulation_framework.py`, these defaults keep outputs clean:

- `SAVE_CLUSTER_PNG = False`
- `SAVE_STEP1_SPLIT_PICKLE = False`
- `SAVE_STEP3_CLUSTERED_Q = False`
- `SAVE_RAW_EVENT_TABLES = False`

If you enable these flags, additional intermediate files (PNG plots, pickles, raw event tables, clustered_q CSVs) will be generated.

## Notes on interpretation

- If Step 3 shows only 2 chains, verify `BACKBONE_MODE` and whether Step 1 recovered only 2 top-level branches.
- Recovery metrics are scenario-dependent; a monotone ordering is not guaranteed
  and must be evaluated rather than assumed.
- `results/simulation/README.md` is auto-refreshed by the runner with exact settings used for each run.
