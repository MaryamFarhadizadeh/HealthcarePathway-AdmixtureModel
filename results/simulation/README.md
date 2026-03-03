# Simulation Results

This folder contains outputs generated from simulated data only.
Real-data outputs remain in `results/step1`, `results/step2`, and `results/step3`.

## Last Updated (UTC)
2026-03-03 12:35:27 UTC

## Active Settings Profile
- profile: `SIMULATION_SETTINGS`
- seeds: `[35, 123, 2025]`
- min_state_count: `5`
- n_clusters: `3`
- use_filtered_step1_results: `False`
- fixed_code_map_enabled: `True`

## Simplification Thresholds
- importance_keep_threshold: `20.0`
- unimportant_threshold: `20.0`
- prune_rel_threshold: `0.1`
- prune_abs_threshold: `3`
- manual_protected_state_ids: `[91, 166]`

## Simulation Code Mapping
`code_map` comes from `SIMULATION_SETTINGS.code_map`.
It is used to convert simulation codes (e.g., `a`, `e`, `i`) to numeric state IDs.

## Output Structure
- `data/`: simulated input exports and state mapping
- `step1/`: train/test splits, graphs, transition-split artifacts
- `step2/`: filtered transition matrices and q-vectors
- `step3/`: clustering outputs, plots, and validation summary
- `evaluation/`: recovery metrics (produced by `experiments/evaluate_simulation_recovery.py`)

## Simulation Generator Metadata
- BACKBONE_MODE: `three_source`
- CFG: `ScenarioConfig(p_add_vertical_noise=0.35, max_vertical_noise=1, p_add_horizontal_noise=0.2, max_horizontal_nodes=1, p_repeat_node=0.08, p_skip_node=0.05, p_add_anchor=0.75, major_alpha=8.0, minor_alpha=1.2)`
- N_PATIENTS: `400`
- RANDOM_SEED: `42`
- SIMULATION_SCENARIO: `step1_medium`
