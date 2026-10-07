# Reproducing the simulation study

This workflow reproduces the simulation study of the paper (Supplementary
Section S1, Table S2 and Figure S1) from scratch. It generates the synthetic
cohorts, runs the complete framework (pathway discovery, admixture estimation
and clustering) and evaluates how well the known structure is recovered.
No clinical data are needed.

## Environment and installation

Reference environment:

- Python 3.12.5, macOS 13.5, Apple arm64.
- Graphviz system executable `dot`: 12.1.0 (20240811.2233).
- Python packages, including transitive runtime dependencies:
  [`requirements-reproduction.txt`](../requirements-reproduction.txt).
- The Graphviz executable is a system dependency and is installed separately.
  Verify it with `dot -V`; installing the Python `graphviz` package alone is not enough.

In a new environment, from the repository root:

```bash
python3.12 -m venv .venv-reproduction
.venv-reproduction/bin/python -m pip install -r requirements-reproduction.txt
dot -V
.venv-reproduction/bin/python experiments/reproduce_simulations.py --output-dir results/reproduction/run_001
```

The pinned versions describe macOS; other platforms may need additional
platform-specific dependencies and have not been tested. No R installation is
required. The entry point adds the local `src` directory to its import path, so
`pip install -e .` is not required.

## Configuration and randomness

[`low.json`](../configs/reproducibility/low.json),
[`moderate.json`](../configs/reproducibility/moderate.json) and
[`high.json`](../configs/reproducibility/high.json) are complete configurations
of the three scenarios. Each includes the backbone pathways, the numeric code
mapping, auxiliary and anchor codes, generator perturbations, filtering,
simplification and clustering settings.

| Setting | Low | Moderate | High |
|---|---:|---:|---:|
| Patients per generated cohort | 400 | 400 | 400 |
| Symmetric Dirichlet concentration | 0.8 | 1.0 | 1.2 |
| Backbone source resampling probability | 0.10 | 0.22 | 0.35 |
| Vertical-noise trigger probability | 0.20 | 0.35 | 0.55 |
| Maximum additional codes | 1 | 1 | 2 |
| Horizontal-noise trigger probability | 0.10 | 0.20 | 0.25 |
| Maximum horizontal nodes per insertion | 1 | 1 | 2 |
| Node repetition probability | 0.05 | 0.08 | 0.10 |
| Node skipping probability | 0.02 | 0.05 | 0.06 |
| Dominant-backbone anchor probability | 0.95 | 0.75 | 0.75 |

Vertical noise draws the number of extra codes uniformly from zero up to the
configured maximum once triggered. Repeated identical consecutive blocks are
removed by the generator. Noise and anchor probabilities vary across scenarios
together with the Dirichlet concentration and backbone switching.

All scenarios use generation seed 42, train-test split seeds 35, 123 and 2025,
a training fraction of 0.8, k-means clustering seed 42 with 20 initializations,
and plotting jitter seed 42. The three split seeds resample **one generated
cohort per scenario**; they are not independent cohort replicates.

The entry point runs the analysis in a subprocess with `PYTHONHASHSEED=0`,
single-threaded BLAS/OpenMP, `LOKY_MAX_CPU_COUNT=1` and a headless Matplotlib
backend. Runtime caches are kept inside the run directory. Graph layouts and PDF
timestamps can differ between runs even when all numerical outputs agree.

The `algorithm` section of each configuration records fixed behavior of the
code rather than adjustable parameters: uniform initial admixture weights; EM
with at most 50 iterations and a stopping tolerance of 1e-6 on the weight
change; a likelihood floor of 1e-8; and SLSQP with SciPy defaults (in SciPy
1.17.0: `maxiter=100`, `ftol=1e-6`, numerical Jacobian step
`eps=1.4901161193847656e-8`), with the simplex constraint and [0, 1] bounds
supplied by the estimator.

## Running and inspecting outputs

All three scenarios:

```bash
.venv-reproduction/bin/python experiments/reproduce_simulations.py --output-dir results/reproduction/run_001
```

A single scenario, in a different new directory:

```bash
.venv-reproduction/bin/python experiments/reproduce_simulations.py --scenarios low --output-dir results/reproduction/low_001
```

The output directory must not exist yet. If a run fails, `run_manifest.json`
records the status `failed` and each scenario keeps its log; start again in a
new directory after correcting the cause.

| Output | Content |
|---|---|
| `run_manifest.json` | Completion status, environment, Git commit, source/configuration hashes, output hashes, timestamps |
| `<scenario>/configuration.json` | Exact configuration used for that scenario |
| `<scenario>/run.log` | Analysis and evaluation log |
| `<scenario>/data/simulation_metadata.json` | Generator configuration, including every perturbation |
| `<scenario>/data/simulated_events.csv` | Generated codes and event-block time indices, with latent labels for evaluation |
| `<scenario>/data/simulated_framework_df.csv` | Model input: `patient_num`, `pathOrder`, `states` only; no latent labels |
| `<scenario>/data/simulated_patients_truth.csv` | Patient IDs, latent theta and dominant backbone, used for evaluation |
| `<scenario>/data/state_mapping.csv` | Synthetic code-to-state dictionary |
| `<scenario>/step1/full_data/graph.pdf` | Pathway graph of the full synthetic cohort (as in Figure S1) |
| `<scenario>/step1/training_splits/seed_*/` | Training graph and patient IDs of each split |
| `<scenario>/step2/filtered_transition_matrices/seed_*/` | Component transition matrices and the mapping of q columns to branches |
| `<scenario>/step2/q_vectors_seed_*_*.csv` | Admixture weights of the test patients |
| `<scenario>/step3/seed_*_*/` | Cluster assignments, centers, admixture and cluster figures |
| `<scenario>/evaluation/` | Recovery metrics |
| `metrics_by_split.csv` | Recovery and cluster metrics per scenario, split and optimizer |
| `simulation_summary.csv` | Means and standard deviations over splits, for both optimizers |
| `simulation_table.tex` | Table S2-style table of the means, for both optimizers |
| `table_s2_comparison.csv` | Comparison of this run with the values reported in Table S2 |
| `artifact_map.csv` | Mapping of outputs to the supplementary figures and tables |

## Evaluation

Continuous recovery matches the estimated to the true components by the
permutation with the lowest mean absolute error (MAE). Correlation is the mean
of the component-wise Pearson correlations. ARI compares each patient's dominant
estimated component with the dominant latent backbone; it is not the ARI of the
k-means clusters. The evaluation requires the same number of estimated and true
components and a one-to-one match of patients.

## Implementation notes

- Rare-state filtering in the simulation uses state frequencies of the whole
  synthetic cohort (threshold 5).
- Simulated event blocks are serialized by time and state into unique sequence
  positions.
- Component transition matrices are estimated from the trajectories of the
  patients assigned to each root branch of the pathway graph.
- The generator preferentially follows each patient's dominant backbone, so the
  latent theta is not identical to the parameter of the fitted transition mixture.
- Admixture weights are estimated from each test patient's observed trajectory.
