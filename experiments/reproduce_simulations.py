"""Reproduce the current synthetic workflow in a new, self-contained run directory.

Uses the existing generator and analysis routines; no clinical data are read.
Means over splits are compared with the values reported in Supplementary Table S2.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "reproducibility"
SCENARIOS = ("low", "moderate", "high")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_config(scenario):
    config = json.loads((CONFIG_DIR / f"{scenario}.json").read_text())
    if config["name"] != scenario or config["schema_version"] != 1:
        raise ValueError("Unexpected configuration name or schema")
    # These describe existing algorithm behavior, not new tuning controls.
    expected = {
        "train_fraction": 0.8,
        "filter_scope": "full_cohort",
        "event_representation": "individual_codes_sorted_by_time_then_state",
        "initial_weights": "uniform",
        "likelihood_floor": 1e-8,
        "em_max_iterations": 50,
        "em_weight_tolerance": 1e-6,
        "em_unsupported_transition": "uniform_responsibilities",
        "slsqp_options": "scipy_defaults",
        "clustering_seed": 42,
        "clustering_n_init": 20,
        "plot_jitter_seed": 42,
    }
    if config["algorithm"] != expected:
        raise ValueError("Algorithm settings describe fixed source behavior; update code and audit together")
    return config


def runtime_environment():
    requirements = ROOT / "requirements-reproduction.txt"
    packages = {}
    for line in requirements.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name = line.split("==")[0]
            packages[name] = importlib.metadata.version(name)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "graphviz": subprocess.run(
            ["dot", "-V"], capture_output=True, text=True, check=True
        ).stderr.strip(),
        "environment_variables": {
            name: os.environ.get(name) for name in (
                "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT",
                "MPLBACKEND",
            )
        },
    }


def source_inventory():
    files = sorted((ROOT / "src" / "pathway_admixture").glob("*.py"))
    files += [ROOT / "Simulation" / "Simulation.py"]
    files += [ROOT / "experiments" / name for name in (
        "run_simulation_framework.py", "evaluate_simulation_recovery.py",
        "reproduce_simulations.py",
    )]
    files += sorted(CONFIG_DIR.glob("*.json"))
    files += [ROOT / "requirements-reproduction.txt"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def summarize_run(output, scenarios):
    import pandas as pd

    frames = []
    artifacts = []
    for scenario in scenarios:
        base = output / scenario
        recovery = pd.read_csv(base / "evaluation" / "recovery_metrics.csv")
        cluster = pd.read_csv(base / "step3" / "cluster_validation_metrics.csv")
        merged = recovery.merge(
            cluster[["seed_num", "method", "silhouette", "calinski_harabasz", "davies_bouldin"]],
            left_on=["seed", "method"], right_on=["seed_num", "method"],
            validate="one_to_one", how="left", indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError("Recovery/clustering provenance mismatch")
        config = load_config(scenario)
        if len(merged) != len(config["framework"]["seeds"]) * 2:
            raise ValueError("Incomplete split/optimizer results")
        merged.insert(0, "scenario", scenario)
        frames.append(merged.drop(columns=["_merge", "seed_num"]))
        artifacts.extend([
            {"scenario": scenario, "manuscript_target": "Figure S1 (individual panel)",
             "output": f"{scenario}/step1/full_data/graph.pdf"},
            {"scenario": scenario, "manuscript_target": "Table S2 (new evaluation)",
             "output": f"{scenario}/evaluation/recovery_metrics.csv"},
        ])
    splits = pd.concat(frames, ignore_index=True)
    splits.to_csv(output / "metrics_by_split.csv", index=False)
    metrics = ["global_mae", "mean_component_rmse", "mean_component_corr",
               "ari_latent_backbone", "silhouette", "calinski_harabasz", "davies_bouldin"]
    summary = splits.groupby(["scenario", "method"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "simulation_summary.csv", index=False)

    # Values reported in Supplementary Table S2 (EM), used as a reference; never overwritten.
    reported = json.loads((CONFIG_DIR / "table_s2_reference.json").read_text())
    checks = []
    columns = {"MAE": "global_mae_mean", "Correlation": "mean_component_corr_mean",
               "ARI": "ari_latent_backbone_mean", "Silhouette": "silhouette_mean"}
    for _, row in summary[summary["method"] == "em"].iterrows():
        for metric, column in columns.items():
            old = reported["em"][row["scenario"]][metric]
            new = float(row[column])
            checks.append({"scenario": row["scenario"], "metric": metric,
                           "reported": old, "this_run": new,
                           "difference": new - old,
                           "matches_reported_3dp": round(new, 3) == old})
    pd.DataFrame(checks).to_csv(output / "table_s2_comparison.csv", index=False)
    pd.DataFrame(artifacts).to_csv(output / "artifact_map.csv", index=False)
    lines = [r"% Means over splits from this run.",
             r"\begin{tabular}{llrrrr}",
             r"Scenario & Optimizer & MAE & Correlation & ARI & Silhouette \\", r"\hline"]
    for _, row in summary.iterrows():
        values = " & ".join(f"{row[column]:.3f}" for column in columns.values())
        lines.append(f"{row['scenario'].capitalize()} & {row['method'].upper()} & {values}" + r" \\")
    lines.append(r"\end{tabular}")
    (output / "simulation_table.tex").write_text("\n".join(lines) + "\n")


def run_worker(output, scenarios):
    sys.path.insert(0, str(ROOT / "src"))
    import run_simulation_framework as runner
    import evaluate_simulation_recovery as evaluator
    from pathway_admixture.settings_profiles import FrameworkSettings, SimplificationSettings

    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios, "environment": runtime_environment(),
        "source_sha256": source_inventory(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip() or None,
            }
    write_json(output / "run_manifest.json", manifest)
    try:
        for scenario in scenarios:
            config = load_config(scenario)
            dest = output / scenario
            dest.mkdir()
            write_json(dest / "configuration.json", config)
            settings = dict(config["framework"])
            settings["seeds"] = tuple(settings["seeds"])
            simp = dict(settings["simplification"])
            simp["manual_protected_state_ids"] = tuple(simp["manual_protected_state_ids"])
            settings["simplification"] = SimplificationSettings(**simp)
            runner.SIMULATION_SETTINGS = FrameworkSettings(**settings)
            runner.SEEDS = list(settings["seeds"])
            runner.MIN_STATE_COUNT = settings["min_state_count"]
            runner.N_CLUSTERS = settings["n_clusters"]
            runner.RESULT_BASE = dest
            runner.SAVE_RAW_EVENT_TABLES = True
            runner.SAVE_STEP3_CLUSTERED_Q = True
            print(f"Running {scenario}: 400 patients, three splits, EM and SLSQP", flush=True)
            with (dest / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                runner.main(generator_config=config["generator"])
                evaluator.main(dest)
            print(f"Completed {scenario}", flush=True)
        summarize_run(output, scenarios)
        manifest["status"] = "complete"
        manifest["output_sha256"] = {
            str(p.relative_to(output)): file_hash(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "run_manifest.json" and ".runtime" not in p.parts
        }
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_manifest.json", manifest)
    print(f"Results and Table S2 comparison: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Duplicate scenarios are not allowed")
    if args.worker:
        if os.environ.get("PATHWAY_REPRO_WORKER") != str(output):
            parser.error("--worker is reserved for the controlled subprocess")
        return run_worker(output, args.scenarios)
    for scenario in args.scenarios:
        load_config(scenario)
    if shutil.which("dot") is None:
        parser.error("Graphviz 'dot' is required; see docs/reproducibility.md")
    if output.exists():
        parser.error(f"Output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    env = dict(os.environ)
    env.update({
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "1",
        "MPLCONFIGDIR": str(output / ".runtime" / "matplotlib"),
        "XDG_CACHE_HOME": str(output / ".runtime" / "cache"),
        "PATHWAY_REPRO_WORKER": str(output),
    })
    result = subprocess.run([
        sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
        "--output-dir", str(output), "--scenarios", *args.scenarios,
    ], env=env, cwd=ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
