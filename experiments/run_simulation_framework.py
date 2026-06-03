"""
Run the full pathway admixture framework on simulated data.

Outputs are written under:
    results/simulation/
so real-data results remain unchanged.
"""

import pickle
import random
import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import graphviz
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

# Stabilize BLAS / joblib threading
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathway_admixture.step1_pathway import (
    train_test_split,
    build_patient_objects,
    build_prefix_tree,
    run_simplification_pipeline,
    render_train_graph,
)
from pathway_admixture.transitions import build_branch_transition_matrices
from pathway_admixture.preprocessing import (
    filter_rare_states,
    apply_filter_to_split,
    make_sequences,
)
from pathway_admixture.admixture import (
    estimate_q_matrix_em,
    estimate_q_matrix_slsqp,
)
from pathway_admixture.clustering import cluster_q_vectors
from pathway_admixture.settings_profiles import SIMULATION_SETTINGS


# =====================================================
# Configuration
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]
SIM_SCRIPT = BASE_DIR / "Simulation" / "Simulation.py"
RESULT_BASE = BASE_DIR / "results" / "simulation"

SEEDS = list(SIMULATION_SETTINGS.seeds)
MIN_STATE_COUNT = SIMULATION_SETTINGS.min_state_count
N_CLUSTERS = SIMULATION_SETTINGS.n_clusters

CUSTOM_CLUSTER_COLORS = ["#6C8EBF", "#9BBFA6", "#D9A66B"]
CUSTOM_PATHWAY_COLORS = ["#809FD1", "#8787BF", "#B8C7D9"]
SIMULATION_STEP1_PALETTE = [
    "#8787BF", "#8FC4D1", "#8EEDC3", "#B8C7D9", "#9CF6FB",
    "#4682B4", "#809FD1", "#D1ABCF", "#A0D6B4", "#D9A6A6",
    "#9BBFA6", "#D9C39B", "#A8B8D8", "#C0D8E8", "#B9D9D9",
    "#AEC6CF", "#C3B1E1", "#FFD6A5", "#BDE0FE", "#E2F0CB",
]

# Output policy for repository cleanliness.
SAVE_CLUSTER_PNG = False
SAVE_STEP1_SPLIT_PICKLE = False
SAVE_STEP3_CLUSTERED_Q = False
SAVE_RAW_EVENT_TABLES = False


# =====================================================
# Metadata Writer
# =====================================================

def write_simulation_readme(result_base: Path, simulation_metadata=None):
    readme_path = result_base / "README.md"
    s = SIMULATION_SETTINGS
    simp = s.simplification

    readme_text = f"""# Simulation Results

Generated using a dominant-backbone + controlled-switching simulation design.

## Last Updated (UTC)
{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Active Settings Profile
- seeds: {list(s.seeds)}
- min_state_count: {s.min_state_count}
- n_clusters: {s.n_clusters}

## Simplification Thresholds
- importance_keep_threshold: {simp.importance_keep_threshold}
- unimportant_threshold: {simp.unimportant_threshold}
- prune_rel_threshold: {simp.prune_rel_threshold}
- prune_abs_threshold: {simp.prune_abs_threshold}

## Output Structure
- data/
- step1/
- step2/
- step3/
- evaluation/
"""

    if simulation_metadata:
        readme_text += "\n## Simulation Generator Metadata\n"
        for key in sorted(simulation_metadata.keys()):
            readme_text += f"- {key}: {simulation_metadata[key]}\n"

    readme_path.write_text(readme_text)


# =====================================================
# Load Simulation
# =====================================================

def load_and_simulate_events():
    import importlib.util

    spec = importlib.util.spec_from_file_location("simulation_module", SIM_SCRIPT)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)

    random.seed(sim.RANDOM_SEED)
    cohort = sim.simulate_cohort(sim.N_PATIENTS)

    metadata = {
        "N_PATIENTS": getattr(sim, "N_PATIENTS", None),
        "RANDOM_SEED": getattr(sim, "RANDOM_SEED", None),
        "SIMULATION_SCENARIO": getattr(sim, "SIMULATION_SCENARIO", None),
        "BACKBONE_MODE": getattr(sim, "BACKBONE_MODE", None),
        "DIRICHLET_ALPHA": getattr(sim.CFG, "dirichlet_alpha", None),
        "BACKBONE_CODES": sorted(
            {
                code
                for path in getattr(sim, "BACKBONES", {}).values()
                for code in path
            }
        ),
    }

    rows = []
    patient_rows = []

    for patient in cohort:
        latent_backbone = patient.get("latent_backbone", patient.get("backbone"))
        patient_rows.append(
            {
                "patient_id": patient["patient_id"],
                "latent_backbone": latent_backbone,
                "theta_B1": patient.get("theta_B1"),
                "theta_B2": patient.get("theta_B2"),
                "theta_B3": patient.get("theta_B3"),
            }
        )

        t = 0
        for block in patient["trajectory"]:
            for code in block:
                rows.append(
                    {
                        "patient_id": patient["patient_id"],
                        "time": t,
                        "code": code,
                        "latent_backbone": latent_backbone,
                    }
                )
            t += 1

    return pd.DataFrame(rows), pd.DataFrame(patient_rows), metadata


# =====================================================
# Framework Conversion
# =====================================================

def to_framework_df(df_events):
    sim_code_map = SIMULATION_SETTINGS.code_map

    if sim_code_map is None:
        codes = sorted(df_events["code"].astype(str).unique())
        code_to_state = {c: i + 1 for i, c in enumerate(codes)}
    else:
        code_to_state = dict(sim_code_map)

    df = df_events.copy()
    df["patient_num"] = df["patient_id"]
    df["states"] = df["code"].map(code_to_state).astype(int)
    df = df.sort_values(["patient_num", "time", "states"])
    df["pathOrder"] = df.groupby("patient_num").cumcount() + 1

    return df[["patient_num", "pathOrder", "states"]], code_to_state


def build_simulation_step1_color_map(state_values):
    state_values = sorted({int(s) for s in state_values})
    color_map = {"0": "#8787BF"}
    for i, state in enumerate(state_values):
        if state == 0:
            continue
        color_map[str(state)] = SIMULATION_STEP1_PALETTE[i % len(SIMULATION_STEP1_PALETTE)]
    if 166 in state_values:
        color_map["166"] = "#5477A7"
    return color_map


def _map_state_label_text(label_text, state_id_to_code, highlight_codes=None):
    # Example inputs:
    #   "119"
    #   "10, 119"
    highlight_codes = set(highlight_codes or [])
    out = []
    for token in label_text.split(","):
        s = token.strip()
        if s.lstrip("-").isdigit():
            key = int(s)
            code = str(state_id_to_code.get(key, s))
            if code in highlight_codes:
                out.append(f"<FONT COLOR='red'>{code}</FONT>")
            else:
                out.append(code)
        else:
            if s in highlight_codes:
                out.append(f"<FONT COLOR='red'>{s}</FONT>")
            else:
                out.append(s)
    return ", ".join(out)


def render_train_graph_with_code_labels(
    mystart,
    istates,
    output_path_no_suffix,
    color_map,
    state_id_to_code,
    highlight_codes=None,
):
    dot = render_train_graph(
        mystart,
        istates,
        show_legend=False,
        color_map=color_map,
    )

    # Replace only the first line in node labels:
    # <<B>{STATE_LABEL}<BR/>(n = ...)</B>>
    pattern = re.compile(r"(<<B>)([^<]+)(<BR/>\(n = \d+\)</B>>)")

    def repl(match):
        prefix, state_label, suffix = match.groups()
        mapped = _map_state_label_text(
            state_label,
            state_id_to_code,
            highlight_codes=highlight_codes,
        )
        return f"{prefix}{mapped}{suffix}"

    src = pattern.sub(repl, dot.source)
    graphviz.Source(src).render(output_path_no_suffix, format="pdf", cleanup=True)


def make_cluster_scatter(q_clustered, q_cols, output_path_png, output_path_pdf):
    if len(q_cols) < 2:
        return

    sns.set_theme(style="white", font_scale=1.1)
    clusters = sorted(q_clustered["cluster"].unique())
    color_map = {cl: CUSTOM_CLUSTER_COLORS[i] for i, cl in enumerate(clusters)}

    plt.figure(figsize=(7, 6))
    for cl in clusters:
        sub = q_clustered[q_clustered["cluster"] == cl]
        plt.scatter(
            sub[q_cols[0]],
            sub[q_cols[1]],
            s=90,
            alpha=0.95,
            edgecolor="white",
            linewidth=0.6,
            color=color_map[cl],
            label=f"Cluster {int(cl) + 1}",
            zorder=2,
        )

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r"Admixture weight $q_{i1}$")
    plt.ylabel(r"Admixture weight $q_{i2}$")
    plt.title("Admixture space", weight="bold", pad=10)
    plt.legend(title="Patient cluster", frameon=False)
    sns.despine()
    plt.tight_layout()
    if SAVE_CLUSTER_PNG:
        plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, dpi=400, bbox_inches="tight")
    plt.close()


def make_cluster_scatter_jitter(q_clustered, q_cols, output_path_png, output_path_pdf):
    if len(q_cols) < 2:
        return

    rng = np.random.default_rng(42)
    jitter = rng.normal(0.0, 5e-3, size=(len(q_clustered), 2))
    jittered = q_clustered[q_cols[:2]].values + jitter
    clusters = sorted(q_clustered["cluster"].unique())
    color_map = {cl: CUSTOM_CLUSTER_COLORS[i] for i, cl in enumerate(clusters)}
    point_colors = q_clustered["cluster"].map(color_map)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        jittered[:, 0],
        jittered[:, 1],
        c=point_colors,
        s=65,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color_map[cl], markersize=8,
            label=f"Cluster {int(cl) + 1}",
        )
        for cl in clusters
    ]
    plt.legend(handles=handles, title="Cluster", frameon=True)
    plt.xlabel(r"Admixture weight $q_{i1}$")
    plt.ylabel(r"Admixture weight $q_{i2}$")
    plt.title("Admixture space for Axes")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if SAVE_CLUSTER_PNG:
        plt.savefig(output_path_png, dpi=300)
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.close()


def make_cluster_size_with_q(q_clustered, output_path_pdf):
    q_cols = sorted(
        [c for c in q_clustered.columns if c.startswith("q")],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )
    if not q_cols:
        return

    cluster_counts = q_clustered["cluster"].value_counts().sort_index()
    cluster_means = q_clustered.groupby("cluster")[q_cols].mean()
    colors = CUSTOM_PATHWAY_COLORS[: len(q_cols)]
    cluster_labels = [str(int(c) + 1) for c in cluster_counts.index]

    fig, ax = plt.subplots(figsize=(7, 6))
    bottoms = np.zeros(len(cluster_counts))
    for i, (col, color) in enumerate(zip(q_cols, colors)):
        values = cluster_means[col].values * cluster_counts.values
        ax.bar(
            cluster_labels,
            values,
            bottom=bottoms,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=f"Pathway {i+1}",
        )
        bottoms += values

    ax.set_xlabel("Patient cluster")
    ax.set_ylabel("Number of patients")
    ax.set_title("Pathway-specific model contributions", weight="bold", pad=10)
    for i, count in enumerate(cluster_counts.values):
        ax.text(i, count + max(cluster_counts.values) * 0.02, str(count), ha="center", fontweight="bold")

    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="white", label=f"Pathway {i+1}")
        for i in range(len(q_cols))
    ]
    ax.legend(handles=legend_handles, title="Pathway", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path_pdf, dpi=400, bbox_inches="tight")
    plt.close()


def make_admixture_barplot(q_clustered, output_path_pdf):
    q_cols = sorted(
        [c for c in q_clustered.columns if c.startswith("q")],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )
    if not q_cols:
        return

    data_q = q_clustered[q_cols].copy()
    for col in q_cols:
        data_q[col] = pd.to_numeric(data_q[col], errors="coerce")

    data_q["dominant"] = data_q[q_cols].idxmax(axis=1)
    data_q["dominant_val"] = data_q[q_cols].max(axis=1)
    data_q = data_q.sort_values(["dominant", "dominant_val"], ascending=[True, False])
    data_q = data_q.drop(columns=["dominant", "dominant_val"]).reset_index(drop=True)

    n = len(data_q)
    k = len(q_cols)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
    for i in range(n):
        bottom = 0.0
        for j in range(k):
            val = data_q.iloc[i, j]
            ax.bar(i, val, bottom=bottom, width=1.0, color=CUSTOM_PATHWAY_COLORS[j % len(CUSTOM_PATHWAY_COLORS)], linewidth=0)
            bottom += val

    ax.set_ylabel("Admixture weight")
    ax.set_xlabel("Patients (ordered by dominant pathway contribution)")
    ax.set_ylim(0, 1)
    ax.set_xlim([-1, n])
    ax.set_title("Admixture proportions across the three pathway-specific models")
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Patch(facecolor=CUSTOM_PATHWAY_COLORS[i % len(CUSTOM_PATHWAY_COLORS)], edgecolor="none", label=f"Pathway {i+1}")
        for i in range(k)
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path_pdf, dpi=300)
    plt.close()


def plot_metric_with_na(ax, df, metric_col, title, color, ylim=None):
    labels = df["seed"].tolist()
    raw_vals = df[metric_col]
    vals = raw_vals.fillna(0.0).to_numpy()
    x = np.arange(len(labels))

    ax.bar(x, vals, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Train-test split")
    ax.set_ylabel("Validation score")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(vals) > 0:
        mean_val = np.nanmean(vals)
        ax.axhline(mean_val, linestyle="--", linewidth=1.2, color="black", alpha=0.6)
    for idx, raw_val in enumerate(raw_vals):
        if pd.isna(raw_val):
            y_text = (ylim[1] * 0.05) if ylim else 0.05
            ax.text(idx, y_text, "N/A", ha="center", fontsize=9)
    sns.despine(ax=ax)


def save_validation_figure(cluster_quality_df, out_path, panel_prefix=False):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    t1 = "A. Silhouette Score" if panel_prefix else "Silhouette Score"
    t2 = "B. Calinski-Harabasz Index" if panel_prefix else "Calinski-Harabasz Index"
    t3 = "C. Davies-Bouldin Index" if panel_prefix else "Davies-Bouldin Index"

    plot_metric_with_na(axes[0], cluster_quality_df, "silhouette", t1, "#E4EEFB", ylim=(0, 1))
    plot_metric_with_na(axes[1], cluster_quality_df, "calinski_harabasz", t2, "#B6BDC8")
    plot_metric_with_na(axes[2], cluster_quality_df, "davies_bouldin", t3, "#97BEE6")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()


# =====================================================
# Main
# =====================================================

def main():

    step1_dir = RESULT_BASE / "step1" / "training_splits"
    step1_full_dir = RESULT_BASE / "step1" / "full_data"
    step2_dir = RESULT_BASE / "step2"
    step3_dir = RESULT_BASE / "step3"
    data_dir = RESULT_BASE / "data"

    for p in [step1_dir, step1_full_dir, step2_dir, step3_dir, data_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("Generating simulated data...")
    df_events, df_patients_truth, sim_meta = load_and_simulate_events()
    write_simulation_readme(RESULT_BASE, sim_meta)

    df_framework, code_to_state = to_framework_df(df_events)
    state_id_to_code = {v: k for k, v in code_to_state.items()}
    backbone_code_set = set(sim_meta.get("BACKBONE_CODES", []))

    if SAVE_RAW_EVENT_TABLES:
        df_events.to_csv(data_dir / "simulated_events.csv", index=False)
        df_framework.to_csv(data_dir / "simulated_framework_df.csv", index=False)
    # Keep truth table because recovery evaluation consumes it.
    df_patients_truth.to_csv(data_dir / "simulated_patients_truth.csv", index=False)

    simulation_step1_color_map = build_simulation_step1_color_map(df_framework["states"].astype(int).tolist())

    print("Step 1 full-data graph...")
    patvec_full = build_patient_objects(df_framework)
    mystart_full = build_prefix_tree(patvec_full)
    mystart_full, istates_full = run_simplification_pipeline(
        mystart_full,
        importance_keep_threshold=SIMULATION_SETTINGS.simplification.importance_keep_threshold,
        unimportant_threshold=SIMULATION_SETTINGS.simplification.unimportant_threshold,
        prune_rel_threshold=SIMULATION_SETTINGS.simplification.prune_rel_threshold,
        prune_abs_threshold=SIMULATION_SETTINGS.simplification.prune_abs_threshold,
        manual_protected_state_ids=SIMULATION_SETTINGS.simplification.manual_protected_state_ids,
    )
    render_train_graph_with_code_labels(
        mystart_full,
        istates_full,
        step1_full_dir / "graph",
        simulation_step1_color_map,
        state_id_to_code,
        highlight_codes=backbone_code_set,
    )

    print("Step 1...")
    split_results = []

    for seed in SEEDS:
        print(f"  Seed {seed}")

        a_df_train, a_df_test = train_test_split(df_framework, seed=seed)
        train_ids = set(a_df_train["patient_num"].unique())
        test_ids = set(a_df_test["patient_num"].unique())

        patvec_train = build_patient_objects(a_df_train)
        mystart_train = build_prefix_tree(patvec_train)

        mystart_train, istates_train = run_simplification_pipeline(
            mystart_train,
            importance_keep_threshold=SIMULATION_SETTINGS.simplification.importance_keep_threshold,
            unimportant_threshold=SIMULATION_SETTINGS.simplification.unimportant_threshold,
            prune_rel_threshold=SIMULATION_SETTINGS.simplification.prune_rel_threshold,
            prune_abs_threshold=SIMULATION_SETTINGS.simplification.prune_abs_threshold,
            manual_protected_state_ids=SIMULATION_SETTINGS.simplification.manual_protected_state_ids,
        )

        branch_matrices = build_branch_transition_matrices(
            mystart_train,
            a_df_train,
            state_list=sorted(a_df_train["states"].unique()),
        )
        seed_dir = step1_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        render_train_graph_with_code_labels(
            mystart_train,
            istates_train,
            seed_dir / "graph",
            simulation_step1_color_map,
            state_id_to_code,
            highlight_codes=backbone_code_set,
        )

        split_results.append(
            {
                "seed": seed,
                "graph": mystart_train,
                "branch_matrices": branch_matrices,
                "train_ids": train_ids,
                "test_ids": test_ids,
            }
        )

    if SAVE_STEP1_SPLIT_PICKLE:
        with open(step1_dir / "split_results.pkl", "wb") as f:
            pickle.dump(split_results, f)

    print("Step 2...")
    df_filtered = filter_rare_states(df_framework, min_count=MIN_STATE_COUNT)
    all_states = sorted(df_filtered["states"].unique())

    for split in split_results:
        seed = split["seed"]
        print(f"  Seed {seed}")

        train_filtered, test_filtered = apply_filter_to_split(
            df_filtered,
            split["train_ids"],
            split["test_ids"],
        )

        filtered_branch_matrices = build_branch_transition_matrices(
            split["graph"],
            train_filtered,
            state_list=all_states,
        )

        test_sequences = make_sequences(test_filtered)
        test_patient_ids = sorted(test_filtered["patient_num"].unique())
        transition_matrices = [m.to_numpy() for m in filtered_branch_matrices.values()]
        state_to_index = {s: i for i, s in enumerate(all_states)}

        q_em = estimate_q_matrix_em(test_sequences, transition_matrices, state_to_index)
        q_slsqp = estimate_q_matrix_slsqp(test_sequences, transition_matrices, state_to_index)

        q_em.insert(0, "patient_num", test_patient_ids)
        q_slsqp.insert(0, "patient_num", test_patient_ids)

        q_em.to_csv(step2_dir / f"q_vectors_seed_{seed}_em.csv", index=False)
        q_slsqp.to_csv(step2_dir / f"q_vectors_seed_{seed}_slsqp.csv", index=False)

    print("Step 3...")
    cluster_quality_rows = []

    q_files = sorted(step2_dir.glob("q_vectors_seed_*_*.csv"))

    for q_path in q_files:
        parts = q_path.stem.split("_")
        seed = parts[3]
        method = parts[4]

        q_df = pd.read_csv(q_path)

        q_clustered, centers_df, silhouette, calinski, davies = cluster_q_vectors(
            q_df, n_clusters=N_CLUSTERS
        )
        seed_result_dir = step3_dir / f"seed_{seed}_{method}"
        seed_result_dir.mkdir(parents=True, exist_ok=True)
        if SAVE_STEP3_CLUSTERED_Q:
            q_clustered.to_csv(seed_result_dir / "clustered_q.csv", index=False)
        centers_df.to_csv(seed_result_dir / "cluster_centers.csv", index=False)
        with open(seed_result_dir / "silhouette.txt", "w") as f:
            f.write(f"{silhouette:.6f}")

        q_cols = [c for c in q_df.columns if c.startswith("q")]
        make_cluster_scatter(
            q_clustered,
            q_cols,
            seed_result_dir / "cluster_plot.png",
            seed_result_dir / "cluster_plot.pdf",
        )
        make_cluster_scatter_jitter(
            q_clustered,
            q_cols,
            seed_result_dir / "cluster_plot_jitter.png",
            seed_result_dir / "cluster_plot_jitter.pdf",
        )
        make_cluster_size_with_q(q_clustered, seed_result_dir / "cluster_sizes_with_q.pdf")
        make_admixture_barplot(q_clustered, seed_result_dir / "admixture_barplot.pdf")

        cluster_quality_rows.append(
            {
                "seed": f"{seed}_{method}",
                "seed_num": int(seed),
                "method": method,
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
            }
        )

    if cluster_quality_rows:
        cluster_quality_df = pd.DataFrame(cluster_quality_rows)
        cluster_quality_df = cluster_quality_df.sort_values(["method", "seed_num"]).reset_index(drop=True)
        cluster_quality_df.to_csv(step3_dir / "cluster_validation_metrics.csv", index=False)

        save_validation_figure(cluster_quality_df, step3_dir / "cluster_validation.pdf", panel_prefix=True)
        for method_name in ["em", "slsqp"]:
            method_df = cluster_quality_df[cluster_quality_df["method"] == method_name].copy()
            if method_df.empty:
                continue
            method_df["seed"] = method_df["seed_num"].astype(str)
            save_validation_figure(
                method_df,
                step3_dir / f"cluster_validation_{method_name}.pdf",
                panel_prefix=False,
            )

    print("\nSimulation framework run completed successfully.")


if __name__ == "__main__":
    main()
