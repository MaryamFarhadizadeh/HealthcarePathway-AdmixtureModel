"""
Evaluate simulation recovery: estimated q-vectors vs ground-truth theta
and latent backbone.

Reads:
    results/simulation/data/simulated_patients_truth.csv
    results/simulation/step2/q_vectors_seed_*_*.csv

Writes:
    results/simulation/evaluation/recovery_metrics.csv
    results/simulation/evaluation/recovery_summary.csv
"""

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

BASE_DIR = Path(__file__).resolve().parents[1]
SIM_DIR = BASE_DIR / "results" / "simulation"

TRUTH_PATH = SIM_DIR / "data" / "simulated_patients_truth.csv"
STEP2_DIR = SIM_DIR / "step2"
OUT_DIR = SIM_DIR / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# Helper: permutation-invariant matching
# =====================================================

def best_column_matching(est_df: pd.DataFrame, truth_df: pd.DataFrame):
    est_cols = [c for c in est_df.columns if c.startswith("q")]
    truth_cols = [c for c in truth_df.columns if c.startswith("theta_")]

    k = min(len(est_cols), len(truth_cols))
    est_cols = est_cols[:k]
    truth_cols = truth_cols[:k]

    best_perm = None
    best_mae = np.inf

    for perm in permutations(truth_cols, k):
        mae = np.mean(
            np.abs(
                est_df[est_cols].to_numpy()
                - truth_df[list(perm)].to_numpy()
            )
        )
        if mae < best_mae:
            best_mae = mae
            best_perm = perm

    return est_cols, list(best_perm), float(best_mae)


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b))


# =====================================================
# Main Evaluation
# =====================================================

def main():

    if not TRUTH_PATH.exists():
        raise FileNotFoundError(f"Missing truth file: {TRUTH_PATH}")

    truth = pd.read_csv(TRUTH_PATH)
    truth = truth.rename(columns={"patient_id": "patient_num"})
    truth["patient_num"] = truth["patient_num"].astype(str)

    q_files = sorted(STEP2_DIR.glob("q_vectors_seed_*_*.csv"))
    if not q_files:
        raise FileNotFoundError(f"No q-vector files in {STEP2_DIR}")

    rows = []

    for q_path in q_files:

        parts = q_path.stem.split("_")
        if len(parts) < 5:
            continue

        seed = parts[3]
        method = parts[4]

        q = pd.read_csv(q_path)
        q["patient_num"] = q["patient_num"].astype(str)

        merged = q.merge(truth, on="patient_num", how="inner")
        if merged.empty:
            continue

        # -----------------------------
        # Continuous recovery (theta)
        # -----------------------------
        est_cols, truth_cols_matched, global_mae = best_column_matching(
            merged[[c for c in merged.columns if c.startswith("q")]],
            merged[[c for c in merged.columns if c.startswith("theta_")]],
        )

        per_comp = []
        for est_c, tru_c in zip(est_cols, truth_cols_matched):
            mae = float(np.mean(np.abs(merged[est_c] - merged[tru_c])))
            rmse = float(np.sqrt(np.mean((merged[est_c] - merged[tru_c]) ** 2)))
            corr = corr_safe(merged[est_c], merged[tru_c])
            per_comp.append((est_c, tru_c, mae, rmse, corr))

        # -----------------------------
        # Discrete recovery (latent backbone)
        # -----------------------------
        if "latent_backbone" in merged.columns:
            true_labels = merged["latent_backbone"]

            q_cols = est_cols
            predicted_labels = merged[q_cols].idxmax(axis=1)

            ari = adjusted_rand_score(true_labels, predicted_labels)
        else:
            ari = np.nan

        row = {
            "seed": seed,
            "method": method,
            "n_patients": len(merged),
            "k_components": len(est_cols),
            "global_mae": global_mae,
            "mean_component_mae": float(np.mean([x[2] for x in per_comp])),
            "mean_component_rmse": float(np.mean([x[3] for x in per_comp])),
            "mean_component_corr": float(np.nanmean([x[4] for x in per_comp])),
            "ari_latent_backbone": ari,
            "mapping": "; ".join([f"{a}->{b}" for a, b, *_ in per_comp]),
        }

        rows.append(row)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        print("No evaluable q files found.")
        return

    metrics.to_csv(OUT_DIR / "recovery_metrics.csv", index=False)

    summary = (
        metrics.groupby("method", as_index=False)[
            [
                "global_mae",
                "mean_component_mae",
                "mean_component_rmse",
                "mean_component_corr",
                "ari_latent_backbone",
            ]
        ]
        .mean()
        .sort_values("global_mae")
    )

    summary.to_csv(OUT_DIR / "recovery_summary.csv", index=False)

    print(f"\nSaved recovery metrics to {OUT_DIR}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
