# src/pathway_admixture/preprocessing.py

import pandas as pd


# -------------------------------------------------
# Global State Filtering
# -------------------------------------------------
def filter_rare_states(a_df, min_count=10):
    """
    Remove states that occur fewer than min_count times globally.
    """
    state_counts = a_df["states"].value_counts()
    states_to_keep = state_counts[state_counts >= min_count].index
    df_filtered = a_df[a_df["states"].isin(states_to_keep)].copy()
    return df_filtered


# -------------------------------------------------
# Apply filtering to a specific split
# -------------------------------------------------
def apply_filter_to_split(df_filtered, train_ids, test_ids):
    """
    Intersect filtered data with train/test patient IDs.
    Recompute pathOrder after filtering.
    """

    train_filtered = df_filtered[
        df_filtered["patient_num"].isin(train_ids)
    ].copy()

    test_filtered = df_filtered[
        df_filtered["patient_num"].isin(test_ids)
    ].copy()

    # Recompute pathOrder
    train_filtered = train_filtered.sort_values(
        ["patient_num", "pathOrder"]
    )
    train_filtered["pathOrder"] = (
        train_filtered.groupby("patient_num").cumcount() + 1
    )

    test_filtered = test_filtered.sort_values(
        ["patient_num", "pathOrder"]
    )
    test_filtered["pathOrder"] = (
        test_filtered.groupby("patient_num").cumcount() + 1
    )

    return train_filtered, test_filtered


# -------------------------------------------------
# Convert dataframe to sequences
# -------------------------------------------------
def make_sequences(df):
    """
    Convert patient dataframe into list of integer sequences.
    """
    sequences = []

    for _, group in df.groupby("patient_num"):
        sequence = group.sort_values("pathOrder")["states"].tolist()
        sequences.append([int(float(s)) for s in sequence])

    return sequences
