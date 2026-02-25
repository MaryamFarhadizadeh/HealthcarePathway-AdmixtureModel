import numpy as np
import pandas as pd

# PREVIOUS VERSION (kept for reference as requested):
# def build_branch_transition_matrices(graph_root, patient_df, state_list=None):
#
#     if state_list is None:
#         state_list = sorted(patient_df["states"].astype(int).unique())
#
#     state_to_idx = {s: i for i, s in enumerate(state_list)}
#     matrices = {}
#
#     for child in graph_root.children:
#
#         branch_patids = set(child.patid)
#         branch_df = patient_df[
#             patient_df["patient_num"].astype(str).isin(branch_patids)
#         ]
#
#         counts = np.zeros((len(state_list), len(state_list)))
#
#         for _, subdf in branch_df.groupby("patient_num"):
#             seq = subdf.sort_values("pathOrder")["states"].astype(int).tolist()
#             for s, s_next in zip(seq[:-1], seq[1:]):
#                 i, j = state_to_idx[s], state_to_idx[s_next]
#                 counts[i, j] += 1
#
#         with np.errstate(divide="ignore", invalid="ignore"):
#             probs = counts / counts.sum(axis=1, keepdims=True)
#             probs[np.isnan(probs)] = 0.0
#
#         branch_label = f"{list(child.states.keys())[0]}_n{len(branch_patids)}"
#         matrices[branch_label] = pd.DataFrame(
#             probs, index=state_list, columns=state_list
#         )
#
#     return matrices


def build_branch_transition_matrices(graph_root, patient_df, state_list):
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    matrices = {}

    for child in graph_root.children:
        branch_patids = set(child.patid)
        branch_df = patient_df[
            patient_df["patient_num"].astype(str).isin(branch_patids)
        ]

        counts = np.zeros((len(state_list), len(state_list)))

        for _, subdf in branch_df.groupby("patient_num"):
            seq = (
                subdf.sort_values("pathOrder")["states"]
                .astype(int)
                .tolist()
            )

            # Keep only filtered states from the provided state_list.
            seq = [s for s in seq if s in state_to_idx]

            for s, s_next in zip(seq[:-1], seq[1:]):
                i = state_to_idx[s]
                j = state_to_idx[s_next]
                counts[i, j] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            probs = counts / counts.sum(axis=1, keepdims=True)
            probs[np.isnan(probs)] = 0.0

        branch_label = f"{list(child.states.keys())[0]}_n{len(branch_patids)}"
        matrices[branch_label] = pd.DataFrame(
            probs,
            index=state_list,
            columns=state_list,
        )

    return matrices
