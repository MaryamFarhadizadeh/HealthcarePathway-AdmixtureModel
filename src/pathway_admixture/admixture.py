"""
Admixture Estimation Module

Contains:
- SLSQP optimizer-based estimator
- EM-based estimator
- Log-likelihood utilities
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =====================================================
# 1️⃣ Log-Likelihood (Shared)
# =====================================================

def sequence_log_likelihood(q, sequence, transition_matrices, state_to_index):
    """
    Returns log P(sequence | q)
    """
    log_prob = 0.0
    K = len(transition_matrices)

    for i in range(len(sequence) - 1):
        s_from = sequence[i]
        s_to = sequence[i + 1]

        from_idx = state_to_index.get(s_from)
        to_idx = state_to_index.get(s_to)

        if from_idx is None or to_idx is None:
            continue

        transition_prob = sum(
            q[k] * transition_matrices[k][from_idx, to_idx]
            for k in range(K)
        )

        transition_prob = max(transition_prob, 1e-8)
        log_prob += np.log(transition_prob)

    return log_prob


def negative_log_likelihood(q, sequence, transition_matrices, state_to_index):
    """
    For optimizer (minimize)
    """
    return -sequence_log_likelihood(
        q, sequence, transition_matrices, state_to_index
    )


def total_log_likelihood(q_df, sequences, transition_matrices, state_to_index):
    """
    Compute total log-likelihood across all patients.
    Useful for comparing EM vs SLSQP.
    """
    total_ll = 0.0

    for i, sequence in enumerate(sequences):
        q = q_df.iloc[i].values
        total_ll += sequence_log_likelihood(
            q, sequence, transition_matrices, state_to_index
        )

    return total_ll


# =====================================================
# 2️⃣ SLSQP Estimator
# =====================================================

def estimate_q_matrix_slsqp(sequences, transition_matrices, state_to_index):
    K = len(transition_matrices)
    individual_qs = []

    for sequence in sequences:

        if len(sequence) < 2:
            individual_qs.append(np.ones(K) / K)
            continue

        initial_q = np.ones(K) / K
        cons = {"type": "eq", "fun": lambda q: np.sum(q) - 1}
        bounds = [(0, 1)] * K

        result = minimize(
            negative_log_likelihood,
            initial_q,
            args=(sequence, transition_matrices, state_to_index),
            constraints=cons,
            bounds=bounds,
            method="SLSQP",
        )

        individual_qs.append(result.x)

    return pd.DataFrame(individual_qs, columns=[f"q{k}" for k in range(K)])


# =====================================================
# 3️⃣ EM Estimator
# =====================================================

def estimate_q_matrix_em(
    sequences,
    transition_matrices,
    state_to_index,
    num_iters=50,
    eps=1e-6,
):

    K = len(transition_matrices)
    all_q = []

    for sequence in sequences:

        if len(sequence) < 2:
            all_q.append(np.ones(K) / K)
            continue

        theta = np.ones(K) / K
        transitions = [
            (sequence[i], sequence[i+1])
            for i in range(len(sequence)-1)
        ]

        for _ in range(num_iters):

            responsibilities = []

            for s_from, s_to in transitions:

                from_idx = state_to_index.get(s_from)
                to_idx = state_to_index.get(s_to)

                if from_idx is None or to_idx is None:
                    responsibilities.append(np.ones(K) / K)
                    continue

                probs = np.array([
                    theta[k] * transition_matrices[k][from_idx, to_idx]
                    for k in range(K)
                ])

                total = probs.sum()

                if total == 0:
                    responsibilities.append(np.ones(K) / K)
                else:
                    responsibilities.append(probs / total)

            responsibilities = np.array(responsibilities)
            theta_new = responsibilities.mean(axis=0)

            if np.linalg.norm(theta - theta_new) < eps:
                break

            theta = theta_new

        all_q.append(theta)

    return pd.DataFrame(all_q, columns=[f"q{k}" for k in range(K)])
