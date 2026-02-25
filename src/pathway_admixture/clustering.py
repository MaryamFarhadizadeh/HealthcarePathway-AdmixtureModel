# """
# Clustering Module

# Provides utilities to cluster admixture coefficient matrices (q-vectors).
# """

# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score


# # =====================================================
# # Main Clustering Function
# # =====================================================

# def cluster_q_vectors(
#     q_df: pd.DataFrame,
#     n_clusters: int = 3,
#     random_state: int = 42,
# ):
#     """
#     Perform KMeans clustering in admixture space.

#     Parameters
#     ----------
#     q_df : pd.DataFrame
#         DataFrame containing q-vectors (columns must start with 'q').
#     n_clusters : int
#         Number of clusters.
#     random_state : int
#         Random seed for reproducibility.

#     Returns
#     -------
#     q_df_clustered : pd.DataFrame
#         Input DataFrame with added 'cluster' column.
#     centers_df : pd.DataFrame
#         Cluster centers in q-space.
#     silhouette : float
#         Silhouette score.
#     """

#     # Extract q-columns
#     q_cols = [c for c in q_df.columns if c.startswith("q")]

#     if len(q_cols) == 0:
#         raise ValueError("No q-columns found in input DataFrame.")

#     X = q_df[q_cols].values

#     # Handle small datasets
#     if X.shape[0] < n_clusters:
#         raise ValueError(
#             f"Number of samples ({X.shape[0]}) smaller than n_clusters ({n_clusters})."
#         )

#     # Fit KMeans
#     kmeans = KMeans(
#         n_clusters=n_clusters,
#         random_state=random_state,
#         n_init=20,
#     )

#     clusters = kmeans.fit_predict(X)

#     # Add cluster labels
#     q_df_clustered = q_df.copy()
#     q_df_clustered["cluster"] = clusters

#     # Compute silhouette score (only valid if >1 cluster)
#     if len(np.unique(clusters)) > 1:
#         silhouette = silhouette_score(X, clusters)
#     else:
#         silhouette = np.nan

#     centers_df = pd.DataFrame(
#         kmeans.cluster_centers_,
#         columns=q_cols,
#     )

#     return q_df_clustered, centers_df, silhouette
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import pandas as pd
import numpy as np


def cluster_q_vectors(q_df, n_clusters=3, random_state=42):

    q_cols = [c for c in q_df.columns if c.startswith("q")]
    if len(q_cols) == 0:
        raise ValueError("No q-columns found in input DataFrame.")

    X = q_df[q_cols].values
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("Input DataFrame has no rows.")

    # KMeans and validation metrics are undefined for impossible/degenerate settings.
    n_unique_points = np.unique(X, axis=0).shape[0]
    effective_clusters = min(n_clusters, n_samples, n_unique_points)

    kmeans = KMeans(
        n_clusters=effective_clusters,
        random_state=random_state,
        n_init=20,
    )

    clusters = kmeans.fit_predict(X)

    q_df_clustered = q_df.copy()
    q_df_clustered["cluster"] = clusters

    n_labels = len(np.unique(clusters))
    if n_labels >= 2 and n_labels < n_samples:
        silhouette = silhouette_score(X, clusters)
        calinski = calinski_harabasz_score(X, clusters)
        davies = davies_bouldin_score(X, clusters)
    else:
        silhouette = np.nan
        calinski = np.nan
        davies = np.nan

    return (
        q_df_clustered,
        pd.DataFrame(kmeans.cluster_centers_, columns=q_cols),
        silhouette,
        calinski,
        davies,
    )
