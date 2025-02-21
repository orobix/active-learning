from typing import Any

import numpy as np
from modAL.utils.selection import multi_argmin
from sklearn.cluster import KMeans


# calculates matrix of distance from centroids
def kmeans_min_distances(K: int, X_pool: np.ndarray, for_each: bool = False, random_state: int = None) -> np.ndarray:
    """Performs a K-Means clustering on the pool of unlabeled instances
    and calculates the distance from the closest centroid for each
    instance in the pool.

    Args:
        K: number of clusters to use in the K-Means clustering
        X_pool: pool of unlabeled
        for_each: specifies whether to calculate distance from closest point or to each centroid
        random_state: optional random state seed to provide to K-Means for reproducibility

    Returns:
        A vector with the distance from the closest centroid for each instance,
        or a matrix of distances from each centroid if for_each is true
    """
    kmeans = KMeans(n_clusters=K, random_state=random_state)
    kmeans.fit(X_pool)

    centroids = kmeans.cluster_centers_

    dist = np.empty((X_pool.shape[0], K))  # distances from each cluster center
    for i in range(0, K):
        dist[:, i] = np.linalg.norm(X_pool - centroids[i], axis=1)
        # euclidean distance

    if for_each is True:
        return dist
    else:
        distmin = dist.min(axis=1)  # vector of minimum distances from the centroids
        return distmin


def query_kmeans(
    classifier: Any, X_pool: np.ndarray, K: int, n_instances: int = 1, random_state: int = None
) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to be queried/added to the labeled set
    via K-Means and returns those instances. The selected instances are the
    ones closest to a centroid.

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        K: number of clusters to use in the K-Means clustering
        n_instances: number of instances to be selected
        random_state: optional random state seed to provide to K-Means for reproducibility

    Returns:
        The indexes of the selected instances, and the selected instances
    """
    utility = kmeans_min_distances(K, X_pool, random_state)

    return multi_argmin(utility, n_instances=n_instances)


def query_kmeans_foreach(
    classifier: Any, X_pool: np.ndarray, K: int, n_instances: int = 1, random_state: int = None
) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to be queried/added to the labeled set
    via K-Means and returns those instances. The selected instances are the
    ones closest to each centroid.

    Notes: if number of points is lower than number of clusters, returns all the points

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        K: number of clusters to use in the K-Means clustering
        n_instances: number of instances to be selected
        random_state: optional random state seed to provide to K-Means for reproducibility

    Returns:
        The indexes of the selected instances, and the selected instances
    """
    if X_pool.shape[0] < n_instances:
        selected_idxs = [*range(X_pool.shape[0])]
        return selected_idxs, X_pool
    if n_instances < K:
        K = n_instances

    dist_matrix = kmeans_min_distances(K, X_pool, for_each=True, random_state=random_state)
    sorted_idxs = np.argsort(dist_matrix, axis=0)
    n_for_each = n_instances // K

    selected_idxs = np.array(sorted_idxs[0 : n_for_each + 1, :].flatten()).tolist()
    selected_idxs = selected_idxs[0:n_instances]
    return selected_idxs, X_pool[selected_idxs, :]
