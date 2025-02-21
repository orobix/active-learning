from typing import Any

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def query_coreset(
    classifier: Any, X_pool: np.ndarray, X_start: np.ndarray | None, n_instances: int = 1
) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to queried/added to be labeled pool with the k-center greedy algorithm.

    Note:
        If X_start == None (no labeled data is available), a random sample of instances is returned.
        Useful for the first iteration of the querying cycle

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        X_start: set with already selected instances
        n_instances: number of instances to be selected

    Returns:
        List of two:
        The indexes of the selected instances, and the selected instances
    """
    if X_start is None:  # random if there is no labeled set
        random_idx = np.random.choice(range(X_pool.shape[0]), size=n_instances, replace=False)
        random_instances = X_pool[random_idx, :]
        return random_idx, random_instances
    else:
        selected_idx = []
        X_pool_original = X_pool
        div_matrix = euclidean_distances(X_pool, X_start)
        for _ in range(n_instances):
            distmin = div_matrix.min(axis=1)

            idx = np.argmax(distmin)
            selected_idx.append(idx)

            X_start = np.vstack([X_start, X_pool[idx]])
            X_pool = np.delete(X_pool, idx, axis=0)

            div_matrix = np.delete(div_matrix, idx, axis=0)

            if X_pool.shape[0] > 0:  # if there are still rows in the pool
                new_col = euclidean_distances(X_pool, X_start[-1, :].reshape(1, -1))
                div_matrix = np.hstack([div_matrix, new_col])

        selected_idx = np.array(selected_idx, dtype=int)
        return selected_idx, X_pool_original[selected_idx, :]
